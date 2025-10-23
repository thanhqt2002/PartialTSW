import math
import torch

class TWConcurrentLines():
    def __init__(self, 
                 ntrees=1000, 
                 nlines=5, 
                 p=2,
                 delta=2, 
                 mass_division='distance_based', 
                 ftype='linear',
                 d=3,
                 degree=3,
                 radius=2.0,
                 pow_beta=1,
                 device="cuda"):
        """
        Class for computing the Generalized Tree Wasserstein distance between two distributions.
        Args:
            ntrees (int): Number of trees.
            nlines (int): Number of lines per tree.
            p (int): Level of the norm.
            delta (float): Negative inverse of softmax temperature for distance-based mass division.
            mass_division (str): How to divide the mass, one of 'uniform', 'distance_based'.
            ftype (str): Type of defining function ('linear', 'poly', 'circular', 'augmented', 'pow').
            d (int): Dimension of the input space (used if ftype='poly' or ftype='augmented').
            degree (int): Degree of the polynomial (used if ftype='poly').
            radius (float): Radius of the circle (used if ftype='circular').
            pow_beta (float): Contribution between linear and pow (used if ftype='pow').
            uot_ulambda (float): lambda value in close-form for regularized EPT.
            uot_beta (float): beta value in close-form for regularized EPT.
            uot_alpha (float): alpha value in close-form for regularized EPT.
            uot_wr (float): wr value in close-form for regularized EPT. 
            device (str): Device to run the code, follows torch convention (default is "cuda").
        """
        self.ntrees = ntrees
        self.device = device
        self.nlines = nlines
        self.p = p
        self.delta = delta
        self.mass_division = mass_division
        
        self.ftype = ftype
        self.d = d
        self.degree = degree
        self.radius = radius
        self.pow_beta = pow_beta

        if self.ftype == 'pow':
            self.mapping = lambda X : X + self.pow_beta * X ** 3
            
            self.dtheta = d
        elif self.ftype == 'poly':
            self.powers = TWConcurrentLines.get_powers(d, degree).to(device)
            self.mapping = lambda X : TWConcurrentLines.poly_features(X, self.powers)

            self.dtheta = self.powers.shape[1]
        else:
            self.dtheta = d

        assert self.mass_division in ['uniform', 'distance_based'], \
            "Invalid mass division. Must be one of 'uniform', 'distance_based'"
        assert self.ftype in ['linear', 'poly', 'circular', 'pow', 'circular_concentric'], \
            "Invalid ftype. Must be one of 'linear', 'poly', 'circular', 'pow', 'circular_concentric'"

    def __call__(self, X, Y, theta, intercept, total_mass_X=1, total_mass_Y=1):
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        # Get mass
        N, dn = X.shape
        M, dm = Y.shape
        assert dn == dm

        return self.compute_tw(X, Y, theta, intercept,total_mass_X, total_mass_Y)

    def compute_tw(self, X, Y, theta, intercept, total_mass_X=1, total_mass_Y=1):
        if self.ftype == 'poly' or self.ftype == 'pow':
            X = self.mapping(X)
            Y = self.mapping(Y)

        mass_X, axis_coordinate_X = self.project(X, theta=theta, intercept=intercept)
        mass_X = mass_X * total_mass_X
        mass_Y, axis_coordinate_Y = self.project(Y, theta=theta, intercept=intercept)
        mass_Y = mass_Y * total_mass_Y
        tw = self.tw_concurrent_lines(mass_X, mass_Y, axis_coordinate_X, axis_coordinate_Y)[0]
        return tw

    def tw_concurrent_lines(self, mass_X, mass_Y, axis_coordinate_X, axis_coordinate_Y):
        """
        Args:
            mass_X: (num_trees, num_lines, num_points)
            mass_Y: (num_trees, num_lines, num_points)
            axis_coordinate_X: (num_trees, num_lines, num_points)
            axis_coordinate_Y: (num_trees, num_lines, num_points)
        """

        combined_axis_coordinate = torch.cat((axis_coordinate_X, axis_coordinate_Y), dim=2)
        mass_XY = torch.cat((mass_X, -mass_Y), dim=2)

        coord_sorted, indices = torch.sort(combined_axis_coordinate, dim=-1)
        num_trees, num_lines = mass_XY.shape[0], mass_XY.shape[1]

        # generate the cumulative sum of mass
        sub_mass = torch.gather(mass_XY, 2, indices)
        sub_mass_target_cumsum = torch.cumsum(sub_mass, dim=-1)

        if self.ftype != 'circular_concentric' and self.ftype != 'circular':
            sub_mass_right_cumsum = sub_mass + torch.sum(sub_mass, dim=-1, keepdim=True) - sub_mass_target_cumsum
            sub_mass_target_cumsum = torch.where(coord_sorted > 0, sub_mass_right_cumsum, sub_mass_target_cumsum)

        ### compute edge length
        if self.ftype != 'circular_concentric':

            # add root to the sorted coordinate by insert 0 to the first position <= 0
            root = torch.zeros(num_trees, num_lines, 1, device=self.device) 
            root_indices = torch.searchsorted(coord_sorted, root)
            coord_sorted_with_root = torch.zeros(num_trees, num_lines, mass_XY.shape[2] + 1, device=self.device)
            # distribute other points to the correct position
            edge_mask = torch.ones_like(coord_sorted_with_root, dtype=torch.bool)
            edge_mask.scatter_(2, root_indices, False)
            coord_sorted_with_root[edge_mask] = coord_sorted.flatten()
            # compute edge length
            edge_length = coord_sorted_with_root[:, :, 1:] - coord_sorted_with_root[:, :, :-1]
        else:
            prepend_tensor = torch.zeros((num_trees, 1, 1), device=coord_sorted.device)
            coord_sorted_with_prepend = torch.cat([prepend_tensor, coord_sorted], dim=-1)
            edge_length = coord_sorted_with_prepend[..., 1:] - coord_sorted_with_prepend[..., :-1]

        # compute TW distance
        subtract_mass = (torch.abs(sub_mass_target_cumsum) ** self.p) * edge_length
        subtract_mass_sum = torch.sum(subtract_mass, dim=[-1,-2])

        # if unbalanced:
        #     uT = torch.sum(mass_X, dim=(-2, -1))
        #     vT = torch.sum(mass_Y, dim=(-2, -1))
            
        #     # higher beta means more regularization and less allowed unbalanced transport
        #     scalar = self.uot_beta * self.uot_ulambda / 2
        #     reg = (self.uot_wr + scalar - self.uot_alpha) * torch.abs(uT - vT) - scalar * (uT + vT)
        #     subtract_mass_sum += reg
        
        tw = torch.mean(subtract_mass_sum) ** (1/self.p)
        return tw, sub_mass_target_cumsum, edge_length

    def project(self, input, theta, intercept):
        N = input.shape[0]
        num_trees = theta.shape[0]
        num_lines = theta.shape[1]

        # all lines has the same point which is root
        input_translated = (input - intercept) #[T,B,D]
        if self.ftype == 'circular':
            axis_coordinate = torch.norm(input_translated.unsqueeze(1) - theta.unsqueeze(2) * self.radius, dim=-1)
        elif self.ftype == 'circular_concentric':
            axis_coordinate = torch.norm(input_translated, dim=-1).unsqueeze(1) # [T,1,B]
        else:
            axis_coordinate = torch.matmul(theta, input_translated.transpose(1, 2))
        
        if self.mass_division == 'uniform':
            mass_input = mass * torch.ones((num_trees, num_lines, N), device=self.device) / (N * num_lines)
        elif self.mass_division =='distance_based':
            if self.ftype == 'circular_concentric':
                input_projected_translated = torch.einsum('tlb,tld->tlbd', axis_coordinate.repeat(1, num_lines, 1), theta)
            else: 
                input_projected_translated = torch.einsum('tlb,tld->tlbd', axis_coordinate, theta)
            dist = (torch.norm(input_projected_translated - input_translated.unsqueeze(1), dim = -1))
            weight = -self.delta*dist
            mass_input = torch.softmax(weight, dim=-2)/N
        
            # if uot_topk != self.nlines and uot_topk > 0:
            #     # Get top-k indices and create a mask
            #     _, indices = torch.topk(mass_input, uot_topk, dim=-2)
            #     mask = torch.zeros_like(mass_input)
            #     mask.scatter_(dim=-2, index=indices, value=1)
            #     mass_input *= mask
        return mass_input, axis_coordinate

    @staticmethod
    def get_power_generator(dim, degree):
        '''
        This function calculates the powers of a homogeneous polynomial
        e.g.

        list(get_powers(dim=2,degree=3))
        [(0, 3), (1, 2), (2, 1), (3, 0)]

        list(get_powers(dim=3,degree=2))
        [(0, 0, 2), (0, 1, 1), (0, 2, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0)]
        '''
        if dim == 1:
            yield (degree,)
        else:
            for value in range(degree + 1):
                for permutation in TWConcurrentLines.get_power_generator(dim - 1,degree - value):
                    yield (value,) + permutation

    @staticmethod
    def get_powers(dim, degree):
        powers = TWConcurrentLines.get_power_generator(dim, degree)
        return torch.stack([torch.tensor(p) for p in powers], dim=1)         
    
    @staticmethod
    def homopoly(dim, degree):
        '''
        calculates the number of elements in a homogeneous polynomial
        '''
        return int(
            math.factorial(degree+dim-1) /
            (math.factorial(degree) * math.factorial(dim-1))
        )

    @staticmethod
    def poly_features(input, powers):
        return torch.pow(input.unsqueeze(-1), powers.unsqueeze(0)).prod(dim=1)

    @staticmethod
    def calculate_geometric_median(X, tol=1e-7, max_iter=20, epsilon=1e-12):
        """
        Calculates the geometric median for a set of points using Weiszfeld's algorithm.
        Operates on torch.Tensor.

        The geometric median is the point minimizing the sum of Euclidean distances
        to the points in the set X.

        Args:
            X (torch.Tensor): Tensor of points, shape (N, dim).
            tol (float): Tolerance for convergence. The algorithm stops when the
                         change in the median estimate is less than this value. Default 1e-7.
            max_iter (int): Maximum number of iterations allowed. Default 100.
            epsilon (float): A small value added to denominators to prevent division by zero,
                             especially when the estimate gets very close to an input point. Default 1e-12.

        Returns:
            torch.Tensor: The geometric median coordinates (dim,). Returns a tensor of NaNs
                          if the input tensor X has N=0 points.
        """
        if not isinstance(X, torch.Tensor):
             raise TypeError("Input must be a torch.Tensor")

        device = X.device
        dtype = X.dtype

        if X.dim() != 2:
            raise ValueError(f"Input tensor X must be 2D (N, dim), but got shape {X.shape}")

        N, dim = X.shape

        if N == 0:
            # Return NaN tensor with the expected dimension
            return torch.full((dim,), float('nan'), device=device, dtype=dtype)
        if N == 1:
            # If only one point, it's the median
            return X[0]

        # Initial guess: Centroid (mean of points)
        y = torch.mean(X, dim=0) # Shape (dim,)

        for iteration in range(max_iter):
            # Calculate distances from current estimate y to all points X_i
            # diffs = X - y results in shape (N, dim)
            distances = torch.linalg.norm(X - y, dim=1) # Shape (N,)

            # --- Handling Coincidence: Check if y is very close to any X_i ---
            coincident_mask = distances < epsilon
            if torch.any(coincident_mask):
                # Find the index of the first point y is close to
                coincident_idx = torch.where(coincident_mask)[0][0].item() # Get scalar index
                point_j = X[coincident_idx] # This point might be the median

                # Calculate sum of normalized vectors from point_j to all *other* points
                other_indices = torch.arange(N, device=device) != coincident_idx
                if not torch.any(other_indices):
                     # This happens only if N=1 initially (already handled)
                     # or if all points collapsed to one (point_j is the median)
                     return point_j

                other_points = X[other_indices]
                diff_vectors = point_j - other_points # Shape (N-1, dim)
                norms = torch.linalg.norm(diff_vectors, dim=1) # Shape (N-1,)

                # Filter out potential zero norms (if point_j is identical to another point)
                valid_norms_mask = norms > epsilon
                if not torch.any(valid_norms_mask):
                    # All other points are identical to point_j. So point_j is the median.
                    return point_j

                # Calculate norm only for points distinct from point_j
                normalized_vectors = diff_vectors[valid_norms_mask] / norms[valid_norms_mask].unsqueeze(1)
                sum_normalized_vectors = torch.sum(normalized_vectors, dim=0) # Shape (dim,)
                norm_sum_vectors = torch.linalg.norm(sum_normalized_vectors) # Scalar

                # Check the median condition: || Sum[(xj-xi)/||xj-xi||] || <= 1
                # If condition holds, point_j is the geometric median
                if norm_sum_vectors <= 1.0 + epsilon: # Use tolerance
                    # print(f"Median found at input point index {coincident_idx} in {iteration+1} iterations.")
                    return point_j
                # Else: point_j is not the median. The standard update below will move y away.

            # --- Regular Weiszfeld Update Step ---
            # Calculate weights: w_i = 1 / ||y - x_i||
            # Add epsilon to prevent division by zero.
            weights = 1.0 / (distances + epsilon)
            weights = weights.unsqueeze(1)  # Shape (N, 1) for broadcasting

            # Calculate sum of weights
            sum_weights = torch.sum(weights) # Scalar

            # Check if sum_weights is too small (highly unlikely with epsilon)
            if sum_weights < epsilon:
                print(f"Warning: Sum of weights is near zero at iteration {iteration}. Returning current estimate.")
                return y # Return current estimate

            # Calculate the weighted sum of points: Sum[w_i * x_i]
            weighted_sum = torch.sum(X * weights, dim=0) # Shape (dim,)

            # Calculate the next estimate for the median
            y_next = weighted_sum / sum_weights # Shape (dim,)

            # Check for convergence: if ||y_next - y|| is small enough
            delta = torch.linalg.norm(y_next - y) # Scalar

            # Update the estimate for the next iteration
            y = y_next

            if delta < tol:
                # print(f"Converged in {iteration+1} iterations.")
                return y

        # If the loop finishes without converging
        # print(f"Warning: Geometric median calculation did not converge within {max_iter} iterations.")
        # print(f"Final delta: {delta}, Tolerance: {tol}")
        return y # Return the last estimate

def svd_orthogonalize(matrix):
    U, _, _ = torch.linalg.svd(matrix, full_matrices=False)
    return U

def generate_trees_frames(ntrees, nlines, d, mean=128, std=0.1, device='cuda', gen_mode='gaussian_raw'):    
    # random root as gaussian distribution with given mean and std
    assert gen_mode in ['gaussian_raw', 'gaussian_orthogonal'], "Invalid gen_mode"
    root = torch.randn(ntrees, 1, d, device=device) * std + mean
    intercept = root
    
    if gen_mode == 'gaussian_raw':
        theta = torch.randn(ntrees, nlines, d, device=device)
        theta = theta / torch.norm(theta, dim=-1, keepdim=True)
    elif gen_mode == 'gaussian_orthogonal':
        assert nlines <= d, "Support dim should be greater than or equal to number of lines to generate orthogonal lines"
        theta = torch.randn(ntrees, d, nlines, device=device)
        theta = svd_orthogonalize(theta)
        theta = theta.transpose(-2, -1)
    
    return theta, intercept

if __name__ == "__main__":
    from torch.profiler import profile, record_function, ProfilerActivity
    # N = 32 * 32
    # M = 32 * 32
    # dn = dm = 128
    # ntrees = 2048
    # nlines = 2
    
    N = 5
    M = 5
    dn = dm = 3
    ntrees = 7
    nlines = 2
    
    TW_obj = torch.compile(TWConcurrentLines(ntrees=ntrees, mass_division='distance_based'))
    theta, intercept = generate_trees_frames(ntrees, nlines, dn, gen_mode="gaussian_orthogonal")
    X = torch.rand(N, dn).to("cuda")
    Y = torch.rand(M, dm).to("cuda")
    TW_obj(X, Y, theta, intercept)

    TW_obj = torch.compile(TWConcurrentLines(ntrees=ntrees, mass_division='distance_based', ftype='circular', radius=2))
    theta, intercept = generate_trees_frames(ntrees, nlines, dn, gen_mode="gaussian_orthogonal")
    X = torch.rand(N, dn).to("cuda")
    Y = torch.rand(M, dm).to("cuda")
    TW_obj(X, Y, theta, intercept)

    TW_obj = torch.compile(TWConcurrentLines(ntrees=ntrees, mass_division='distance_based', ftype='circular_concentric'))
    theta, intercept = generate_trees_frames(ntrees, nlines, dn, gen_mode="gaussian_orthogonal")
    X = torch.rand(N, dn).to("cuda")
    Y = torch.rand(M, dm).to("cuda")
    TW_obj(X, Y, theta, intercept)

    dtheta = TWConcurrentLines.homopoly(dn, 3)
    TW_obj = torch.compile(TWConcurrentLines(ntrees=ntrees, mass_division='distance_based', ftype='poly', d=dn, degree=3))
    theta, intercept = generate_trees_frames(ntrees, nlines, dtheta, gen_mode="gaussian_orthogonal")
    X = torch.rand(N, dn).to("cuda")
    Y = torch.rand(M, dm).to("cuda")
    TW_obj(X, Y, theta, intercept)

    # theta, intercept = generate_trees_frames(ntrees, nlines, dn, dn, gen_mode="gaussian_orthogonal")
    # X = torch.rand(N, dn).to("cuda")
    # Y = torch.rand(M, dm).to("cuda")
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
    #     tw = TW_obj(X, Y, theta, intercept)
    #     TW_obj(X, Y, theta, intercept)

    TW_obj = torch.compile(TWConcurrentLines(ntrees=ntrees, mass_division='distance_based'))
    theta, intercept = generate_trees_frames(ntrees, nlines, dn)
    X = torch.rand(N, dn).to("cuda")
    Y = torch.rand(M, dm).to("cuda")
    TW_obj(X, Y, theta, intercept)

    # prof.export_chrome_trace("trace_concurrent.json")
    # with open("profile_result_concurrent.txt", "w") as f:
    #     table_str = prof.key_averages().table(sort_by="cpu_time_total", top_level_events_only=True)
    #     f.write(table_str)
    #     print(table_str)