# contains the code for the Tree-Sliced Wasserstein with nonlinear projection
import math
import torch

try:
    from .utils import generate_trees_frames
except ImportError:
    from utils import generate_trees_frames

class TSW():
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
            ftype (str): Type of defining function ('linear', 'poly', 'circular', 'pow').
            d (int): Dimension of the input space (used if ftype='poly').
            degree (int): Degree of the polynomial (used if ftype='poly' or ftype='pow').
            pow_beta (float): Contribution between linear and pow (used if ftype='pow').
            radius (float): Radius of the circle (used if ftype='circular').
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
            self.mapping = lambda X : X + self.pow_beta * X ** self.degree
            
            self.dtheta = d
        elif self.ftype == 'poly':
            self.powers = TSW.get_powers(d, degree).to(device)
            self.mapping = lambda X : TSW.poly_features(X, self.powers)

            self.dtheta = self.powers.shape[1]
        else:
            self.dtheta = d

        assert self.mass_division in ['uniform', 'distance_based'], \
            "Invalid mass division. Must be one of 'uniform', 'distance_based'"
        assert self.ftype in ['linear', 'poly', 'circular', 'pow', 'circular_r0'], \
            "Invalid ftype. Must be one of 'linear', 'poly', 'circular', 'pow', 'circular_r0'"

    def __call__(self, X, Y, theta, intercept, optimize_mapping=False):
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        # Get mass
        N, dn = X.shape
        M, dm = Y.shape
        assert dn == dm and M == N

        if optimize_mapping:
            self.optimize_mapping(X, Y, theta, intercept)
        else:
            return self.compute_tw(X, Y, theta, intercept)

    def compute_tw(self, X, Y, theta, intercept):
        if self.ftype == 'poly' or self.ftype == 'pow':
            X = self.mapping(X)
            Y = self.mapping(Y)

        mass_X, axis_coordinate_X = self.project(X, theta=theta, intercept=intercept)
        mass_Y, axis_coordinate_Y = self.project(Y, theta=theta, intercept=intercept)
        tw = self.tw_concurrent_lines(mass_X, axis_coordinate_X, mass_Y, axis_coordinate_Y)[0]
        return tw

    def optimize_mapping(self, X_original, Y_original, theta, intercept):
        X_detach = X_original.detach()
        Y_detach = Y_original.detach()
        for _ in range(self.num_iter):
            # generate input mapping
            X_mapped = self.mapping(X_detach)
            Y_mapped = self.mapping(Y_detach)

            # compute tsw
            mass_X, axis_coordinate_X = self.project(X_mapped, theta=theta, intercept=intercept)
            mass_Y, axis_coordinate_Y = self.project(Y_mapped, theta=theta, intercept=intercept)
            negative_tw = -self.tw_concurrent_lines(mass_X, axis_coordinate_X, mass_Y, axis_coordinate_Y)[0]
            # optimize mapping
            reg = self.alpha * (torch.norm(X_mapped, dim=-1).mean() + torch.norm(Y_mapped, dim=-1).mean())
            self.mapping_optimizer.zero_grad()
            (reg + negative_tw).backward()
            self.mapping_optimizer.step()

    def tw_concurrent_lines(self, mass_X, axis_coordinate_X, mass_Y, axis_coordinate_Y):
        """
        Args:
            mass_X: (num_trees, num_lines, num_points)
            axis_coordinate_X: (num_trees, num_lines, num_points)
            mass_Y: (num_trees, num_lines, num_points)
            axis_coordinate_Y: (num_trees, num_lines, num_points)
        """
        combined_axis_coordinate = torch.cat((axis_coordinate_X, axis_coordinate_Y), dim=2)
        mass_XY = torch.cat((mass_X, -mass_Y), dim=2)
        
        coord_sorted, indices = torch.sort(combined_axis_coordinate, dim=-1)
        num_trees, num_lines = mass_XY.shape[0], mass_XY.shape[1]

        # generate the cumulative sum of mass
        sub_mass = torch.gather(mass_XY, 2, indices)
        sub_mass_target_cumsum = torch.cumsum(sub_mass, dim=-1)
        
        ### compute edge length
        if self.ftype != 'circular_r0' and self.ftype != 'circular':
            sub_mass_right_cumsum = sub_mass + torch.sum(sub_mass, dim=-1, keepdim=True) - sub_mass_target_cumsum
            mask_right = torch.nonzero(coord_sorted > 0, as_tuple=True)
            sub_mass_target_cumsum[mask_right] = sub_mass_right_cumsum[mask_right]

            # add root to the sorted coordinate by insert 0 to the first position <= 0
            root = torch.zeros(num_trees, num_lines, 1, device=self.device) 
            root_indices = torch.searchsorted(coord_sorted, root)
            coord_sorted_with_root = torch.zeros(num_trees, num_lines, mass_XY.shape[2] + 1, device=self.device)
            # distribute other points to the correct position
            edge_mask = torch.ones_like(coord_sorted_with_root, dtype=torch.bool)
            edge_mask.scatter_(2, root_indices, False)
            coord_sorted_with_root[edge_mask] = coord_sorted.flatten()
            
            edge_length = coord_sorted_with_root[:, :, 1:] - coord_sorted_with_root[:, :, :-1]
        else:
            prepend_tensor = torch.zeros((num_trees, coord_sorted.shape[1], 1), device=coord_sorted.device)
            coord_sorted_with_prepend = torch.cat([prepend_tensor, coord_sorted], dim=-1)
            edge_length = coord_sorted_with_prepend[..., 1:] - coord_sorted_with_prepend[..., :-1]

        # compute TW distance
        subtract_mass = (torch.abs(sub_mass_target_cumsum) ** self.p) * edge_length
        subtract_mass_sum = torch.sum(subtract_mass, dim=[-1,-2])
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
        elif self.ftype == 'circular_r0':
            axis_coordinate = torch.norm(input_translated, dim=-1).unsqueeze(1) # [T,1,B]
        else:
            axis_coordinate = torch.matmul(theta, input_translated.transpose(1, 2))
        
        if self.mass_division == 'uniform':
            mass_input = torch.ones((num_trees, num_lines, N), device=self.device) / (N * num_lines)
        elif self.mass_division =='distance_based':
            if self.ftype == 'circular_r0':
                input_projected_translated = torch.einsum('tlb,tld->tlbd', axis_coordinate.repeat(1, num_lines, 1), theta)
            else: 
                input_projected_translated = torch.einsum('tlb,tld->tlbd', axis_coordinate, theta)
            dist = (torch.norm(input_projected_translated - input_translated.unsqueeze(1), dim = -1))
            weight = -self.delta*dist
            mass_input = torch.softmax(weight, dim=-2)/N
        
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
                for permutation in TSW.get_power_generator(dim - 1,degree - value):
                    yield (value,) + permutation

    @staticmethod
    def get_powers(dim, degree):
        powers = TSW.get_power_generator(dim, degree)
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

if __name__ == "__main__":    
    N = 5
    M = 5
    dn = dm = 3
    ntrees = 7
    nlines = 2
    
    TW_obj = torch.compile(TSW(ntrees=ntrees, mass_division='distance_based'))
    theta, intercept = generate_trees_frames(ntrees, nlines, dn, gen_mode="gaussian_orthogonal")
    X = torch.rand(N, dn).to("cuda")
    Y = torch.rand(M, dm).to("cuda")
    TW_obj(X, Y, theta, intercept)

    TW_obj = torch.compile(TSW(ntrees=ntrees, mass_division='distance_based', ftype='circular', radius=2))
    theta, intercept = generate_trees_frames(ntrees, nlines, dn, gen_mode="gaussian_raw")
    X = torch.rand(N, dn).to("cuda")
    Y = torch.rand(M, dm).to("cuda")
    TW_obj(X, Y, theta, intercept)

    TW_obj = torch.compile(TSW(ntrees=ntrees, mass_division='distance_based', ftype='circular_r0'))
    theta, intercept = generate_trees_frames(ntrees, nlines, dn, gen_mode="gaussian_raw")
    X = torch.rand(N, dn).to("cuda")
    Y = torch.rand(M, dm).to("cuda")
    TW_obj(X, Y, theta, intercept)

    TW_obj = torch.compile(TSW(ntrees=ntrees, mass_division='distance_based', ftype='pow', d=dn, degree=3))
    theta, intercept = generate_trees_frames(ntrees, nlines, dn, gen_mode="gaussian_raw")
    X = torch.rand(N, dn).to("cuda")
    Y = torch.rand(M, dm).to("cuda")
    TW_obj(X, Y, theta, intercept)

    dtheta = TSW.homopoly(dn, 3)
    TW_obj = torch.compile(TSW(ntrees=ntrees, mass_division='distance_based', ftype='poly', d=dn, degree=3))
    theta, intercept = generate_trees_frames(ntrees, nlines, dtheta, gen_mode="gaussian_raw")
    X = torch.rand(N, dn).to("cuda")
    Y = torch.rand(M, dm).to("cuda")
    TW_obj(X, Y, theta, intercept)