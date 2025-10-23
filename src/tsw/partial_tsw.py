import torch

try:
    from .utils import generate_trees_frames
except ImportError:
    from utils import generate_trees_frames

class PartialTSW():
    def __init__(self, 
                 ntrees=1000, 
                 nlines=5, 
                 p=2,
                 delta=2, 
                 mass_division='distance_based', 
                 device="cuda"):
        """
        Class for computing the Partial Tree-Sliced Wasserstein distance between two distributions.
        Args:
            ntrees (int): Number of trees.
            nlines (int): Number of lines per tree.
            p (int): Level of the norm.
            delta (float): Negative inverse of softmax temperature for distance-based mass division.
            mass_division (str): How to divide the mass, one of 'uniform', 'distance_based'.
            device (str): Device to run the code, follows torch convention (default is "cuda").
        """
        self.ntrees = ntrees
        self.device = device
        self.nlines = nlines
        self.p = p
        self.delta = delta
        self.mass_division = mass_division

        assert self.mass_division in ['uniform', 'distance_based'], \
            "Invalid mass division. Must be one of 'uniform', 'distance_based'"

    def __call__(self, X, Y, theta, intercept, total_mass_X=None, total_mass_Y=None):
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        # Get mass
        N, dn = X.shape
        M, dm = Y.shape
        assert dn == dm

        if total_mass_X is None:
            total_mass_X = torch.tensor(1.0, device=self.device)
        if total_mass_Y is None:
            total_mass_Y = torch.tensor(1.0, device=self.device)

        return self.compute_tw(X, Y, theta, intercept, total_mass_X, total_mass_Y)

    def compute_tw(self, X, Y, theta, intercept, total_mass_X, total_mass_Y):
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

        sub_mass_right_cumsum = sub_mass + torch.sum(sub_mass, dim=-1, keepdim=True) - sub_mass_target_cumsum
        sub_mass_target_cumsum = torch.where(coord_sorted > 0, sub_mass_right_cumsum, sub_mass_target_cumsum)

        ### compute edge length
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
        axis_coordinate = torch.matmul(theta, input_translated.transpose(1, 2))
        
        if self.mass_division == 'uniform':
            mass_input = torch.ones((num_trees, num_lines, N), device=self.device) / (N * num_lines)
        elif self.mass_division =='distance_based':
            input_projected_translated = torch.einsum('tlb,tld->tlbd', axis_coordinate, theta)
            dist = (torch.norm(input_projected_translated - input_translated.unsqueeze(1), dim = -1))
            weight = -self.delta*dist
            mass_input = torch.softmax(weight, dim=-2)/N
        
        return mass_input, axis_coordinate


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    
    N = 5
    M = 5
    dn = dm = 3
    ntrees = 7
    nlines = 2
    
    # Test with gaussian_orthogonal
    TSW_obj = torch.compile(PartialTSW(ntrees=ntrees, mass_division='distance_based'))
    theta, intercept = generate_trees_frames(ntrees, nlines, dn, gen_mode="gaussian_orthogonal")
    X = torch.rand(N, dn).to("cuda")
    Y = torch.rand(M, dm).to("cuda")
    result = TSW_obj(X, Y, theta, intercept)
    print(f"Test 1 (gaussian_orthogonal): {result}")

    # Test with gaussian_raw
    TSW_obj = torch.compile(PartialTSW(ntrees=ntrees, mass_division='distance_based'))
    theta, intercept = generate_trees_frames(ntrees, nlines, dn, gen_mode="gaussian_raw")
    X = torch.rand(N, dn).to("cuda")
    Y = torch.rand(M, dm).to("cuda")
    result = TSW_obj(X, Y, theta, intercept)
    print(f"Test 2 (gaussian_raw): {result}")

    # Test with partial transport (unbalanced masses)
    # Use tensors instead of scalars for total_mass to ensure proper gradient flow,
    # avoid type conversion overhead, and maintain consistency with PyTorch's computation graph
    TSW_obj = torch.compile(PartialTSW(ntrees=ntrees, mass_division='distance_based'))
    theta, intercept = generate_trees_frames(ntrees, nlines, dn, gen_mode="gaussian_orthogonal")
    X = torch.rand(N, dn).to("cuda")
    Y = torch.rand(M, dm).to("cuda")
    total_mass_X = torch.tensor(0.8, device="cuda")
    total_mass_Y = torch.tensor(0.6, device="cuda")
    result = TSW_obj(X, Y, theta, intercept, total_mass_X=total_mass_X, total_mass_Y=total_mass_Y)
    print(f"Test 3 (partial transport): {result}")
