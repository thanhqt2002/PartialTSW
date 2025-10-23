import torch
import torch.nn as nn
import numpy as np
import gdown
import os
import ot
import matplotlib.pyplot as plt
import sys
sys.path.append("./ALAE")
from alae_ffhq_inference import load_model, decode

# Imports needed for the new function
import concurrent.futures
from PIL import Image
from tqdm import tqdm

class ResidualMLP(nn.Module):
    """
    An MLP with residual connections, adjustable depth, and a scaled residual
    pathway.

    Args:
        dim (int): Input/output feature dimension.
        num_hidden_blocks (int): # intermediate hidden blocks (≥ 0).
                                 Total hidden layers = num_hidden_blocks + 1 (initial expansion).
        hidden_dim_multiplier (int): Hidden size = dim × multiplier.
        dropout_rate (float): Dropout probability.
        res_scale (float): Initial scale factor applied to the residual branch.
        learnable_scale (bool): If True, `res_scale` becomes a learnable parameter
                                (LayerScale). If False, it stays fixed.
    """
    def __init__(
        self,
        dim: int = 512,
        num_hidden_blocks: int = 2,
        hidden_dim_multiplier: int = 2,
        dropout_rate: float = 0.1,
        res_scale: float = 0.1,
        learnable_scale: bool = True,
    ):
        super().__init__()
        if num_hidden_blocks < 0:
            raise ValueError("num_hidden_blocks must be non-negative")

        self.input_dim = dim
        hidden_dim = dim * hidden_dim_multiplier

        layers = []

        # 1. Initial Expansion Block
        layers.extend([
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        ])

        # 2. Intermediate Hidden Blocks
        for _ in range(num_hidden_blocks):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            ])

        # 3. Output Projection Layer
        layers.append(nn.Linear(hidden_dim, dim))

        # Assemble the core MLP
        self.mlp_core = nn.Sequential(*layers)

        # 4. Residual scale (fixed or learnable)
        if learnable_scale:
            self.res_scale = nn.Parameter(torch.tensor(res_scale, dtype=torch.float32))
        else:
            self.register_buffer("res_scale", torch.tensor(res_scale, dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            z (Tensor): Shape (batch_size, dim).

        Returns:
            Tensor: Same shape as `z`.
        """
        if z.shape[-1] != self.input_dim:
            raise ValueError(
                f"Input last dimension ({z.shape[-1]}) "
                f"does not match expected ({self.input_dim})"
            )

        residual = self.mlp_core(z)
        return z + self.res_scale * residual

class ModulationBlock(nn.Module):
    """
    A block performing internal modulation (FiLM-like).
    It processes an input tensor through a main path and uses a controller
    path (operating on the same input) to generate scale (gamma) and shift (beta)
    parameters, which modulate the output of the main path.
    Includes a residual connection within the block.
    """
    def __init__(self, embed_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # Controller MLP: Predicts gamma and beta
        # Simple single linear layer for controller
        self.controller = nn.Linear(embed_dim, embed_dim * 2)

        # Main processing path
        self.main_path = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
            # You could add more layers here if desired
        )

        # Initialize controller weights for near-identity modulation initially
        # Initialize gamma weights near zero -> gamma near 1 after adding 1
        # Initialize beta weights near zero -> beta near 0
        nn.init.zeros_(self.controller.weight)
        nn.init.zeros_(self.controller.bias)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x shape: (batch_size, embed_dim)
        main_out = self.main_path(x)

        # Get modulation parameters gamma, beta
        # modulation shape: (batch_size, embed_dim * 2)
        modulation = self.controller(x)
        # Split into gamma and beta, each (batch_size, embed_dim)
        # Add 1 to gamma so initial modulation is multiplication by ~1
        gamma = modulation[:, :self.embed_dim] + 1
        beta = modulation[:, self.embed_dim:]

        # Apply modulation
        modulated_out = gamma * main_out + beta

        # Residual connection within the block
        block_output = x + modulated_out
        return block_output


class FiLMedResidualMLP(nn.Module):
    """
    An MLP using internal modulation blocks (FiLM-like) and an overall
    residual connection z' = z + D_core(z).
    """
    def __init__(self,
                 dim: int = 512,
                 num_modulation_blocks: int = 3, # Adjustable depth
                 hidden_dim_multiplier: int = 2,
                 dropout_rate: float = 0.1):
        """
        Args:
            dim: Input and output dimension.
            num_modulation_blocks: Number of internal modulation blocks.
            hidden_dim_multiplier: Factor for hidden dimension (dim * multiplier).
            dropout_rate: Dropout probability.
        """
        super().__init__()
        if num_modulation_blocks < 1:
             raise ValueError("num_modulation_blocks must be at least 1.")

        self.input_dim = dim
        hidden_dim = dim * hidden_dim_multiplier

        # Initial projection layer
        self.initial_layer = nn.Linear(dim, hidden_dim)

        # Sequence of modulation blocks
        self.modulation_blocks = nn.ModuleList(
            [ModulationBlock(hidden_dim, dropout_rate) for _ in range(num_modulation_blocks)]
        )

        # Final normalization and projection back to input dimension
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.final_projection = nn.Linear(hidden_dim, dim)

        # Initialize final projection weights/bias to zero for residual connection
        nn.init.zeros_(self.final_projection.weight)
        nn.init.zeros_(self.final_projection.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: z' = z + D_core(z)
        """
        if z.shape[-1] != self.input_dim:
             raise ValueError(f"Input tensor last dimension ({z.shape[-1]}) "
                              f"does not match model's expected input dimension ({self.input_dim})")

        # Calculate the core transformation D_core(z)
        hidden = self.initial_layer(z)
        for block in self.modulation_blocks:
            hidden = block(hidden)
        hidden_norm = self.final_norm(hidden)
        residual_update = self.final_projection(hidden_norm)

        # Overall residual connection
        z_prime = z + residual_update
        return z_prime

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.layer3 = nn.Linear(128, 1)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = torch.sigmoid(self.layer3(out))
        return out

def decode_and_plot(T, latent_to_map, inp_images, number_of_samples, device):
    model = load_model("ALAE/configs/ffhq.yaml", training_artifacts_dir="ALAE/training_artifacts/ffhq/").to(device)
    mapped_all = []
    with torch.no_grad():
        for k in range(number_of_samples):
            mapped = T(latent_to_map)
            mapped_all.append(mapped)

    mapped = torch.stack(mapped_all, dim=1)

    decoded_all = []
    with torch.no_grad():
        for k in range(number_of_samples):
            decoded_img = decode(model, mapped[:, k])
            decoded_img = ((decoded_img * 0.5 + 0.5) * 255).type(torch.long).clamp(0, 255).cpu().type(torch.uint8).permute(0, 2, 3, 1).numpy()
            decoded_all.append(decoded_img)

    decoded_all = np.stack(decoded_all, axis=1)
    
    fig, axes = plt.subplots(number_of_samples+1, latent_to_map.shape[0], figsize=(latent_to_map.shape[0], number_of_samples+1), dpi=200)

    for i, ind in enumerate(range(latent_to_map.shape[0])):
        axes[0, i].imshow(inp_images[ind])
        for k in range(number_of_samples):
            axes[k+1, i].imshow(decoded_all[ind, k])

            axes[k+1, i].get_xaxis().set_visible(False)
            axes[k+1, i].set_yticks([])

        axes[0, i].get_xaxis().set_visible(False)
        axes[0, i].set_yticks([])

    fig.tight_layout(pad=0.05)
    
    return fig, axes

def download_data():
    os.makedirs("data", exist_ok=True)
    urls = {
        "data/age.npy": "https://drive.google.com/uc?id=1Vi6NzxCsS23GBNq48E-97Z9UuIuNaxPJ",
        "data/gender.npy": "https://drive.google.com/uc?id=1SEdsmQGL3mOok1CPTBEfc_O1750fGRtf",
        "data/latents.npy": "https://drive.google.com/uc?id=1ENhiTRsHtSjIjoRu1xYprcpNd8M9aVu8",
        "data/test_images.npy": "https://drive.google.com/uc?id=1SjBWWlPjq-dxX4kxzW-Zn3iUR3po8Z0i",
    }

    for name, url in urls.items():
        gdown.download(url, os.path.join(f"{name}"), quiet=False)

def load_inds(input_data, target_data, train_size, test_size):
    gender = np.load("data/gender.npy")
    age = np.load("data/age.npy")
    train_gender, test_gender = gender[:train_size], gender[train_size:]
    train_age, test_age = age[:train_size], age[train_size:]
    
    if input_data == "MAN":
        x_inds_train = np.arange(train_size)[(train_gender == "male").reshape(-1)]
        x_inds_test = np.arange(test_size)[(test_gender == "male").reshape(-1)]
    elif input_data == "WOMAN":
        x_inds_train = np.arange(train_size)[(train_gender == "female").reshape(-1)]
        x_inds_test = np.arange(test_size)[(test_gender == "female").reshape(-1)]
    elif input_data == "ADULT":
        x_inds_train = np.arange(train_size)[
            (train_age > 44).reshape(-1)*(train_age != -1).reshape(-1)
        ]
        x_inds_test = np.arange(test_size)[
            (test_age > 44).reshape(-1)*(test_age != -1).reshape(-1)
        ]
    elif input_data == "YOUNG":
        x_inds_train = np.arange(train_size)[
            ((train_age > 16) & (train_age <= 44)).reshape(-1)*(train_age != -1).reshape(-1)
        ]
        x_inds_test = np.arange(test_size)[
            ((test_age > 16) & (test_age <= 44)).reshape(-1)*(test_age != -1).reshape(-1)
        ]

    if target_data == "MAN":
        y_inds_train = np.arange(train_size)[(train_gender == "male").reshape(-1)]
        y_inds_test = np.arange(test_size)[(test_gender == "male").reshape(-1)]
    elif target_data == "WOMAN":
        y_inds_train = np.arange(train_size)[(train_gender == "female").reshape(-1)]
        y_inds_test = np.arange(test_size)[(test_gender == "female").reshape(-1)]
    elif target_data == "ADULT":
        y_inds_train = np.arange(train_size)[
            (train_age > 44).reshape(-1)*(train_age != -1).reshape(-1)
        ]
        y_inds_test = np.arange(test_size)[
            (test_age > 44).reshape(-1)*(test_age != -1).reshape(-1)
        ]
    elif target_data == "ADULT-MAN":
        male_train = np.arange(train_size)[(train_gender == "male")]
        male_test = np.arange(test_size)[(test_gender == "male")]
        
        y_inds_train = male_train[(train_age[male_train].reshape(-1) > 44)]#*(train_age != -1)
        y_inds_test = male_test[(test_age[male_test].reshape(-1) > 44)]#*(test_age != -1).reshape(-1)
    elif target_data == "YOUNG":
        y_inds_train = np.arange(train_size)[
            ((train_age > 16) & (train_age <= 44)).reshape(-1)*(train_age != -1).reshape(-1)
        ]
        y_inds_test = np.arange(test_size)[
            ((test_age > 16) & (test_age <= 44)).reshape(-1)*(test_age != -1).reshape(-1)
        ]
    return x_inds_train, y_inds_train, x_inds_test, y_inds_test

def print_stats(msg, stats_file):
    print(msg)
    print(msg, file=stats_file)

def load_test_images(image_dir, indices_to_load, base_index=59998, max_workers=64):
    """Loads specified PNG images from a directory using multiple threads.

    Args:
        image_dir (str): Path to the directory containing images.
        indices_to_load (list or np.array): List of relative indices (within the test set)
                                             to load. Filenames are constructed using
                                             base_index + relative_index.
        base_index (int): The starting index for filenames (e.g., 60000).
        max_workers (int): Maximum number of worker threads.

    Returns:
        np.array: A NumPy array containing the loaded and normalized images
                  in the order specified by indices_to_load. Returns None if
                  no images could be loaded.
    """

    # --- Define Inner Worker Function ---
    def load_normalize_single_image(relative_idx):
        """Loads and normalizes a single image based on its relative index."""
        original_index = base_index + relative_idx
        filename = os.path.join(image_dir, f"ffhq_test_{original_index:05d}.png")
        try:
            with Image.open(filename) as img:
                img_np = np.array(img.convert('RGB'))
                img_np_normalized = (img_np.astype(np.float32) / 127.5) - 1.0
                return relative_idx, img_np_normalized
        except FileNotFoundError:
            print(f"Warning: Image file not found: {filename}")
            return relative_idx, None
        except Exception as e:
            print(f"Warning: Error loading image {filename}: {e}")
            return relative_idx, None
    # --- End Inner Worker Function ---

    results_dict = {} # Use dict to store results keyed by index
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx in indices_to_load:
            futures.append(executor.submit(load_normalize_single_image, idx))

        # Iterate without tqdm
        for future in concurrent.futures.as_completed(futures):
            relative_idx, img_data = future.result()
            if img_data is not None:
                results_dict[relative_idx] = img_data

    # Assemble results in the original order
    loaded_images_ordered = []
    missing_count = 0
    for idx in indices_to_load:
        img = results_dict.get(idx)
        if img is not None:
            loaded_images_ordered.append(img)
        else:
            missing_count += 1

    if missing_count > 0:
        print(f"Warning: {missing_count} out of {len(indices_to_load)} images could not be loaded.")

    if not loaded_images_ordered:
        print(f"Error: Failed to load any images for the specified indices from {image_dir}.")
        return None # Indicate failure

    # Stack the loaded images
    final_images_array = np.stack(loaded_images_ordered, axis=0)
    return final_images_array.transpose(0, 3, 1, 2)
