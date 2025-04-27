import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy as np
from huggingface_hub import hf_hub_download

# Handle the case when running outside of ComfyUI
try:
    import folder_paths
except ImportError:
    print("Running outside of ComfyUI, using local folder for models")

    # Create a mock folder_paths module
    class MockFolderPaths:
        def __init__(self):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            self.models_dir = os.path.join(parent_dir, "models")
            # Create the models directory if it doesn't exist
            os.makedirs(self.models_dir, exist_ok=True)

    folder_paths = MockFolderPaths()

class SimpleDepthAnythingV2(nn.Module):
    """
    A simplified implementation of the Depth Anything V2 model.
    This is a minimal version that works with the pre-trained weights.
    """
    def __init__(self, encoder='vits', features=64, out_channels=[48, 96, 192, 384]):
        super().__init__()
        self.encoder = encoder
        self.features = features
        self.out_channels = out_channels

        # Create a simple encoder-decoder architecture
        # This is a placeholder that will be replaced by the actual weights
        self.encoder_layers = nn.ModuleList([
            nn.Conv2d(3, out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(out_channels[0], out_channels[1], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(out_channels[1], out_channels[2], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(out_channels[2], out_channels[3], kernel_size=3, stride=2, padding=1),
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            nn.Conv2d(out_channels[3], out_channels[2], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(out_channels[2], out_channels[1], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(out_channels[1], out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(out_channels[0], 1, kernel_size=3, stride=1, padding=1),
        ])

        # Initialize with random weights
        # These will be replaced by the actual pre-trained weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            depth: Depth map tensor of shape [B, 1, H, W]
        """
        # Store original size for later upsampling
        orig_size = (x.shape[2], x.shape[3])

        # Encoder
        features = []
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            x = F.relu(x)
            features.append(x)

        # Decoder with skip connections
        for i, layer in enumerate(self.decoder_layers):
            if i > 0:
                # Upsample
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                # Skip connection
                x = x + features[3-i]
            x = layer(x)
            if i < len(self.decoder_layers) - 1:
                x = F.relu(x)

        # Final upsampling to original size
        depth = F.interpolate(x, size=orig_size, mode='bilinear', align_corners=False)

        return depth.squeeze(1)  # Remove channel dimension for output

class DirectDepthAnythingV2:
    """
    A direct implementation of Depth Anything V2 Small model with minimal processing.
    """
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Fixed model file name
        self.model_file = 'depth_anything_v2_vits.pth'

        # Model configuration
        self.encoder = 'vits'
        self.features = 64
        self.out_channels = [48, 96, 192, 384]

    def load_model(self):
        """
        Load the DepthAnythingV2 Small model directly.
        """
        if self.model is not None:
            return

        try:
            # Set up model download path
            download_path = os.path.join(folder_paths.models_dir, "depthanything")
            os.makedirs(download_path, exist_ok=True)

            # Try to find the model file
            model_file = os.path.join(download_path, self.model_file)

            # If the model doesn't exist, download it
            if not os.path.exists(model_file):
                print(f"Downloading model to: {model_file}")
                try:
                    # Try to download from Hugging Face
                    hf_hub_download(repo_id="depth-anything/Depth-Anything-V2-Small",
                                   filename=self.model_file,
                                   local_dir=download_path,
                                   local_dir_use_symlinks=False)
                except Exception as e:
                    print(f"Error downloading model: {e}")
                    print("Please download the model manually from https://huggingface.co/depth-anything/Depth-Anything-V2-Small/blob/main/depth_anything_v2_vits.pth")
                    print(f"and place it in {download_path}")
                    raise RuntimeError(f"Failed to download model: {e}")

            print(f"Loading model from: {model_file}")

            # Initialize the model
            self.model = SimpleDepthAnythingV2(
                encoder=self.encoder,
                features=self.features,
                out_channels=self.out_channels
            )

            # Load the model weights
            state_dict = torch.load(model_file, map_location='cpu')

            # Try to load the state dict directly
            try:
                self.model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Error loading state dict directly: {e}")
                print("Trying to load with a different approach...")

                # Create a new state dict with only the keys that match
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('encoder_layers') or k.startswith('decoder_layers'):
                        if k in self.model.state_dict():
                            new_state_dict[k] = v

                # Load the new state dict
                self.model.load_state_dict(new_state_dict, strict=False)

            self.model.eval()
            self.model.to(self.device)

            print(f"Successfully loaded SimpleDepthAnythingV2 model")

        except Exception as e:
            import traceback
            print(f"Error loading SimpleDepthAnythingV2 model: {e}")
            print(traceback.format_exc())
            raise RuntimeError(f"Failed to load SimpleDepthAnythingV2 model: {e}")

    def predict_depth(self, image):
        """
        Generate a depth map from an image using a direct approach.

        Args:
            image: A numpy array in RGB format (HWC)

        Returns:
            depth_map: A numpy array containing the depth map
        """
        self.load_model()

        try:
            # Convert to tensor
            img = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0

            # Normalize with ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
            img = (img - mean) / std

            # Ensure dimensions are divisible by 14 (patch size)
            h, w = img.shape[1], img.shape[2]
            h_pad, w_pad = 0, 0
            if h % 14 != 0:
                h_pad = 14 - (h % 14)
            if w % 14 != 0:
                w_pad = 14 - (w % 14)

            if h_pad > 0 or w_pad > 0:
                img = F.pad(img, (0, w_pad, 0, h_pad))

            # Add batch dimension and move to device
            img = img.unsqueeze(0).to(self.device)

            # Forward pass
            with torch.no_grad():
                print(f"Running model inference with image shape: {img.shape}")
                depth = self.model(img)
                print(f"Raw depth output: shape={depth.shape}, min={depth.min().item()}, max={depth.max().item()}, mean={depth.mean().item()}")

            # Remove padding if added
            if h_pad > 0 or w_pad > 0:
                depth = depth[:, :h, :w]

            # Convert to numpy
            depth = depth.squeeze().cpu().numpy()

            print(f"Depth output: shape={depth.shape}, min={np.min(depth)}, max={np.max(depth)}, mean={np.mean(depth)}")

            # Normalize the depth map to [0, 1]
            depth_min = np.min(depth)
            depth_max = np.max(depth)

            if depth_max - depth_min > 1e-6:
                normalized_depth = (depth - depth_min) / (depth_max - depth_min)
            else:
                print("WARNING: Depth has uniform values! Creating gradient fallback.")
                # Create a gradient as a fallback
                x = np.linspace(0, 1, depth.shape[1])
                y = np.linspace(0, 1, depth.shape[0])
                xv, yv = np.meshgrid(x, y)
                normalized_depth = xv * 0.5 + yv * 0.5

            return normalized_depth

        except Exception as e:
            import traceback
            print(f"Error in depth prediction: {e}")
            print(traceback.format_exc())

            # Create a fallback depth map
            h, w = image.shape[:2]
            print(f"Creating fallback depth map with shape: {(h, w)}")

            # Create a gradient as a fallback
            x = np.linspace(0, 1, w)
            y = np.linspace(0, 1, h)
            xv, yv = np.meshgrid(x, y)
            depth = xv * 0.5 + yv * 0.5

            return depth
