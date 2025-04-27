import torch
from PIL import Image
import numpy as np
import tqdm
import os
import sys
import cv2
from comfy.utils import ProgressBar

# Add the current directory to the path so we can import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import our depth estimation implementation
try:
    from depth_estimator import DepthEstimator
    print("Successfully imported DepthEstimator")
except ImportError as e:
    print(f"Error importing DepthEstimator: {e}")

    # Define a placeholder class that will show a clear error
    class DepthEstimator:
        def __init__(self):
            print("ERROR: DepthEstimator could not be imported!")

        def load_model(self):
            print("ERROR: DepthEstimator model could not be loaded!")
            return None

        def predict_depth(self, image):
            print("ERROR: DepthEstimator model could not be used for inference!")
            # Return a blank depth map
            h, w = image.shape[:2]
            return np.zeros((h, w), dtype=np.float32)

class SBS_V2_by_SamSeen:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.depth_model = None
        self.original_depths = []

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "depth_scale": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 100.0, "step": 0.5}),
                "blur_radius": ("INT", {"default": 3, "min": 1, "max": 51, "step": 2}),
                "invert_depth": ("BOOLEAN", {"default": False}),
                "mode": (["Parallel", "Cross-eyed"], {"default": "Cross-eyed"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("stereoscopic_image", "depth_map")
    FUNCTION = "process"
    CATEGORY = "👀 SamSeen"
    DESCRIPTION = "Create stunning side-by-side (SBS) stereoscopic images and videos with automatic depth map generation using Depth-Anything-V2. Perfect for VR content, 3D displays, and image sequences!"

    def load_depth_model(self):
        """
        Load the depth model.
        """
        # Create a new instance of our depth model if needed
        if self.depth_model is None:
            print("Creating new DepthEstimator instance")
            self.depth_model = DepthEstimator()

        # Load the model
        try:
            self.depth_model.load_model()
            print("Successfully loaded DepthEstimator model")
        except Exception as e:
            import traceback
            print(f"Error loading DepthEstimator model: {e}")
            print(traceback.format_exc())

        return self.depth_model

    def generate_depth_map(self, image):
        """
        Generate a depth map from an image or batch of images.
        """
        try:
            # Load the model if not already loaded
            depth_model = self.load_depth_model()

            # Process the image
            B, H, W, C = image.shape
            pbar = ProgressBar(B)
            out = []

            # Store original depth maps for each image in the batch
            self.original_depths = []

            # Process each image in the batch
            for b in range(B):
                # Convert tensor to numpy for processing
                img_np = image[b].cpu().numpy() * 255.0  # Scale to 0-255
                img_np = img_np.astype(np.uint8)

                print(f"Processing image {b+1}/{B} with shape: {img_np.shape}")

                # Use our depth model's predict_depth method
                depth = depth_model.predict_depth(img_np)

                print(f"Raw depth output: shape={depth.shape}, min={np.min(depth)}, max={np.max(depth)}, mean={np.mean(depth)}")

                # Make sure depth is normalized to [0,1]
                if np.min(depth) < 0 or np.max(depth) > 1:
                    depth = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)

                # Save the original depth map for the SBS generation
                self.original_depths.append(depth.copy())

                # Convert back to tensor - keep as grayscale
                # First expand to 3 channels (all with same values) for ComfyUI compatibility
                depth_tensor = torch.from_numpy(depth).float().unsqueeze(0)  # Add channel dimension

                out.append(depth_tensor)
                pbar.update(1)

            # Stack the depth maps
            depth_out = torch.stack(out)

            print(f"Stacked depth maps: shape={depth_out.shape}, min={depth_out.min().item()}, max={depth_out.max().item()}, mean={depth_out.mean().item()}")

            # Make sure it's in the right format for ComfyUI (B,H,W,C)
            # For grayscale, we need to expand to 3 channels for ComfyUI compatibility
            if len(depth_out.shape) == 3:  # [B,1,H,W]
                depth_out = depth_out.permute(0, 2, 3, 1).cpu().float()  # [B,H,W,1]
                depth_out = depth_out.repeat(1, 1, 1, 3)  # [B,H,W,3]
            elif len(depth_out.shape) == 4:  # [B,C,H,W]
                depth_out = depth_out.permute(0, 2, 3, 1).cpu().float()  # [B,H,W,C]

            print(f"Final depth map shape: {depth_out.shape}, min: {depth_out.min().item()}, max: {depth_out.max().item()}, mean: {depth_out.mean().item()}")

            return depth_out
        except Exception as e:
            import traceback
            print(f"Error generating depth map: {e}")
            print(traceback.format_exc())
            # Return a blank depth map in case of error
            B, H, W, C = image.shape
            print(f"Creating blank depth map with shape: {(B, H, W, C)}")
            return torch.zeros((B, H, W, C), dtype=torch.float32)

    def process(self, base_image, depth_scale, blur_radius, invert_depth=False, mode="Cross-eyed"):
        """
        Create a side-by-side (SBS) stereoscopic image from a standard image or image sequence.
        The depth map is automatically generated using our custom depth estimation approach.

        Parameters:
        - base_image: tensor representing the base image(s) with shape [B,H,W,C].
        - depth_scale: float representing the scaling factor for depth.
        - blur_radius: integer controlling the smoothness of the depth map.
        - invert_depth: boolean to invert the depth map (swap foreground/background).
        - mode: "Parallel" or "Cross-eyed" viewing mode.

        Returns:
        - sbs_image: the stereoscopic image(s).
        - depth_map: the generated depth map(s).
        """
        # Update the depth model parameters
        if self.depth_model is not None:
            # Set default edge_weight for compatibility
            self.depth_model.edge_weight = 0.5
            # Keep gradient_weight for compatibility but set to 0
            self.depth_model.gradient_weight = 0.0
            self.depth_model.blur_radius = blur_radius

        # Generate depth map
        print(f"Generating depth map with blur_radius={blur_radius}, invert_depth={invert_depth}...")
        depth_map = self.generate_depth_map(base_image)

        # Get batch size
        B = base_image.shape[0]

        # Process each image in the batch
        sbs_images = []
        enhanced_depth_maps = []

        for b in range(B):
            # Get the current image from the batch
            current_image = base_image[b].cpu().numpy()  # Get image b from batch
            current_image_pil = Image.fromarray((current_image * 255).astype(np.uint8))  # Convert to PIL

            # Get the current depth map
            if hasattr(self, 'original_depths') and len(self.original_depths) > b:
                # Use the original grayscale depth map for this image in the batch
                depth_for_sbs = self.original_depths[b].copy()
                print(f"Using original depth map for image {b+1}/{B}: shape={depth_for_sbs.shape}, min={np.min(depth_for_sbs)}, max={np.max(depth_for_sbs)}")
            else:
                # If original depth is not available, extract from the colored version
                current_depth_map = depth_map[b].cpu().numpy()  # Get depth map b from batch

                # Debug info
                print(f"Depth map shape: {current_depth_map.shape}, min: {current_depth_map.min()}, max: {current_depth_map.max()}, mean: {current_depth_map.mean()}")

                # If we have a colored depth map, use the red channel (which should have our depth values)
                if len(current_depth_map.shape) == 3 and current_depth_map.shape[2] == 3:
                    depth_for_sbs = current_depth_map[:, :, 0].copy()  # Use red channel
                else:
                    depth_for_sbs = current_depth_map.copy()

            # Invert depth if requested (swap foreground/background)
            if invert_depth:
                print("Inverting depth map (swapping foreground/background)")
                depth_for_sbs = 1.0 - depth_for_sbs

            # Convert the depth map to a PIL image for processing
            depth_map_img = Image.fromarray((depth_for_sbs * 255).astype(np.uint8), mode='L')

            # Get dimensions and resize depth map to match base image
            width, height = current_image_pil.size
            depth_map_img = depth_map_img.resize((width, height), Image.NEAREST)
            fliped = 0 if mode == "Parallel" else width

            # Create an empty image for the side-by-side result
            sbs_image = np.zeros((height, width * 2, 3), dtype=np.uint8)
            depth_scaling = depth_scale / width
            pbar = ProgressBar(height)

            # Fill the base images
            for y in range(height):
                for x in range(width):
                    color = current_image_pil.getpixel((x, y))
                    sbs_image[y, width + x] = color
                    sbs_image[y, x] = color

            # generating the shifted image
            for y in tqdm.tqdm(range(height)):
                pbar.update(1)
                for x in range(width):
                    try:
                        depth_value = depth_map_img.getpixel((x, y))
                        if isinstance(depth_value, tuple):
                            depth_value = depth_value[0]
                        pixel_shift = int(depth_value * depth_scaling)

                        new_x = x + pixel_shift

                        if new_x >= width:
                            new_x = width - 1
                        if new_x < 0:
                            new_x = 0

                        for i in range(pixel_shift+10):
                            if new_x + i >= width or new_x < 0:
                                break
                            new_coords = (y, new_x + i + fliped)
                            sbs_image[new_coords] = current_image_pil.getpixel((x, y))
                    except Exception as e:
                        print(f"Error processing pixel at ({x}, {y}): {e}")

            # Convert to tensor
            sbs_image_tensor = torch.tensor(sbs_image.astype(np.float32) / 255.0)

            # Create a properly formatted depth map for output
            # Make sure it's normalized to [0,1]
            if np.min(depth_for_sbs) < 0 or np.max(depth_for_sbs) > 1:
                depth_gray = cv2.normalize(depth_for_sbs, None, 0, 1, cv2.NORM_MINMAX)
            else:
                depth_gray = depth_for_sbs

            # Convert to 3-channel grayscale (all channels have same value)
            depth_3ch = np.stack([depth_gray, depth_gray, depth_gray], axis=-1)

            # Convert to tensor format
            enhanced_depth_map = torch.tensor(depth_3ch)

            # Add to our batch lists
            sbs_images.append(sbs_image_tensor)
            enhanced_depth_maps.append(enhanced_depth_map)

        # Stack the results to create batched tensors
        sbs_images_batch = torch.stack(sbs_images)
        enhanced_depth_maps_batch = torch.stack(enhanced_depth_maps)

        # Print final output stats
        print(f"Final SBS image batch shape: {sbs_images_batch.shape}, min: {sbs_images_batch.min().item()}, max: {sbs_images_batch.max().item()}")
        print(f"Final depth map batch shape: {enhanced_depth_maps_batch.shape}, min: {enhanced_depth_maps_batch.min().item()}, max: {enhanced_depth_maps_batch.max().item()}")

        return (sbs_images_batch, enhanced_depth_maps_batch)
