import torch
from PIL import Image
import numpy as np
import tqdm
from comfy.utils import ProgressBar

class SBS_External_Depthmap_by_SamSeen:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "depth_map": ("IMAGE",),
                "depth_scale": ("INT", {"default": 30}),
                "mode": (["Parallel", "Cross-eyed"], {}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("stereoscopic_image",)
    FUNCTION = "process"
    CATEGORY = "👀 SamSeen"
    DESCRIPTION = "Legacy version: Create side-by-side (SBS) stereoscopic images and videos using your own custom depth maps. For advanced users who want complete control over the 3D effect."

    def process(self, base_image, depth_map, depth_scale, mode="Cross-eyed"):
        """
        Create a side-by-side (SBS) stereoscopic image from a standard image.

        Parameters:
        - base_image: tensor representing the base image.
        - depth_map: tensor representing the depth map.
        - depth_scale: integer representing the scaling factor for depth.
        - mode: "Parallel" or "Cross-eyed" viewing mode.

        Returns:
        - sbs_image: the stereoscopic image.
        """
        # Convert tensor to numpy array and then to PIL Image
        image_np = base_image.cpu().numpy().squeeze(0)  # Convert from CxHxW to HxWxC
        image = Image.fromarray((image_np * 255).astype(np.uint8))  # Convert float [0,1] tensor to uint8 image

        # Convert depth map tensor to numpy array and then to PIL Image
        depth_map_np = depth_map.cpu().numpy().squeeze(0)  # Convert from CxHxW to HxWxC

        # Check if depth_map_np has the right shape for PIL
        if len(depth_map_np.shape) == 3 and depth_map_np.shape[2] == 3:
            # Already in RGB format
            depth_map_img = Image.fromarray((depth_map_np * 255).astype(np.uint8))
        elif len(depth_map_np.shape) == 2:
            # Single channel - convert to grayscale
            depth_map_img = Image.fromarray((depth_map_np * 255).astype(np.uint8), mode='L')
        else:
            # Try to convert to a format PIL can handle
            print(f"Unusual depth map shape: {depth_map_np.shape}, attempting to convert")
            if len(depth_map_np.shape) == 3:
                # Take first channel if it's a multi-channel image
                depth_map_np = depth_map_np[:, :, 0]
            depth_map_img = Image.fromarray((depth_map_np * 255).astype(np.uint8), mode='L')

        # Get dimensions and resize depth map to match base image
        width, height = image.size
        depth_map_img = depth_map_img.resize((width, height), Image.NEAREST)
        fliped = 0 if mode == "Parallel" else width

        # Create an empty image for the side-by-side result
        sbs_image = np.zeros((height, width * 2, 3), dtype=np.uint8)
        depth_scaling = depth_scale / width
        pbar = ProgressBar(height)

        # Fill the base images
        for y in range(height):
            for x in range(width):
                color = image.getpixel((x, y))
                sbs_image[y, width + x] = color
                sbs_image[y, x] = color

        # generating the shifted image
        for y in tqdm.tqdm(range(height)):
            pbar.update(1)
            for x in range(width):

                depth_value = depth_map_img.getpixel((x, y))[0]
                pixel_shift = int(depth_value * depth_scaling)

                new_x = x + pixel_shift

                if new_x >= width:
                    new_x = width
                if new_x <= 0:
                    new_x = 0

                for i in range(pixel_shift+10):
                    if new_x + i  >= width or new_x  < 0:
                        break
                    new_coords = (y, new_x + i + fliped)
                    sbs_image[new_coords] = image.getpixel((x, y))

        # Convert back to tensor if needed
        sbs_image_tensor = torch.tensor(sbs_image.astype(np.float32) / 255.0).unsqueeze(0)  # Convert back to CxHxW

        return (sbs_image_tensor,)
