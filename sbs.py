import torch
from PIL import Image
import numpy as np
import tqdm


class SideBySide:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "depth_map": ("IMAGE",),
                "depth_scale": ("INT", {"default": 50}),
            },
            
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "SideBySide"
    CATEGORY = "👀 SamSeen"

    def SideBySide(self, base_image, depth_map, depth_scale):
        """
        Create a side-by-side (SBS) stereoscopic image from a standard image and a depth map.

        Parameters:
        - base_image: numpy array representing the base image.
        - depth_map: numpy array representing the depth map.
        - depth_scale: integer representing the scaling factor for depth.

        Returns:
        - sbs_image: numpy array containing the stereoscopic image.
        """
        # Convert tensor to numpy array and then to PIL Image
        image_np = base_image.cpu().numpy().squeeze(0)  # Convert from CxHxW to HxWxC
        image = Image.fromarray((image_np * 255).astype(np.uint8))  # Convert float [0,1] tensor to uint8 image

        depth_map_np = depth_map.cpu().numpy().squeeze(0)  # Convert from CxHxW to HxWxC
        depth_map_img = Image.fromarray((depth_map_np * 255).astype(np.uint8))  # Convert float [0,1] tensor to uint8 image

        # Get dimensions and resize depth map to match base image
        width, height = image.size
        depth_map_img = depth_map_img.resize((width, height), Image.NEAREST)

        # Create an empty image for the side-by-side result
        sbs_image = np.zeros((height, width * 2, 3), dtype=np.uint8)
        depth_scaling = depth_scale / width

        # Fill the right half of the SBS image with the base image
        for y in range(height):
            for x in range(width):
                color = image.getpixel((x, y))
                sbs_image[y, width + x] = color


        # Apply depth map to shift pixels in the left half of the SBS image
        for y in tqdm.tqdm(range(height)):
            for x in range(width):
                depth_value = depth_map_img.getpixel((x, y))[0]
                pixel_shift = int(depth_value * depth_scaling) + 1
                if x + pixel_shift < width:
                    new_coords = (x + 1, y) if x + 1 < width else (x, y)
                    for i in range(pixel_shift):
                        sbs_image[y, width + x - i] = image.getpixel(new_coords)

        # Copy the base image to the left half of the SBS image
        for y in range(height):
            for x in range(width):
                color = image.getpixel((x, y))
                sbs_image[y, x] = color

        # Convert back to tensor if needed
        sbs_image_tensor = torch.tensor(sbs_image.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)  # Convert back to CxHxW
        return sbs_image_tensor
