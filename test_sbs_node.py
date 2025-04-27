import os
import sys
import numpy as np
from PIL import Image
import cv2
import torch

# Add the current directory to the path so we can import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Create a mock ProgressBar class to replace ComfyUI's ProgressBar
class MockProgressBar:
    def __init__(self, total):
        self.total = total
        self.current = 0

    def update(self, n=1):
        self.current += n
        print(f"Progress: {self.current}/{self.total}")

# Add the mock to sys.modules to avoid import errors
sys.modules['comfy.utils'] = type('MockComfyUtils', (), {'ProgressBar': MockProgressBar})

# Now try to import our SBS node
try:
    from sbs_v2 import SBS_V2_by_SamSeen
    print("Successfully imported SBS_V2_by_SamSeen")
except ImportError as e:
    print(f"Error importing SBS_V2_by_SamSeen: {e}")
    sys.exit(1)

def main():
    """
    Test the SBS node with a sample image.
    """
    # Check if an image path is provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Look for any image in the current directory
        image_files = [f for f in os.listdir('.') if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
        if image_files:
            image_path = image_files[0]
            print(f"Using image: {image_path}")
        else:
            print("No image files found in the current directory.")
            print("Please provide an image path as an argument.")
            sys.exit(1)

    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        sys.exit(1)

    # Load the image
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)

    print(f"Image shape: {img_np.shape}")

    # Convert to tensor format (simulate ComfyUI input)
    import torch
    img_tensor = torch.tensor(img_np.astype(np.float32) / 255.0).unsqueeze(0)  # Add batch dimension

    # Create an instance of our SBS node
    sbs_node = SBS_V2_by_SamSeen()

    # Set parameters
    depth_scale = 30.0
    blur_radius = 15
    invert_depth = False  # Don't invert the depth map again
    mode = "Cross-eyed"

    # Process the image
    print(f"Processing image with parameters: depth_scale={depth_scale}, blur_radius={blur_radius}, invert_depth={invert_depth}, mode={mode}")
    sbs_image, depth_map = sbs_node.process(img_tensor, depth_scale, blur_radius, invert_depth, mode)

    # Convert tensors back to numpy arrays
    sbs_np = sbs_image.cpu().numpy().squeeze(0) * 255.0
    depth_np = depth_map.cpu().numpy().squeeze(0) * 255.0

    # Save the results
    sbs_output = Image.fromarray(sbs_np.astype(np.uint8))
    sbs_output.save('sbs_result.png')
    print(f"SBS image saved to: sbs_result.png")

    depth_output = Image.fromarray(depth_np.astype(np.uint8))
    depth_output.save('depth_result_from_node.png')
    print(f"Depth map saved to: depth_result_from_node.png")

    print("Test completed successfully!")

if __name__ == "__main__":
    main()
