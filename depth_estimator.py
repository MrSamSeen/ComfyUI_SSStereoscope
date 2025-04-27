import os
import sys
import torch
import numpy as np
from PIL import Image
import cv2
import requests
from tqdm import tqdm
from transformers import pipeline, AutoImageProcessor, AutoModelForDepthEstimation

# Handle the case when running outside of ComfyUI
try:
    import folder_paths
except ImportError:
    print("Running outside of ComfyUI, using local folder for models")

    # Create a mock folder_paths module
    class MockFolderPaths:
        def __init__(self):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.models_dir = os.path.join(current_dir, "models")
            # Create the models directory if it doesn't exist
            os.makedirs(self.models_dir, exist_ok=True)

    folder_paths = MockFolderPaths()

def download_file(url, local_path):
    """
    Download a file from a URL to a local path with a progress bar.
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Download the file
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    print(f"Downloading {url} to {local_path}")
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

    with open(local_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()

    if total_size != 0 and progress_bar.n != total_size:
        print("ERROR: Download incomplete")
        return False

    print(f"Download complete: {local_path}")
    return True

class DepthEstimator:
    """
    A depth estimation implementation using the Depth-Anything-V2-Small model from Hugging Face.
    """
    def __init__(self):
        self.processor = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = "depth-anything/Depth-Anything-V2-Small-hf"
        self.blur_radius = 15  # Default blur radius

    def load_model(self):
        """
        Load the Depth-Anything-V2-Small model from Hugging Face.
        """
        if self.model is not None and self.processor is not None:
            return True

        try:
            print(f"Loading Depth-Anything-V2-Small model from {self.model_id}")

            # Load the model and processor
            self.processor = AutoImageProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForDepthEstimation.from_pretrained(self.model_id)

            # Move model to the appropriate device
            self.model.to(self.device)

            print(f"Successfully loaded Depth-Anything-V2-Small model")
            return True

        except Exception as e:
            import traceback
            print(f"Error loading Depth-Anything-V2-Small model: {e}")
            print(traceback.format_exc())

            # Try using the pipeline API as a fallback
            try:
                print("Trying to load model using pipeline API...")
                self.pipe = pipeline(task="depth-estimation", model=self.model_id, device=self.device)
                print("Successfully loaded model using pipeline API")
                return True
            except Exception as e2:
                print(f"Error loading model using pipeline API: {e2}")
                print(traceback.format_exc())

                # Fall back to a simple gradient depth map
                print("Falling back to simple gradient depth map")
                return False

    def predict_depth(self, image):
        """
        Generate a depth map from an image using the Depth-Anything-V2-Small model.
        Lighter areas represent objects closer to the camera, darker areas are farther away.

        Args:
            image: A numpy array in RGB format (HWC)

        Returns:
            depth_map: A numpy array containing the depth map
        """
        try:
            # Make sure blur_radius is odd (required by GaussianBlur)
            blur_radius = self.blur_radius
            if blur_radius % 2 == 0:
                blur_radius += 1

            # Try to load the model
            model_loaded = self.load_model()

            if model_loaded:
                # Convert numpy array to PIL Image if needed
                if isinstance(image, np.ndarray):
                    if image.dtype == np.uint8:
                        pil_image = Image.fromarray(image)
                    else:
                        # Normalize to 0-255 range
                        img_normalized = (image * 255).astype(np.uint8)
                        pil_image = Image.fromarray(img_normalized)
                else:
                    pil_image = image

                # Check which method to use
                if hasattr(self, 'pipe') and self.pipe is not None:
                    # Use the pipeline API
                    result = self.pipe(pil_image)
                    depth_map = result["depth"]

                    # Convert to numpy array if it's not already
                    if isinstance(depth_map, torch.Tensor):
                        depth_map = depth_map.cpu().numpy()
                else:
                    # Use the model and processor directly
                    # Prepare image for the model
                    inputs = self.processor(images=pil_image, return_tensors="pt")

                    # Move inputs to the same device as the model
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    # Run inference
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        predicted_depth = outputs.predicted_depth

                    # Interpolate to original size
                    prediction = torch.nn.functional.interpolate(
                        predicted_depth.unsqueeze(1),
                        size=pil_image.size[::-1],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()

                    # Convert to numpy array
                    depth_map = prediction.cpu().numpy()

                # Normalize depth values to [0, 1] range
                depth_min = depth_map.min()
                depth_max = depth_map.max()
                depth_map = (depth_map - depth_min) / (depth_max - depth_min)

                # Invert the depth map so that closer objects are lighter
                depth_map = 1.0 - depth_map

                # Apply Gaussian blur to smooth the depth map
                depth_map = cv2.GaussianBlur(depth_map, (blur_radius, blur_radius), 0)

                print(f"Generated depth map using Depth-Anything-V2-Small model with shape: {depth_map.shape}")
                print(f"Depth map min: {depth_map.min()}, max: {depth_map.max()}")

                return depth_map
            else:
                # Fall back to simple gradient if model loading failed
                print("Using fallback gradient depth map")

                # Get image dimensions
                if len(image.shape) == 3:
                    h, w, _ = image.shape
                else:
                    h, w = image.shape

                # Create a horizontal gradient from 1.0 (left) to 0.0 (right)
                depth = np.zeros((h, w), dtype=np.float32)
                for y in range(h):
                    for x in range(w):
                        depth[y, x] = 1.0 - (x / w)

                # Apply Gaussian blur to smooth the depth map
                depth = cv2.GaussianBlur(depth, (blur_radius, blur_radius), 0)

                print(f"Generated fallback gradient depth map with shape: {depth.shape}")

                return depth

        except Exception as e:
            import traceback
            print(f"Error in depth prediction: {e}")
            print(traceback.format_exc())

            # Create a fallback depth map
            h, w = image.shape[:2] if len(image.shape) > 1 else (100, 100)
            print(f"Creating fallback depth map with shape: {(h, w)}")

            # Create a gradient as a fallback
            depth = np.zeros((h, w), dtype=np.float32)
            for y in range(h):
                for x in range(w):
                    depth[y, x] = 1.0 - (x / w)

            return depth

# Simple test function
def test_depth(image_path):
    """
    Test the depth estimator on an image.

    Args:
        image_path: Path to the image file
    """
    try:
        # Load the image
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)

        print(f"Image shape: {img_np.shape}")

        # Create a depth model
        depth_model = DepthEstimator()

        # Generate a depth map
        print("Generating depth map...")
        depth_map = depth_model.predict_depth(img_np)

        print(f"Depth map shape: {depth_map.shape}")
        print(f"Depth map min: {np.min(depth_map)}, max: {np.max(depth_map)}, mean: {np.mean(depth_map)}")

        # Save the depth map
        depth_colored = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
        cv2.imwrite('depth_result.png', depth_colored)
        print(f"Result saved to: depth_result.png")

        # Save grayscale version - ensure closer objects are lighter (standard convention)
        # Note: depth_map is already inverted in predict_depth so that closer objects are lighter
        cv2.imwrite('depth_result_gray.png', (depth_map * 255).astype(np.uint8))
        print(f"Grayscale result saved to: depth_result_gray.png")

        return depth_map

    except Exception as e:
        import traceback
        print(f"Error in test_depth: {e}")
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    # Check if an image path is provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Use a default image if none is provided
        image_path = input("Please enter the path to an image file: ")

    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        sys.exit(1)

    test_depth(image_path)
