import os
import subprocess
import numpy as np
import torch
from PIL import Image
import cv2

# Import ComfyUI-specific modules
try:
    import folder_paths
except ImportError:
    # For standalone testing
    class FolderPaths:
        def get_input_directory(self):
            return "input"
        def get_output_directory(self):
            return "output"
        def get_temp_directory(self):
            return "temp"
    folder_paths = FolderPaths()

# Define supported video extensions
video_extensions = ['webm', 'mp4', 'mkv', 'gif', 'mov']

# Check for ffmpeg availability
def get_ffmpeg_path():
    """Get the path to ffmpeg executable"""
    # Try to find ffmpeg in the system path
    try:
        if os.name == 'nt':  # Windows
            ffmpeg_command = "where ffmpeg"
        else:  # Unix/Linux/Mac
            ffmpeg_command = "which ffmpeg"

        result = subprocess.run(ffmpeg_command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass

    # Check if ffmpeg is in the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if os.name == 'nt':  # Windows
        ffmpeg_path = os.path.join(current_dir, "ffmpeg.exe")
    else:  # Unix/Linux/Mac
        ffmpeg_path = os.path.join(current_dir, "ffmpeg")

    if os.path.exists(ffmpeg_path):
        return ffmpeg_path

    # Try to use imageio's ffmpeg
    try:
        import imageio
        return imageio.plugins.ffmpeg.get_exe()
    except:
        pass

    return None

ffmpeg_path = get_ffmpeg_path()

def target_size(width, height, custom_width, custom_height):
    """Calculate target size while maintaining aspect ratio"""
    if custom_width == 0 and custom_height == 0:
        return width, height

    if custom_width == 0:
        # Calculate width based on height
        new_width = int(width * (custom_height / height))
        return new_width, custom_height

    if custom_height == 0:
        # Calculate height based on width
        new_height = int(height * (custom_width / width))
        return custom_width, new_height

    # Both dimensions specified
    return custom_width, custom_height

def extract_frames_from_video(video_path, output_tensors=True, max_width=512, max_height=512):
    """Extract frames from a video file and return as tensors or PIL images"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate target size
    target_w, target_h = target_size(width, height, max_width, max_height)

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize frame if needed
        if width != target_w or height != target_h:
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        if output_tensors:
            # Convert to tensor
            frame_tensor = torch.from_numpy(frame).float() / 255.0
            frames.append(frame_tensor)
        else:
            # Convert to PIL Image
            frame_pil = Image.fromarray(frame)
            frames.append(frame_pil)

    cap.release()

    if output_tensors:
        # Stack tensors into a batch
        if frames:
            return torch.stack(frames)
        return torch.zeros((0, target_h, target_w, 3))

    return frames

def resize_image(image, max_width=512, max_height=512):
    """Resize an image while maintaining aspect ratio"""
    width, height = image.size
    target_w, target_h = target_size(width, height, max_width, max_height)

    if width != target_w or height != target_h:
        return image.resize((target_w, target_h), Image.LANCZOS)

    return image

def combine_frames_to_video(frames, output_path, fps=30, audio_path=None):
    """Combine frames into a video file"""
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg to use video functionality.")

    if not frames:
        raise ValueError("No frames provided")

    # Create temporary directory for frames if it doesn't exist
    temp_dir = os.path.join(folder_paths.get_temp_directory(), "frames")
    os.makedirs(temp_dir, exist_ok=True)

    # Save frames as images
    for i, frame in enumerate(frames):
        if isinstance(frame, torch.Tensor):
            # Convert tensor to PIL Image
            if frame.dim() == 4:  # Batch of images
                frame = frame[0]

            frame = (frame * 255).byte().cpu().numpy()
            frame = Image.fromarray(frame)

        frame.save(os.path.join(temp_dir, f"frame_{i:05d}.png"))

    # Construct ffmpeg command
    ffmpeg_cmd = [
        ffmpeg_path,
        "-y",  # Overwrite output file if it exists
        "-framerate", str(fps),
        "-i", os.path.join(temp_dir, "frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",  # Quality setting (lower is better)
    ]

    # Add audio if provided
    if audio_path and os.path.exists(audio_path):
        ffmpeg_cmd.extend(["-i", audio_path, "-c:a", "aac", "-shortest"])

    ffmpeg_cmd.append(output_path)

    # Run ffmpeg
    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error creating video: {e.stderr.decode()}")

    # Clean up temporary files
    for i in range(len(frames)):
        os.remove(os.path.join(temp_dir, f"frame_{i:05d}.png"))

    return output_path

class SBSVideoUploader:
    @classmethod
    def INPUT_TYPES(s):
        try:
            input_dir = folder_paths.get_input_directory()
            files = []

            if os.path.exists(input_dir):
                for f in os.listdir(input_dir):
                    if os.path.isfile(os.path.join(input_dir, f)):
                        file_parts = f.split('.')
                        if len(file_parts) > 1 and (file_parts[-1].lower() in video_extensions):
                            files.append(f)

            if len(files) == 0:
                files = ["none"]
        except Exception:
            files = ["none"]

        return {
            "required": {
                "video": (sorted(files),),
                "max_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "max_height": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8, "display": "Max height (0 = auto)"}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "display": "Max frames (0 = all)"}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "frame_count")
    FUNCTION = "load_video"
    CATEGORY = "👀 SamSeen"

    def load_video(self, video, max_width, max_height, frame_load_cap=0, skip_first_frames=0, select_every_nth=1):
        if video == "none":
            # Return an empty tensor if no video is selected
            return (torch.zeros((0, 512, 512, 3)), 0)

        try:
            video_path = os.path.join(folder_paths.get_input_directory(), video)

            # Open the video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
        except Exception as e:
            print(f"Error loading video: {e}")
            return (torch.zeros((0, 512, 512, 3)), 0)

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate target size
        target_w, target_h = target_size(width, height, max_width, max_height)

        # Skip frames if needed
        if skip_first_frames > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, skip_first_frames)

        frames = []
        frame_count = 0
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process every nth frame
            if frame_index % select_every_nth == 0:
                # Convert from BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize frame if needed
                if width != target_w or height != target_h:
                    frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

                # Convert to tensor
                frame_tensor = torch.from_numpy(frame).float() / 255.0
                frames.append(frame_tensor)

                frame_count += 1

                # Stop if we've reached the frame cap
                if frame_load_cap > 0 and frame_count >= frame_load_cap:
                    break

            frame_index += 1

        cap.release()

        # Stack tensors into a batch
        if frames:
            return (torch.stack(frames), frame_count)

        return (torch.zeros((0, target_h, target_w, 3)), 0)

class SBSImageUploader:
    @classmethod
    def INPUT_TYPES(s):
        try:
            input_dir = folder_paths.get_input_directory()
            files = []

            if os.path.exists(input_dir):
                for f in os.listdir(input_dir):
                    if os.path.isfile(os.path.join(input_dir, f)):
                        file_parts = f.split('.')
                        if len(file_parts) > 1 and (file_parts[-1].lower() in ['png', 'jpg', 'jpeg', 'webp', 'bmp']):
                            files.append(f)

            if len(files) == 0:
                files = ["none"]
        except Exception:
            files = ["none"]

        return {
            "required": {
                "image": (sorted(files),),
                "max_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "max_height": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8, "display": "Max height (0 = auto)"})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_image"
    CATEGORY = "👀 SamSeen"

    def load_image(self, image, max_width, max_height):
        if image == "none":
            # Return an empty tensor if no image is selected
            return (torch.zeros((1, 512, 512, 3)),)

        try:
            image_path = os.path.join(folder_paths.get_input_directory(), image)

            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
        except Exception as e:
            print(f"Error loading image: {e}")
            return (torch.zeros((1, 512, 512, 3)),)

        # Load the image
        img = Image.open(image_path).convert('RGB')

        # Resize if needed
        width, height = img.size
        target_w, target_h = target_size(width, height, max_width, max_height)

        if width != target_w or height != target_h:
            img = img.resize((target_w, target_h), Image.LANCZOS)

        # Convert to tensor
        img_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)

        return (img_tensor,)

class SBSVideoCombiner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": ("INT", {"default": 30, "min": 1, "max": 120, "step": 1}),
                "filename_prefix": ("STRING", {"default": "SBS_Video"}),
                "format": (["mp4", "webm", "gif"], {"default": "mp4"}),
                "save_output": ("BOOLEAN", {"default": True})
            },
            "optional": {
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    OUTPUT_NODE = True
    FUNCTION = "combine_video"
    CATEGORY = "👀 SamSeen"

    def combine_video(self, images, frame_rate, filename_prefix, format, save_output, audio=None):
        if ffmpeg_path is None:
            raise RuntimeError("ffmpeg not found. Please install ffmpeg to use video functionality.")

        if images.size(0) == 0:
            raise ValueError("No images provided")

        # Get output directory
        output_dir = folder_paths.get_output_directory() if save_output else folder_paths.get_temp_directory()
        os.makedirs(output_dir, exist_ok=True)

        # Generate a unique filename
        counter = 1
        while True:
            filename = f"{filename_prefix}_{counter:05d}.{format}"
            file_path = os.path.join(output_dir, filename)
            if not os.path.exists(file_path):
                break
            counter += 1

        # Create temporary directory for frames
        temp_dir = os.path.join(folder_paths.get_temp_directory(), "frames")
        os.makedirs(temp_dir, exist_ok=True)

        # Save frames as images
        for i in range(images.size(0)):
            frame = images[i].cpu().numpy() * 255
            frame = Image.fromarray(frame.astype(np.uint8))
            frame.save(os.path.join(temp_dir, f"frame_{i:05d}.png"))

        # Construct ffmpeg command
        ffmpeg_cmd = [
            ffmpeg_path,
            "-y",  # Overwrite output file if it exists
            "-framerate", str(frame_rate),
            "-i", os.path.join(temp_dir, "frame_%05d.png")
        ]

        # Add format-specific options
        if format == "mp4":
            ffmpeg_cmd.extend([
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "23"  # Quality setting (lower is better)
            ])
        elif format == "webm":
            ffmpeg_cmd.extend([
                "-c:v", "libvpx-vp9",
                "-pix_fmt", "yuv420p",
                "-crf", "30"  # Quality setting for VP9
            ])
        elif format == "gif":
            ffmpeg_cmd.extend([
                "-filter_complex", "[0:v] split [a][b];[a] palettegen [p];[b][p] paletteuse"
            ])

        # Add audio if provided
        if audio is not None:
            # Create a temporary WAV file for the audio
            audio_path = os.path.join(folder_paths.get_temp_directory(), "temp_audio.wav")

            # Get audio data
            waveform = audio['waveform']
            sample_rate = audio['sample_rate']

            # Save audio to WAV file using scipy
            try:
                import scipy.io.wavfile
                # Convert from torch tensor to numpy array
                audio_data = waveform.squeeze(0).transpose(0, 1).cpu().numpy()
                scipy.io.wavfile.write(audio_path, sample_rate, audio_data)

                # Add audio to ffmpeg command
                ffmpeg_cmd.extend([
                    "-i", audio_path,
                    "-c:a", "aac" if format != "webm" else "libopus",
                    "-shortest"  # End when the shortest input stream ends
                ])
            except Exception as e:
                print(f"Warning: Could not add audio to video: {e}")

        # Add output file
        ffmpeg_cmd.append(file_path)

        # Run ffmpeg
        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error creating video: {e.stderr.decode()}")

        # Clean up temporary files
        for i in range(images.size(0)):
            try:
                os.remove(os.path.join(temp_dir, f"frame_{i:05d}.png"))
            except:
                pass

        if audio is not None and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass

        return (file_path,)
