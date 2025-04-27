print("Initializing SideBySide Stereoscope nodes for immersive 3D content creation...")

# Import the SBS Legacy implementation (External Depth Map)
try:
    from .sbs_with_external_depth import SBS_External_Depthmap_by_SamSeen
    print("Successfully imported SBS_External_Depthmap_by_SamSeen")
except Exception as e:
    print(f"Error importing SBS_External_Depthmap_by_SamSeen: {e}")
    # Create a placeholder class
    class SBS_External_Depthmap_by_SamSeen:
        @classmethod
        def INPUT_TYPES(s):
            return {"required": {"error": ("STRING", {"default": "Error loading SBS_External_Depthmap_by_SamSeen"})}}
        RETURN_TYPES = ("STRING",)
        FUNCTION = "error"
        CATEGORY = "👀 SamSeen"
        def error(self, error):
            return (f"ERROR: {error}",)

# Import the SBS V2 implementation
try:
    from .sbs_v2 import SBS_V2_by_SamSeen
    print("Successfully imported SBS_V2_by_SamSeen")
except Exception as e:
    print(f"Error importing SBS_V2_by_SamSeen: {e}")
    # Create a placeholder class
    class SBS_V2_by_SamSeen:
        @classmethod
        def INPUT_TYPES(s):
            return {"required": {"error": ("STRING", {"default": "Error loading SBS_V2_by_SamSeen"})}}
        RETURN_TYPES = ("STRING",)
        FUNCTION = "error"
        CATEGORY = "👀 SamSeen"
        def error(self, error):
            return (f"ERROR: {error}",)

# Import the video utility nodes
try:
    # First try to import directly
    from video_utils import SBSVideoUploader, SBSImageUploader, SBSVideoCombiner
    print("Successfully imported SBS video utility nodes")
except ImportError:
    try:
        # Then try relative import
        from .video_utils import SBSVideoUploader, SBSImageUploader, SBSVideoCombiner
        print("Successfully imported SBS video utility nodes")
    except Exception as e:
        print(f"Error importing SBS video utility nodes: {e}")
        # Create placeholder classes
        class SBSVideoUploader:
            @classmethod
            def INPUT_TYPES(s):
                return {"required": {"error": ("STRING", {"default": "Error loading SBSVideoUploader"})}}
            RETURN_TYPES = ("STRING",)
            FUNCTION = "error"
            CATEGORY = "👀 SamSeen"
            def error(self, error):
                return (f"ERROR: {error}",)

        class SBSImageUploader:
            @classmethod
            def INPUT_TYPES(s):
                return {"required": {"error": ("STRING", {"default": "Error loading SBSImageUploader"})}}
            RETURN_TYPES = ("STRING",)
            FUNCTION = "error"
            CATEGORY = "👀 SamSeen"
            def error(self, error):
                return (f"ERROR: {error}",)

        class SBSVideoCombiner:
            @classmethod
            def INPUT_TYPES(s):
                return {"required": {"error": ("STRING", {"default": "Error loading SBSVideoCombiner"})}}
            RETURN_TYPES = ("STRING",)
            FUNCTION = "error"
            CATEGORY = "👀 SamSeen"
            def error(self, error):
                return (f"ERROR: {error}",)

# Define the node mappings
NODE_CLASS_MAPPINGS = {
    "SBS_External_Depthmap_by_SamSeen": SBS_External_Depthmap_by_SamSeen,
    "SBS_V2_by_SamSeen": SBS_V2_by_SamSeen,
    "SBS_Video_Uploader": SBSVideoUploader,
    "SBS_Image_Uploader": SBSImageUploader,
    "SBS_Video_Combiner": SBSVideoCombiner
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SBS_External_Depthmap_by_SamSeen": "👀 SBS by SamSeen (Legacy)",
    "SBS_V2_by_SamSeen": "👀 SBS V2 by SamSeen",
    "SBS_Video_Uploader": "👀 SBS Video Uploader by SamSeen",
    "SBS_Image_Uploader": "👀 SBS Image Uploader by SamSeen",
    "SBS_Video_Combiner": "👀 SBS Video Combiner by SamSeen"
}

# Define the web directory for custom UI components
import os
WEB_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), "web")

# Define the JavaScript files to load
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

print("SideBySide Stereoscope nodes initialized successfully! Ready to create amazing 3D content!")
