# Import the modules we need
try:
    from .dpt import DepthAnythingV2
    from .dinov2 import DINOv2
    from .util.blocks import FeatureFusionBlock, _make_scratch

    __all__ = ['DepthAnythingV2', 'DINOv2', 'FeatureFusionBlock', '_make_scratch']

    print("Successfully imported all modules from depth_anything_v2")
except Exception as e:
    print(f"Error importing modules from depth_anything_v2: {e}")
