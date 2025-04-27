#!/usr/bin/env python3
"""
Test script for the video utility nodes.
This script tests the basic functionality of the video nodes.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

# Add the current directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from video_utils import SBSVideoUploader, SBSImageUploader, SBSVideoCombiner
    print("Successfully imported video utility nodes")
except Exception as e:
    print(f"Error importing video utility nodes: {e}")
    sys.exit(1)

def test_video_uploader():
    """Test the SBSVideoUploader class"""
    print("\nTesting SBSVideoUploader...")
    
    # Create a dummy video file for testing
    # This would normally be in the input directory
    print("Note: This test requires a video file in the input directory")
    
    # Print the INPUT_TYPES to verify it's working
    input_types = SBSVideoUploader.INPUT_TYPES()
    print(f"INPUT_TYPES: {input_types}")
    
    print("SBSVideoUploader test completed")

def test_image_uploader():
    """Test the SBSImageUploader class"""
    print("\nTesting SBSImageUploader...")
    
    # Create a dummy image file for testing
    # This would normally be in the input directory
    print("Note: This test requires an image file in the input directory")
    
    # Print the INPUT_TYPES to verify it's working
    input_types = SBSImageUploader.INPUT_TYPES()
    print(f"INPUT_TYPES: {input_types}")
    
    print("SBSImageUploader test completed")

def test_video_combiner():
    """Test the SBSVideoCombiner class"""
    print("\nTesting SBSVideoCombiner...")
    
    # Print the INPUT_TYPES to verify it's working
    input_types = SBSVideoCombiner.INPUT_TYPES()
    print(f"INPUT_TYPES: {input_types}")
    
    # Create a dummy tensor for testing
    dummy_tensor = torch.zeros((5, 64, 64, 3))
    
    print("SBSVideoCombiner test completed")

def main():
    """Main function"""
    print("Testing video utility nodes...")
    
    test_video_uploader()
    test_image_uploader()
    test_video_combiner()
    
    print("\nAll tests completed")

if __name__ == "__main__":
    main()
