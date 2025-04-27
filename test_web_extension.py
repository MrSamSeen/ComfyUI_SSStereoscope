#!/usr/bin/env python3
"""
Test script to verify that the web directory is set up correctly.
"""

import os
import sys

# Add the current directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the WEB_DIRECTORY variable from __init__.py
try:
    from __init__ import WEB_DIRECTORY
    print(f"WEB_DIRECTORY: {WEB_DIRECTORY}")
    
    # Check if the directory exists
    if os.path.exists(WEB_DIRECTORY):
        print(f"Directory exists: {WEB_DIRECTORY}")
        
        # Check if the js directory exists
        js_dir = os.path.join(WEB_DIRECTORY, "js")
        if os.path.exists(js_dir):
            print(f"JS directory exists: {js_dir}")
            
            # Check if the main.js file exists
            main_js = os.path.join(js_dir, "main.js")
            if os.path.exists(main_js):
                print(f"main.js exists: {main_js}")
            else:
                print(f"main.js does not exist: {main_js}")
        else:
            print(f"JS directory does not exist: {js_dir}")
    else:
        print(f"Directory does not exist: {WEB_DIRECTORY}")
except ImportError as e:
    print(f"Error importing WEB_DIRECTORY: {e}")
    sys.exit(1)

print("\nChecking file structure...")
for root, dirs, files in os.walk(WEB_DIRECTORY):
    for file in files:
        print(f"File: {os.path.join(root, file)}")

print("\nTest completed.")
