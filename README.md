# SideBySide Stereoscopic 3D Generator for ComfyUI

<p>Side by Side Stereoscope node for ComfyUI by SamSeen - Transform any 2D image into immersive 3D content!</p>

<div style="background-color: #f0f7ff; border-left: 5px solid #0078d7; padding: 15px; margin: 20px 0; border-radius: 5px;">
  <h3 style="color: #0078d7; margin-top: 0;">🎬 NEW: Create Stunning 3D Videos with Integrated Video Tools!</h3>
  <p>With our new built-in depth map generation and integrated video processing tools, creating 3D videos is now easier than ever! Simply:</p>
  <ol>
    <li>Use the <strong>👀 SBS Video Uploader</strong> to convert your video to an image sequence</li>
    <li>Process each frame through the <strong>👀 SBS V2</strong> node for automatic depth map generation</li>
    <li>Combine the processed frames back into a video with <strong>👀 SBS Video Combiner</strong></li>
  </ol>
  <p>No external depth maps or separate video processing tools needed - everything is handled automatically in one seamless workflow!</p>
</div>

<img width="1254" alt="simple_workflow" src="https://github.com/MrSamSeen/ComfyUI_SSStereoscope/assets/8541558/a7ae4c8b-6c38-47b2-a280-84cd851ef254">

<img width="1254" alt="simple_workflow" src="https://github.com/MrSamSeen/ComfyUI_SSStereoscope/assets/8541558/45016b16-81b7-430c-81a5-ec63627398e8">

<p>Welcome to the SideBySide Node repository for ComfyUI! 🚀 Elevate your AI-generated content with stunning stereoscopic 3D effects! This powerful depth-estimation node transforms ordinary 2D images into immersive side-by-side (SBS) experiences with automatic depth perception. Perfect for creating eye-catching virtual reality content, 3D videos, and image sequences that leap off the screen with realistic depth!</p>

<h2>Introduction</h2>
<p>The SideBySide Stereoscope Node is a cutting-edge 2D-to-3D conversion tool designed for ComfyUI to generate professional-quality stereoscopic images and videos with built-in depth map generation. Powered by Depth-Anything-V2's AI depth estimation, it automatically creates high-quality depth maps from your images, eliminating the need for external depth sources or specialized 3D cameras. This free-viewing 3D solution creates immersive content for VR headsets, 3D displays, and image sequences that can be viewed with cross-eyed or parallel viewing techniques - no special glasses required for an authentic binocular vision experience!</p>

<h2>Available 3D Conversion Nodes</h2>
<p>This repository provides powerful stereoscopic image processing and video handling nodes for creating stunning side-by-side 3D content:</p>

<h3>Core 3D Conversion Nodes</h3>
<ul>
  <li><strong>👀 SBS V2 by SamSeen</strong>: Our flagship AI-powered depth estimation node with built-in depth map extraction powered by Depth-Anything-V2! Just feed it your standard 2D images and watch as it automatically creates immersive stereoscopic 3D content perfect for VR videos, 3D animations, and image sequences with authentic depth perception.</li>
  <li><strong>👀 SBS by SamSeen (Legacy)</strong>: For advanced 3D content creators who want complete depth control, this node lets you provide your own custom depth maps to achieve precisely the stereoscopic effect and binocular vision experience you're looking for in your 3D media projects.</li>
</ul>

<h3>Video Processing Nodes</h3>
<ul>
  <li><strong>👀 SBS Video Uploader by SamSeen</strong>: Easily convert videos to image sequences for 3D processing. Features include frame selection, resizing options (default 512px), and frame rate control to prepare your video content for stereoscopic conversion.</li>
  <li><strong>👀 SBS Image Uploader by SamSeen</strong>: A simple yet powerful tool for loading and resizing single images with customizable dimensions, perfect for creating consistent 3D content from various image sources.</li>
  <li><strong>👀 SBS Video Combiner by SamSeen</strong>: Combine your processed stereoscopic image sequences back into stunning 3D videos with support for multiple formats (MP4, WebM, GIF) and audio integration for complete immersive experiences.</li>
</ul>

<h2>Installation for 3D Content Creation</h2>
<p>To install this stereoscopic 3D ComfyUI extension, clone the repository and add it to the custom_nodes folder in your ComfyUI installation directory:</p>
<pre>git clone https://github.com/MrSamSeen/ComfyUI_SSStereoscope.git
cd ComfyUI_SSStereoscope
pip install -r requirements.txt
</pre>

<h2>Detailed 3D Conversion Functionality</h2>

<h3>👀 SBS V2 by SamSeen - Automatic Depth Estimation</h3>
<p>This advanced 2D-to-3D conversion node automatically generates depth maps from your standard images and creates professional stereoscopic 3D content using AI-powered depth perception technology.</p>

<p><strong>Stereoscopic Parameters:</strong></p>
<ul>
  <li><strong>base_image</strong>: The standard 2D input image you want to transform into immersive 3D</li>
  <li><strong>depth_scale</strong>: Controls the intensity of the stereoscopic effect and depth perception (default: 30.0, range: 1.0-100.0)</li>
  <li><strong>blur_radius</strong>: Smooths the depth map edges for more natural binocular vision (default: 3, range: 1-51, must be odd number)</li>
  <li><strong>invert_depth</strong>: Swaps foreground/background elements for different depth-aware imaging effects (default: False)</li>
  <li><strong>mode</strong>: Choose between "Parallel" (for VR headsets) or "Cross-eyed" free-viewing techniques (default: "Cross-eyed")</li>
</ul>

<p><strong>3D Output Formats:</strong></p>
<ul>
  <li><strong>stereoscopic_image</strong>: The side-by-side (SBS) 3D image ready for VR viewing or free-viewing techniques</li>
  <li><strong>depth_map</strong>: The AI-generated depth map showing spatial relationships (lighter areas represent objects closer to the camera)</li>
</ul>

<h3>👀 SBS by SamSeen (Legacy) - Custom Depth Control</h3>
<p>This specialized 3D image processing node requires you to provide your own custom depth map for complete control over the stereoscopic effect in your 3D media projects.</p>

<p><strong>3D Conversion Parameters:</strong></p>
<ul>
  <li><strong>base_image</strong>: The standard 2D input image for your stereoscopic transformation</li>
  <li><strong>depth_map</strong>: Your custom depth map for precise 3D rendering (lighter areas will appear closer in the final stereogram)</li>
  <li><strong>depth_scale</strong>: Controls the intensity of the stereoscopic effect and perceived depth (default: 30)</li>
  <li><strong>mode</strong>: Choose between "Parallel" (for VR displays) or "Cross-eyed" viewing techniques for different binocular vision experiences</li>
</ul>

<p><strong>Stereoscopic Output:</strong></p>
<ul>
  <li><strong>stereoscopic_image</strong>: The professional-quality side-by-side 3D image ready for immersive viewing in VR applications or specialized 3D displays</li>
</ul>

<h2>AI Depth Estimation Model Setup</h2>

<p>The SBS V2 stereoscopic conversion node leverages the powerful Depth-Anything-V2-Small AI model from Hugging Face for accurate depth map extraction and 3D perception. This advanced depth estimation technology will be downloaded automatically on first use, but if you encounter connectivity issues, you can manually install the depth perception model:</p>

<h3>Automatic AI Model Installation (Recommended)</h3>
<p>The depth estimation model will be downloaded automatically when you first use the SBS V2 node for 2D-to-3D conversion. This seamless setup requires an internet connection during the initial stereoscopic processing run.</p>

<h3>Manual Depth Model Installation</h3>
<p>If the automatic depth map extraction model download fails, follow these steps for manual installation of the 3D perception AI:</p>

<ol>
  <li>Visit the model card on Hugging Face: <a href="https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf">depth-anything/Depth-Anything-V2-Small-hf</a></li>
  <li>Download the main model file directly: <a href="https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/blob/main/model.safetensors">model.safetensors</a> (and any other required files from the repository)</li>
  <li>The plugin uses Hugging Face's transformers library for automatic model loading, which stores models in the standard cache directory:
    <pre>~/.cache/huggingface/hub/models--depth-anything--Depth-Anything-V2-Small-hf/snapshots/[COMMIT_HASH]/</pre>
  </li>
  <li>Create this directory structure and place the downloaded model files there</li>
  <li>For the exact path, check your Hugging Face cache directory or look at error messages during failed automatic downloads which will show the expected path</li>
</ol>

<h3>3D Processing Dependencies</h3>
<p>For optimal stereoscopic image processing and depth-aware rendering, ensure you have these dependencies installed:</p>
<pre>
transformers>=4.51.0  # Required for AI depth estimation
torch>=2.0.0          # Neural network processing for 3D conversion
pillow                # Image processing for stereoscopic output
numpy                 # Numerical processing for depth maps
opencv-python         # Computer vision for 3D visualization
tqdm                  # Progress tracking for batch processing
</pre>

<h2>Optimizing Your Stereoscopic 3D Experience</h2>
<p>To achieve professional-quality depth perception and immersive stereoscopic effects in your 3D media projects:</p>
<ol>
  <li>For the "👀 SBS V2 by SamSeen" depth estimation node, ensure optimal AI performance with the latest transformers library: <code>pip install transformers>=4.51.0</code></li>
  <li>The depth-aware AI model will be downloaded automatically during your first 2D-to-3D conversion - be patient during the initial stereoscopic setup process!</li>
  <li>Fine-tune your binocular vision experience by adjusting the depth scale parameter to find the perfect stereoscopic effect for your VR content</li>
  <li>When creating 3D animation sequences or videos, maintain consistent depth settings across frames for smooth, professional stereoscopic results</li>
  <li>For quick depth map extraction testing, run our 3D conversion diagnostic script: <code>python test_sbs_node.py path/to/your/image.jpg</code></li>
  <li>For cross-eyed viewing comfort, start with lower depth scale values and gradually increase as your eyes adjust to the stereoscopic effect</li>
</ol>

<h2>Transform Your Media with Stereoscopic 3D Content Creation</h2>
<p>With our advanced AI-powered depth map extraction technology, creating professional-quality stereoscopic content for immersive viewing experiences has never been more accessible:</p>
<ul>
  <li>🎬 <strong>3D Video Production:</strong> Process each frame through our depth-aware SBS V2 node to create stunning stereoscopic videos with authentic depth perception for VR headsets and 3D displays</li>
  <li>🔄 <strong>Stereoscopic Batch Processing:</strong> Perfect for creating 3D animation sequences with consistent depth maps across frames, ideal for professional VR content creation and immersive storytelling</li>
  <li>🎮 <strong>Virtual Reality Enhancement:</strong> Create depth-rich stereoscopic side-by-side content that brings your virtual worlds to life with binocular vision effects that enhance spatial awareness</li>
  <li>🖼️ <strong>Specialized 3D Display Content:</strong> Generate professional side-by-side stereoscopic images optimized for 3D monitors, TVs, and free-viewing techniques without special glasses</li>
  <li>📱 <strong>Mobile 3D Applications:</strong> Create cross-platform stereoscopic content that works with smartphone VR viewers and mobile 3D applications for on-the-go immersive experiences</li>
</ul>

<h2>Experiencing Your Stereoscopic 3D Creations</h2>

<p>There are multiple ways to enjoy the immersive depth perception in your side-by-side stereoscopic content:</p>

<ol>
  <li><strong>Cross-eyed Free-Viewing Technique</strong>: Focus on a point between the two side-by-side images and gradually cross your eyes until the images merge into a single 3D image with realistic depth in the center - a popular no-glasses 3D viewing method.</li>
  <li><strong>Parallel Viewing for Binocular Vision</strong>: For larger stereoscopic images, relax your eyes as if looking at a distant object, allowing the side-by-side images to naturally merge into a depth-rich 3D scene.</li>
  <li><strong>Virtual Reality Headsets</strong>: Load your SBS stereoscopic images or videos into any VR viewer app for a fully immersive depth perception experience with proper spatial awareness.</li>
  <li><strong>Specialized 3D Displays</strong>: View your stereoscopic content on 3D-capable monitors, TVs, or projectors that support side-by-side 3D format for professional-quality depth visualization.</li>
  <li><strong>Mobile 3D Applications</strong>: Many smartphone apps can transform your side-by-side content into immersive 3D experiences using various stereoscopic viewing methods, perfect for sharing your 3D creations.</li>
  <li><strong>Stereoscopic Viewers</strong>: Use dedicated stereoscopic viewers or simple cardboard viewers that help your eyes focus on the correct side-by-side image pairs for enhanced depth perception.</li>
</ol>

<h2>Troubleshooting Stereoscopic Generation Issues</h2>

<p>If you encounter any challenges with your 3D content creation, try these solutions:</p>

<ol>
  <li><strong>Depth Estimation Model Download Issues</strong>: Verify your internet connection or follow the manual AI model installation instructions above for proper depth map extraction.</li>
  <li><strong>GPU Memory Limitations</strong>: For high-resolution stereoscopic processing, reduce your image dimensions or switch to CPU mode if you have limited graphics memory for AI depth estimation.</li>
  <li><strong>Depth Map Quality Enhancement</strong>: Fine-tune the blur_radius parameter to achieve smoother depth transitions and more natural binocular vision effects in your stereoscopic output.</li>
  <li><strong>Depth Perception Inversion</strong>: If foreground and background elements appear with incorrect spatial relationships, toggle the invert_depth parameter to correct the stereoscopic effect.</li>
  <li><strong>ComfyUI Compatibility</strong>: For optimal 3D content generation, ensure you're using ComfyUI version 1.5.0 or higher with all required dependencies for proper depth-aware processing.</li>
  <li><strong>Stereoscopic Comfort</strong>: If viewing causes eye strain, reduce the depth_scale parameter to create more comfortable side-by-side 3D content with less extreme depth differences.</li>
</ol>

<h2>Stereoscopic 3D Workflow Examples</h2>
<p>Create professional-quality side-by-side 3D content with these example ComfyUI workflows:</p>
<img width="1254" alt="3D conversion workflow with depth estimation" src="https://github.com/MrSamSeen/ComfyUI_SSStereoscope/assets/8541558/6b972838-07ca-4e64-a3a4-32277ddcf4c7">
<p><em>Basic stereoscopic workflow: Generate depth maps automatically and create immersive 3D content with AI-powered depth perception</em></p>

<h2>Advanced 3D Content Creation Workflow</h2>
<img width="696" alt="Advanced stereoscopic processing pipeline" src="https://github.com/MrSamSeen/ComfyUI_SSStereoscope/assets/8541558/1272a5ad-9d12-4b86-8284-08bbe48bd116">
<p><em>Enhanced 3D generation workflow: Combine multiple processing steps for professional stereoscopic results with precise depth control</em></p>

<h2>Complete 3D Video Processing Workflow</h2>
<p>Our integrated video processing tools make creating stereoscopic 3D videos simple and efficient:</p>
<ol>
  <li><strong>Video Input</strong>: Use the 👀 SBS Video Uploader to load your video and convert it to an image sequence with customizable resolution (default 512px)</li>
  <li><strong>Depth Processing</strong>: Feed the image sequence through the 👀 SBS V2 node for automatic depth map generation and stereoscopic conversion</li>
  <li><strong>Video Output</strong>: Combine the processed stereoscopic frames into a final 3D video using the 👀 SBS Video Combiner with your preferred format and frame rate</li>
</ol>
<p><em>This end-to-end workflow eliminates the need for external video processing tools, making professional 3D video creation accessible to everyone!</em></p>

<h2>Stereoscopic 3D Gallery</h2>
<p>Examples of side-by-side 3D images created with our depth-aware image processing technology:</p>
<img width="1254" alt="Stereoscopic landscape with depth perception" src="https://github.com/MrSamSeen/ComfyUI_SSStereoscope/assets/8541558/ee30f773-bb90-420f-a1a2-25459b678bbe">
<p><em>Landscape with AI-generated depth map for immersive stereoscopic viewing</em></p>

<img width="1254" alt="Side-by-side 3D portrait with binocular vision effect" src="https://github.com/MrSamSeen/ComfyUI_SSStereoscope/assets/8541558/e702dde6-b675-4ff6-842c-1c32116be313">
<p><em>Portrait with enhanced depth perception for realistic 3D effect</em></p>

<img width="1254" alt="Virtual reality compatible stereoscopic image" src="https://github.com/MrSamSeen/ComfyUI_SSStereoscope/assets/8541558/022223be-fd55-400a-a1ba-38628265a119">
<p><em>VR-ready stereoscopic image with optimized depth for comfortable viewing</em></p>

<img width="1254" alt="Cross-eyed viewing 3D image with depth map" src="https://github.com/MrSamSeen/ComfyUI_SSStereoscope/assets/8541558/e711ba22-a393-4580-ad37-43934373e835">
<p><em>Detailed scene with precise depth estimation for cross-eyed free-viewing</em></p>

<img width="1254" alt="Side-by-side 3D animation frame with depth" src="https://github.com/MrSamSeen/ComfyUI_SSStereoscope/assets/8541558/59ff5c1c-ceda-42d9-a8eb-b27ffbb0e8ca">
<p><em>Animation frame converted to stereoscopic 3D with automatic depth extraction</em></p>

<img width="1254" alt="Immersive stereoscopic scene with AI depth" src="https://github.com/MrSamSeen/ComfyUI_SSStereoscope/assets/8541558/8b6d7c5e-aefc-4f22-9872-73b690514497">
<p><em>Complex scene with AI-powered depth perception for immersive stereoscopic experience</em></p>

<img width="1254" alt="Professional 3D content for specialized displays" src="https://github.com/MrSamSeen/ComfyUI_SSStereoscope/assets/8541558/46182b48-f5cb-4d6f-9ae9-ba727da92569">
<p><em>Professional-quality stereoscopic image optimized for 3D displays and VR headsets</em></p>

<p><a href="https://civitai.com/models/546410?modelVersionId=607731">Explore more stereoscopic 3D examples on CivitAI</a></p>

<h2>Join Our 3D Content Creation Community</h2>

<p>We welcome contributions to enhance this stereoscopic imaging tool! Whether you're experienced with depth estimation algorithms, 3D visualization techniques, or just passionate about creating immersive content, feel free to submit pull requests or open issues with your suggestions for improving the depth-aware processing capabilities.</p>

<h2>Open Source License</h2>

<p>This stereoscopic 3D conversion project is licensed under the MIT License - see the LICENSE file for complete details. Feel free to use our depth estimation technology in your own 3D content creation projects while providing appropriate attribution.</p>

<h2>About This Project</h2>

<p>This ComfyUI custom node represents the cutting edge of AI-powered stereoscopic 3D content creation. By leveraging advanced depth estimation technology, it enables seamless 2D to 3D conversion without requiring specialized hardware or external depth maps. Whether you're creating immersive virtual reality experiences, enhancing your photography with depth perception, or producing eye-catching side-by-side 3D animations, this tool streamlines your stereoscopic workflow.</p>

<p>The depth-aware imaging capabilities automatically identify foreground and background elements, creating realistic depth perception that brings your images to life. Perfect for 3D visualization projects, cross-eyed viewing techniques, or parallel view stereograms that don't require special glasses. From single images to batch processing entire animation sequences, transform your media creation process with professional-quality binocular vision effects.</p>
