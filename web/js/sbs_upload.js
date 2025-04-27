import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// Register our custom widget
app.registerExtension({
  name: "SBS.FileUpload",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    // Check if this is one of our nodes that needs a file upload widget
    if (
      nodeData.name === "SBS_Video_Uploader" ||
      nodeData.name === "SBS_Image_Uploader"
    ) {
      // Store the original onNodeCreated function
      const onNodeCreated = nodeType.prototype.onNodeCreated;

      // Override the onNodeCreated function
      nodeType.prototype.onNodeCreated = function () {
        // Call the original onNodeCreated function
        const result = onNodeCreated
          ? onNodeCreated.apply(this, arguments)
          : undefined;

        // Get the widget name based on the node type
        const widgetName =
          nodeData.name === "SBS_Video_Uploader" ? "video" : "image";

        // Find the widget for the file selection
        const fileWidget = this.widgets.find((w) => w.name === widgetName);
        if (!fileWidget) return result;

        // Create the upload button
        this.addWidget("button", "Upload " + widgetName, "upload", () => {
          // Create a file input element
          const input = document.createElement("input");
          input.type = "file";

          // Set accepted file types
          if (nodeData.name === "SBS_Video_Uploader") {
            input.accept = ".mp4,.webm,.mkv,.gif,.mov";
          } else {
            input.accept = ".png,.jpg,.jpeg,.webp,.bmp";
          }

          // Handle file selection
          input.addEventListener("change", async () => {
            if (!input.files.length) return;

            const file = input.files[0];

            // Create a FormData object to send the file
            const formData = new FormData();
            formData.append("image", file);
            formData.append("overwrite", "true");
            formData.append("type", "input");

            try {
              // Upload the file
              const resp = await api.fetchApi("/upload/image", {
                method: "POST",
                body: formData,
              });

              if (resp.status === 200) {
                const data = await resp.json();
                // Set the file widget value to the uploaded file name
                if (data.name) {
                  // Update the file widget value
                  fileWidget.value = data.name;

                  // Call the updatePreview function directly
                  updatePreview(data.name);

                  // No need to preload anymore, the widget will handle it

                  // Notify ComfyUI that the node has changed
                  app.graph.setDirtyCanvas(true);
                }
              } else {
                alert("Error uploading file: " + resp.statusText);
              }
            } catch (error) {
              alert("Error uploading file: " + error.message);
            }
          });

          // Trigger the file selection dialog
          input.click();
        });

        // Add a preview widget using ComfyUI's built-in widgets
        const previewWidget = this.addWidget("image", "Preview", "", "");

        // Try to make the preview larger by adding a style
        setTimeout(() => {
          const imgElement = this.widgets
            .find((w) => w.name === "Preview")
            ?.inputEl?.querySelector("img");
          if (imgElement) {
            imgElement.style.maxHeight = "200px";
            imgElement.style.objectFit = "contain";
          }
        }, 100);

        // Add an info widget for file details
        const infoWidget = this.addWidget("text", "Info", "No file selected");

        // Function to update the preview
        const updatePreview = (value) => {
          if (value && value !== "none") {
            // Update the preview image
            const path = `/input/${value}?t=${Date.now()}`;

            // Get file extension
            const ext = value.split(".").pop().toLowerCase();

            // Set the preview image
            previewWidget.value = path;

            // Show different info based on file type
            if (nodeData.name === "SBS_Video_Uploader") {
              // For videos, show file info
              infoWidget.value = `Video: ${value}\nType: ${ext}`;
            } else {
              // For images, show file info
              infoWidget.value = `Image: ${value}\nType: ${ext}`;
            }
          } else {
            // Clear the preview
            previewWidget.value = "";
            infoWidget.value = "No file selected";
          }
        };

        // Update the preview when the file selection changes
        const onFileChange = fileWidget.callback;
        fileWidget.callback = function (value) {
          const result = onFileChange
            ? onFileChange.call(this, value)
            : undefined;

          // Update the preview
          updatePreview(value);

          // Try to make the preview larger by adding a style (do this again after value changes)
          setTimeout(() => {
            const imgElement = this.widgets
              .find((w) => w.name === "Preview")
              ?.inputEl?.querySelector("img");
            if (imgElement) {
              imgElement.style.maxHeight = "200px";
              imgElement.style.objectFit = "contain";
            }
          }, 100);

          return result;
        };

        return result;
      };
    }
  },
});
