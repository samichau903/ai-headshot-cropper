import { GoogleGenAI, Type } from "@google/genai";

// --- STATE MANAGEMENT ---
const state = {
  file: null,
  originalImageBase64: null,
  croppedImage: null,
  isLoading: false,
  error: null,
};

// --- DOM ELEMENTS ---
let mainContainer, uploaderView, loaderView, resultView, errorContainer, errorMessage;
let dropzone, fileInput, previewImage, uploadPrompt, generateButton, resetButton, croppedImage, downloadLink;

// --- GEMINI API SERVICE ---
async function findHeadshot(base64Image, mimeType) {
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: {
        parts: [
          {
            inlineData: {
              data: base64Image.split(',')[1], // Remove the data URL prefix
              mimeType: mimeType,
            },
          },
          {
            text: `Analyze the uploaded image. Your task is to identify the primary human face and provide its bounding box coordinates. Respond ONLY in JSON format. The coordinates must be integers and represent the top-left corner (x, y) and the dimensions (width, height) of the bounding box relative to the original image size.`,
          },
        ],
      },
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            x: { type: Type.INTEGER },
            y: { type: Type.INTEGER },
            width: { type: Type.INTEGER },
            height: { type: Type.INTEGER },
          },
          required: ["x", "y", "width", "height"],
        },
      },
    });

    const jsonString = response.text;
    const parsedJson = JSON.parse(jsonString);

    if (
        typeof parsedJson.x !== 'number' ||
        typeof parsedJson.y !== 'number' ||
        typeof parsedJson.width !== 'number' ||
        typeof parsedJson.height !== 'number'
    ) {
        throw new Error("Invalid bounding box data received from API.");
    }

    return parsedJson;

  } catch (error) {
    console.error("Error calling Gemini API:", error);
    if (error instanceof Error) {
        if (error.message.includes('API key not valid')) {
            throw new Error('The API Key is invalid. Please ensure it is configured correctly in the environment.');
        }
        throw new Error(`Failed to find headshot: ${error.message}`);
    }
    throw new Error("An unknown error occurred while analyzing the image.");
  }
}

// --- IMAGE UTILITIES ---

const HEADSHOT_DIMENSION = 512; // The target size for the final headshot (e.g., 512x512 pixels).

/**
 * Calculates a 1:1 aspect ratio headshot bounding box based on a face detection box.
 * This function takes the detected face and calculates a larger, square box
 * that is better suited for a headshot.
 * @param {object} faceBox - The bounding box of the detected face {x, y, width, height}.
 * @param {number} imageWidth - The width of the original image.
 * @param {number} imageHeight - The height of the original image.
 * @returns {object} The calculated bounding box for the headshot.
 */
function createHeadshotBox(faceBox, imageWidth, imageHeight) {
    // The ideal side length for the headshot square. 2x the face width is a good starting point.
    const desiredSide = faceBox.width * 2;
    
    // The final side length cannot be larger than the image itself.
    const side = Math.min(desiredSide, imageWidth, imageHeight);

    // Center the crop box horizontally on the detected face.
    let x = faceBox.x + faceBox.width / 2 - side / 2;

    // Center the crop box vertically, but shift it up slightly to give more headroom, typical for headshots.
    let y = faceBox.y + faceBox.height / 2 - side / 2 - faceBox.height * 0.15;

    // Clamp the top-left corner to ensure the final box stays within image boundaries.
    x = Math.max(0, Math.min(x, imageWidth - side));
    y = Math.max(0, Math.min(y, imageHeight - side));

    return {
        x: Math.round(x),
        y: Math.round(y),
        width: Math.round(side),
        height: Math.round(side),
    };
}


/**
 * Crops a portion of an image and resizes it to a target dimension.
 * @param {string} imageUrl - The data URL of the source image.
 * @param {object} box - The bounding box {x, y, width, height} to crop from the source image.
 * @param {number} targetSize - The width and height of the output image.
 * @returns {Promise<string>} A promise that resolves with the data URL of the cropped and resized image.
 */
function cropAndResizeImage(imageUrl, box, targetSize) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        return reject(new Error("Could not get canvas context"));
      }
      
      canvas.width = targetSize;
      canvas.height = targetSize;
      
      // Draw the cropped portion of the source image onto the canvas, resizing it in the process.
      ctx.drawImage(
        img,
        box.x,
        box.y,
        box.width,
        box.height,
        0,
        0,
        targetSize,
        targetSize
      );
      
      const resultDataUrl = canvas.toDataURL("image/png");
      resolve(resultDataUrl);
    };
    img.onerror = () => {
      reject(new Error("Failed to load image for cropping and resizing."));
    };
    img.src = imageUrl;
  });
}

// --- UI UPDATE FUNCTION ---
function render() {
  // Error display
  if (state.error) {
    errorMessage.textContent = state.error;
    errorContainer.classList.remove('hidden');
  } else {
    errorContainer.classList.add('hidden');
  }

  // Main view logic
  if (state.croppedImage) {
    mainContainer.classList.add('hidden');
    resultView.classList.remove('hidden');
    resultView.classList.add('flex');
    croppedImage.src = state.croppedImage;
    downloadLink.href = state.croppedImage;
  } else {
    mainContainer.classList.remove('hidden');
    resultView.classList.add('hidden');
    resultView.classList.remove('flex');
    
    if (state.isLoading) {
      loaderView.classList.remove('hidden');
      loaderView.classList.add('flex');
      uploaderView.classList.add('hidden');
    } else {
      loaderView.classList.add('hidden');
      loaderView.classList.remove('flex');
      uploaderView.classList.remove('hidden');
      
      if (state.originalImageBase64) {
        previewImage.src = state.originalImageBase64;
        previewImage.classList.remove('hidden');
        uploadPrompt.classList.add('hidden');
        generateButton.classList.remove('hidden');
      } else {
        previewImage.classList.add('hidden');
        uploadPrompt.classList.remove('hidden');
        generateButton.classList.add('hidden');
      }
    }
  }

  // Disable buttons
  generateButton.disabled = state.isLoading;
  resetButton.disabled = state.isLoading;
  fileInput.disabled = state.isLoading;
}

// --- EVENT HANDLERS ---
function handleImageSelect(file) {
  if (file && file.type.startsWith('image/')) {
    const reader = new FileReader();
    reader.onload = (e) => {
      state.file = file;
      state.originalImageBase64 = e.target.result;
      state.croppedImage = null;
      state.error = null;
      render();
    };
    reader.readAsDataURL(file);
  } else {
    state.error = "Please upload a valid image file.";
    render();
  }
}

async function handleCropRequest() {
  if (!state.file || !state.originalImageBase64) {
    state.error = "Please select an image first.";
    render();
    return;
  }

  state.isLoading = true;
  state.error = null;
  render();

  try {
    // Get image dimensions by loading the image in memory
    const img = new Image();
    await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = () => reject(new Error("Failed to load image to get dimensions."));
        img.src = state.originalImageBase64;
    });
    const { naturalWidth: imageWidth, naturalHeight: imageHeight } = img;

    // 1. Ask the AI to find the face in the image
    const faceBox = await findHeadshot(state.originalImageBase64, state.file.type);
    
    // 2. Use the face location to calculate a proper headshot crop
    const headshotBox = createHeadshotBox(faceBox, imageWidth, imageHeight);

    // 3. Perform the crop and resize on the original image using the calculated box
    const croppedDataUrl = await cropAndResizeImage(state.originalImageBase64, headshotBox, HEADSHOT_DIMENSION);
    state.croppedImage = croppedDataUrl;
  } catch (err) {
    console.error(err);
    const errorMessage = err instanceof Error ? err.message : "An unexpected error occurred.";
    state.error = `Failed to process image. ${errorMessage}`;
  } finally {
    state.isLoading = false;
    render();
  }
}

function handleReset() {
  state.file = null;
  state.originalImageBase64 = null;
  state.croppedImage = null;
  state.error = null;
  state.isLoading = false;
  fileInput.value = ''; // Reset file input
  render();
}

function setupEventListeners() {
    fileInput.addEventListener('change', (e) => handleImageSelect(e.target.files[0]));
    
    dropzone.addEventListener('dragenter', (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (!state.isLoading) dropzone.classList.add('border-blue-400');
    });
    
    dropzone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove('border-blue-400');
    });

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove('border-blue-400');
        if (!state.isLoading) {
            handleImageSelect(e.dataTransfer.files[0]);
        }
    });

    generateButton.addEventListener('click', handleCropRequest);
    resetButton.addEventListener('click', handleReset);
}


// --- INITIALIZATION ---
document.addEventListener('DOMContentLoaded', () => {
    // Select all elements once the DOM is ready
    mainContainer = document.getElementById('main-container');
    uploaderView = document.getElementById('uploader-view');
    loaderView = document.getElementById('loader-view');
    resultView = document.getElementById('result-view');
    errorContainer = document.getElementById('error-container');
    errorMessage = document.getElementById('error-message');
    dropzone = document.getElementById('dropzone');
    fileInput = document.getElementById('file-input');
    previewImage = document.getElementById('preview-image');
    uploadPrompt = document.getElementById('upload-prompt');
    generateButton = document.getElementById('generate-button');
    resetButton = document.getElementById('reset-button');
    croppedImage = document.getElementById('cropped-image');
    downloadLink = document.getElementById('download-link');
    
    setupEventListeners();
    render(); // Initial render
});