import { GoogleGenAI, Type, Modality } from "@google/genai";

// --- STATE MANAGEMENT ---
const state = {
  file: null,
  originalImageBase64: null,
  croppedImage: null,
  isLoading: false,
  error: null,
};

// --- DOM ELEMENTS ---
let mainContainer, uploaderView, loaderView, loaderMessage, resultView, errorContainer, errorMessage;
let dropzone, fileInput, previewImage, uploadPrompt, generateButton, resetButton, croppedImage, downloadLink;

// --- CONSTANTS ---
const HEADSHOT_WIDTH = 200;
const HEADSHOT_HEIGHT = 300;
const MIN_CROP_WIDTH = 300; // Min width from original image to ensure quality
const MAX_API_DIMENSION = 1024; // Max dimension for analysis image


// --- GEMINI API SERVICES ---

/**
 * Finds the most prominent face in an image and returns its bounding box.
 */
async function findHeadshot(base64Image, mimeType) {
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: {
        parts: [
          { inlineData: { data: base64Image.split(',')[1], mimeType: mimeType } },
          { text: `Analyze the image to find the most prominent person's face for a student headshot. Provide the bounding box from chin to top of hair. Focus on the main subject, even with a noisy background. Respond with only a JSON object containing integer coordinates: x, y, width, height.` },
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
    const parsedJson = JSON.parse(response.text);
    if (typeof parsedJson.x !== 'number' || typeof parsedJson.y !== 'number' || typeof parsedJson.width !== 'number' || typeof parsedJson.height !== 'number') {
      throw new Error("Invalid bounding box data received from API.");
    }
    return parsedJson;
  } catch (error) {
    console.error("Error calling Gemini API for face detection:", error);
    if (error instanceof Error && error.message.includes('API key not valid')) {
      throw new Error('The API Key is invalid. Please ensure it is configured correctly.');
    }
    throw new Error(`Failed to find headshot: ${error.message || "Unknown API error."}`);
  }
}

/**
 * Removes the background from an image, making it transparent.
 */
async function removeImageBackground(base64Image, mimeType) {
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash-image-preview',
      contents: {
        parts: [
          { inlineData: { data: base64Image.split(',')[1], mimeType: mimeType } },
          { text: 'Isolate the main person from the background. Make the background fully transparent. The output should be only the resulting image with a transparent background.' },
        ],
      },
      config: {
        responseModalities: [Modality.IMAGE, Modality.TEXT],
      },
    });
    for (const part of response.candidates[0].content.parts) {
      if (part.inlineData) {
        return `data:image/png;base64,${part.inlineData.data}`;
      }
    }
    throw new Error("The AI did not return an image. It may have returned text instead: " + response.text);
  } catch (error) {
    console.error("Error removing background:", error);
    throw new Error(`Failed to remove background: ${error.message}`);
  }
}


// --- IMAGE UTILITIES ---

/**
 * Resizes an image for API analysis, maintaining aspect ratio.
 */
function resizeForAnalysis(imageUrl, maxDimension) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const { naturalWidth: width, naturalHeight: height } = img;
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      if (!ctx) return reject(new Error("Could not get canvas context"));
      let newWidth, newHeight;
      if (width > height) {
        newWidth = Math.min(width, maxDimension);
        newHeight = Math.round((height * newWidth) / width);
      } else {
        newHeight = Math.min(height, maxDimension);
        newWidth = Math.round((width * newHeight) / height);
      }
      canvas.width = newWidth;
      canvas.height = newHeight;
      ctx.drawImage(img, 0, 0, newWidth, newHeight);
      const mimeType = imageUrl.substring(imageUrl.indexOf(":") + 1, imageUrl.indexOf(";"));
      resolve({ resizedImageUrl: canvas.toDataURL(mimeType), width: newWidth, height: newHeight, mimeType });
    };
    img.onerror = () => reject(new Error("Failed to load image for resizing."));
    img.src = imageUrl;
  });
}

/**
 * Calculates a 2:3 portrait aspect ratio headshot box.
 */
function createHeadshotBox(faceBox, imageWidth, imageHeight) {
    const aspectRatio = HEADSHOT_WIDTH / HEADSHOT_HEIGHT;
    
    // A smaller multiplier makes the crop tighter, so the person appears larger.
    let headshotHeight = faceBox.height * 2.0;
    let headshotWidth = headshotHeight * aspectRatio;

    if (headshotWidth < MIN_CROP_WIDTH) {
        headshotWidth = MIN_CROP_WIDTH;
        headshotHeight = headshotWidth / aspectRatio;
    }

    if (headshotWidth > imageWidth) {
        headshotWidth = imageWidth;
        headshotHeight = headshotWidth / aspectRatio;
    }
    if (headshotHeight > imageHeight) {
        headshotHeight = imageHeight;
        headshotWidth = headshotHeight * aspectRatio;
    }

    let x = faceBox.x + faceBox.width / 2 - headshotWidth / 2;
    // Position the box vertically. A smaller multiplier for the y-offset places
    // the top of the head closer to the top of the frame.
    // 0.05 results in ~5% headroom (e.g., 15px on a 300px tall image).
    let y = faceBox.y - headshotHeight * 0.05;

    x = Math.max(0, Math.min(x, imageWidth - headshotWidth));
    y = Math.max(0, Math.min(y, imageHeight - headshotHeight));

    return { x: Math.round(x), y: Math.round(y), width: Math.round(headshotWidth), height: Math.round(headshotHeight) };
}


/**
 * Crops a portion of an image without resizing.
 */
function cropImage(imageUrl, box) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      if (!ctx) return reject(new Error("Could not get canvas context"));
      canvas.width = box.width;
      canvas.height = box.height;
      ctx.drawImage(img, box.x, box.y, box.width, box.height, 0, 0, box.width, box.height);
      const mimeType = imageUrl.substring(imageUrl.indexOf(":") + 1, imageUrl.indexOf(";"));
      resolve({ imageDataUrl: canvas.toDataURL(mimeType), mimeType: mimeType });
    };
    img.onerror = () => reject(new Error("Failed to load image for cropping."));
    img.src = imageUrl;
  });
}

/**
 * Resizes an image to a final target size.
 */
function resizeFinalImage(imageUrl, targetWidth, targetHeight) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      if (!ctx) return reject(new Error("Could not get canvas context"));
      canvas.width = targetWidth;
      canvas.height = targetHeight;
      ctx.drawImage(img, 0, 0, targetWidth, targetHeight);
      resolve(canvas.toDataURL("image/png")); // Output PNG for transparency
    };
    img.onerror = () => reject(new Error("Failed to load image for final resizing."));
    img.src = imageUrl;
  });
}


// --- UI UPDATE FUNCTION ---
function render() {
  if (state.error) {
    errorMessage.textContent = state.error;
    errorContainer.classList.remove('hidden');
  } else {
    errorContainer.classList.add('hidden');
  }

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
    loaderMessage.textContent = 'Getting image dimensions...';
    const img = new Image();
    await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = () => reject(new Error("Failed to load image to get dimensions."));
        img.src = state.originalImageBase64;
    });
    const { naturalWidth: originalWidth, naturalHeight: originalHeight } = img;
    
    loaderMessage.textContent = 'Resizing for analysis...';
    const { resizedImageUrl, width: resizedWidth, height: resizedHeight, mimeType } = await resizeForAnalysis(state.originalImageBase64, MAX_API_DIMENSION);

    loaderMessage.textContent = 'Detecting face...';
    const faceBoxResized = await findHeadshot(resizedImageUrl, mimeType);
    
    const scaleFactorX = originalWidth / resizedWidth;
    const scaleFactorY = originalHeight / resizedHeight;
    const faceBoxOriginal = {
        x: faceBoxResized.x * scaleFactorX,
        y: faceBoxResized.y * scaleFactorY,
        width: faceBoxResized.width * scaleFactorX,
        height: faceBoxResized.height * scaleFactorY,
    };
    
    const headshotBox = createHeadshotBox(faceBoxOriginal, originalWidth, originalHeight);

    loaderMessage.textContent = 'Cropping headshot...';
    const { imageDataUrl: initialCropUrl, mimeType: cropMimeType } = await cropImage(state.originalImageBase64, headshotBox);

    loaderMessage.textContent = 'Removing background...';
    const transparentImageUrl = await removeImageBackground(initialCropUrl, cropMimeType);

    loaderMessage.textContent = 'Finalizing image...';
    const finalImageUrl = await resizeFinalImage(transparentImageUrl, HEADSHOT_WIDTH, HEADSHOT_HEIGHT);
    state.croppedImage = finalImageUrl;

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
  fileInput.value = '';
  render();
}

function setupEventListeners() {
    fileInput.addEventListener('change', (e) => handleImageSelect(e.target.files[0]));
    dropzone.addEventListener('dragover', (e) => e.preventDefault());
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        if (!state.isLoading) handleImageSelect(e.dataTransfer.files[0]);
    });
    generateButton.addEventListener('click', handleCropRequest);
    resetButton.addEventListener('click', handleReset);
}

// --- INITIALIZATION ---
document.addEventListener('DOMContentLoaded', () => {
    mainContainer = document.getElementById('main-container');
    uploaderView = document.getElementById('uploader-view');
    loaderView = document.getElementById('loader-view');
    loaderMessage = document.getElementById('loader-message');
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
    render();
});
