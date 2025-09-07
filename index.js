
import { removeImageBackground } from './services/geminiService.js';

// --- CONSTANTS ---
const FACE_API_MODELS_URL = 'https://justadudewhohacks.github.io/face-api.js/models';
const HEADSHOT_WIDTH = 200;
const HEADSHOT_HEIGHT = 300;
const MIN_CROP_WIDTH = 300; // Min width from original image to ensure quality
const MAX_API_DIMENSION = 1024; // Max dimension for analysis image
const HEADSHOT_COMPOSITION = {
  // The final crop height will be faceBox.height / HEAD_DOMINANCE_RATIO.
  // A ratio of 0.7 means the face will take up ~70% of the final image height.
  HEAD_DOMINANCE_RATIO: 0.7,
  // Percentage of the final crop height to place above the detected face as headroom.
  // Reduced to 5% for a tighter, more professional composition.
  HEADROOM_RATIO: 0.05,
};

// --- STATE MANAGEMENT ---
const state = {
  file: null,
  originalImageBase64: null,
  croppedImage: null,
  isLoading: false, // For image processing after upload
  modelsLoading: true, // For initial model loading
  error: null,
};

// --- DOM ELEMENTS ---
let mainContainer, uploaderView, loaderView, loaderMessage, resultView, errorContainer, errorMessage;
let dropzone, fileInput, previewImage, uploadPrompt, generateButton, resetButton, croppedImage, downloadLink;


// --- UI & STATE HELPERS ---

/**
 * Updates the loading state and message, then triggers a re-render.
 * @param {boolean} isLoading - Whether the app is in a loading state.
 * @param {string} [message=''] - The message to display on the loader.
 */
function setLoading(isLoading, message = '') {
  state.isLoading = isLoading;
  if (state.isLoading) {
    loaderMessage.textContent = message;
  }
  render();
}

/**
 * Updates the UI based on the current state.
 */
function render() {
  // Toggle error message visibility
  errorContainer.classList.toggle('hidden', !state.error);
  if (state.error) {
    errorMessage.textContent = state.error;
  }

  const showResult = !!state.croppedImage;

  // Toggle between main view and result view
  mainContainer.classList.toggle('hidden', showResult);
  resultView.classList.toggle('hidden', !showResult);
  resultView.classList.toggle('flex', showResult);

  if (showResult) {
    croppedImage.src = state.croppedImage;
    downloadLink.href = state.croppedImage;
  } else {
    const showLoader = state.isLoading || state.modelsLoading;
    loaderView.classList.toggle('hidden', !showLoader);
    loaderView.classList.toggle('flex', showLoader);
    uploaderView.classList.toggle('hidden', showLoader);

    if (showLoader) {
        if (state.modelsLoading) {
            loaderMessage.textContent = 'Preparing AI models...';
        }
        // If state.isLoading is true, the message is already set by setLoading()
    } else { // Uploader is visible
      const hasPreview = !!state.originalImageBase64;
      previewImage.classList.toggle('hidden', !hasPreview);
      uploadPrompt.classList.toggle('hidden', hasPreview);
      generateButton.classList.toggle('hidden', !hasPreview);
      if (hasPreview) {
        previewImage.src = state.originalImageBase64;
      }
    }
  }

  // Disable interactive elements while loading
  const isBusy = state.isLoading || state.modelsLoading;
  generateButton.disabled = isBusy;
  resetButton.disabled = isBusy;
  fileInput.disabled = isBusy;
  dropzone.style.cursor = isBusy ? 'not-allowed' : 'pointer';
}


// --- IMAGE UTILITIES ---

/**
 * Loads an image from a URL and returns a promise that resolves with the HTMLImageElement.
 * @param {string} src - The source URL of the image.
 * @returns {Promise<HTMLImageElement>}
 */
function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = (err) => reject(new Error("Failed to load image for processing.", { cause: err }));
    img.src = src;
  });
}

/**
 * Resizes an image for API analysis, maintaining aspect ratio.
 */
async function resizeForAnalysis(imageUrl, maxDimension) {
  const img = await loadImage(imageUrl);
  const { naturalWidth: width, naturalHeight: height } = img;
  
  if (Math.max(width, height) <= maxDimension) {
    const mimeType = imageUrl.substring(imageUrl.indexOf(":") + 1, imageUrl.indexOf(";"));
    return { resizedImageUrl: imageUrl, width, height, mimeType };
  }

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  
  let newWidth, newHeight;
  if (width > height) {
    newWidth = maxDimension;
    newHeight = (height * newWidth) / width;
  } else {
    newHeight = maxDimension;
    newWidth = (width * newHeight) / height;
  }
  canvas.width = newWidth;
  canvas.height = newHeight;

  ctx.drawImage(img, 0, 0, newWidth, newHeight);
  const mimeType = imageUrl.substring(imageUrl.indexOf(":") + 1, imageUrl.indexOf(";"));
  return { resizedImageUrl: canvas.toDataURL(mimeType), width: newWidth, height: newHeight, mimeType };
}

/**
 * Calculates a 2:3 portrait aspect ratio headshot box from a face detection box.
 */
function createHeadshotBox(faceBox, imageWidth, imageHeight) {
  const aspectRatio = HEADSHOT_WIDTH / HEADSHOT_HEIGHT;
  // Calculate the total height of the crop box so the face takes up the desired percentage.
  let headshotHeight = faceBox.height / HEADSHOT_COMPOSITION.HEAD_DOMINANCE_RATIO;
  let headshotWidth = headshotHeight * aspectRatio;

  // Upscale if the detected face is too small to ensure quality
  if (headshotWidth < MIN_CROP_WIDTH) {
    headshotWidth = MIN_CROP_WIDTH;
    headshotHeight = headshotWidth / aspectRatio;
  }

  // Downscale if the calculated box is larger than the image itself
  if (headshotWidth > imageWidth) {
    headshotWidth = imageWidth;
    headshotHeight = headshotWidth / aspectRatio;
  }
  if (headshotHeight > imageHeight) {
    headshotHeight = imageHeight;
    headshotWidth = headshotHeight * aspectRatio;
  }

  // Center the box horizontally on the face, and position vertically with headroom
  let x = faceBox.x + faceBox.width / 2 - headshotWidth / 2;
  // Position the box with a small amount of headroom above the face
  let y = faceBox.y - headshotHeight * HEADSHOT_COMPOSITION.HEADROOM_RATIO;

  // Clamp the box to stay within the image boundaries
  x = Math.max(0, Math.min(x, imageWidth - headshotWidth));
  y = Math.max(0, Math.min(y, imageHeight - headshotHeight));

  return { x: Math.round(x), y: Math.round(y), width: Math.round(headshotWidth), height: Math.round(headshotHeight) };
}

/**
 * Crops a portion of an image based on a bounding box.
 */
async function cropImage(imageUrl, box) {
  const img = await loadImage(imageUrl);
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  canvas.width = box.width;
  canvas.height = box.height;
  ctx.drawImage(img, box.x, box.y, box.width, box.height, 0, 0, box.width, box.height);
  const mimeType = imageUrl.substring(imageUrl.indexOf(":") + 1, imageUrl.indexOf(";"));
  return { imageDataUrl: canvas.toDataURL(mimeType), mimeType };
}

/**
 * Resizes an image to the final headshot dimensions and adds a white background.
 */
async function resizeFinalImage(imageUrl, targetWidth, targetHeight) {
  const img = await loadImage(imageUrl);
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  canvas.width = targetWidth;
  canvas.height = targetHeight;

  // Fill the background with solid white
  ctx.fillStyle = '#FFFFFF';
  ctx.fillRect(0, 0, targetWidth, targetHeight);

  // Draw the (potentially transparent) image on top of the white background
  ctx.drawImage(img, 0, 0, targetWidth, targetHeight);
  
  // Return as JPEG, which is suitable for solid backgrounds and generally smaller.
  return canvas.toDataURL("image/jpeg");
}


// --- CORE LOGIC & EVENT HANDLERS ---

/**
 * Loads face-api.js models from a remote URL.
 */
async function loadModels() {
  try {
    await faceapi.nets.ssdMobilenetv1.loadFromUri(FACE_API_MODELS_URL);
    state.modelsLoading = false;
  } catch (err) {
    state.error = 'Could not load face detection models. Please refresh the page to try again.';
    console.error('Failed to load face-api models:', err);
  } finally {
    render();
  }
}

/**
 * Finds the most prominent face in an image using face-api.js.
 * @param {HTMLImageElement} imageElement - The image to analyze.
 * @returns {Promise<{x: number, y: number, width: number, height: number}>} A promise that resolves to the bounding box of the detected face.
 */
async function findHeadshot(imageElement) {
  const detection = await faceapi.detectSingleFace(imageElement);
  if (!detection) {
    throw new Error("Could not find a face in the image. Please try another one.");
  }
  return detection.box;
}

/**
 * Handles the user selecting an image file.
 * @param {File} file - The selected file.
 */
function handleImageSelect(file) {
  if (!file || !file.type.startsWith('image/')) {
    state.error = "Please upload a valid image file.";
    render();
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    state.file = file;
    state.originalImageBase64 = e.target.result;
    state.croppedImage = null;
    state.error = null;
    render();
  };
  reader.onerror = () => {
      state.error = "Could not read the selected file.";
      render();
  }
  reader.readAsDataURL(file);
}

/**
 * Orchestrates the full headshot generation process.
 */
async function handleCropRequest() {
  if (!state.file || !state.originalImageBase64) {
    state.error = "Please select an image first.";
    render();
    return;
  }
  state.error = null;
  setLoading(true, 'Initializing...');

  try {
    // 1. Get original image dimensions
    const originalImage = await loadImage(state.originalImageBase64);
    const { naturalWidth: originalWidth, naturalHeight: originalHeight } = originalImage;
    
    // 2. Resize for efficient client-side analysis
    setLoading(true, 'Resizing for analysis...');
    const { resizedImageUrl, width: resizedWidth, height: resizedHeight } = await resizeForAnalysis(state.originalImageBase64, MAX_API_DIMENSION);

    // 3. Detect the face in the resized image using face-api.js
    setLoading(true, 'Detecting face...');
    const analysisImage = await loadImage(resizedImageUrl);
    const faceBoxResized = await findHeadshot(analysisImage);
    
    // 4. Scale the detected face box back to original image coordinates
    const scaleFactorX = originalWidth / resizedWidth;
    const scaleFactorY = originalHeight / resizedHeight;
    const faceBoxOriginal = {
      x: faceBoxResized.x * scaleFactorX,
      y: faceBoxResized.y * scaleFactorY,
      width: faceBoxResized.width * scaleFactorX,
      height: faceBoxResized.height * scaleFactorY,
    };
    
    // 5. Calculate the final headshot crop box
    const headshotBox = createHeadshotBox(faceBoxOriginal, originalWidth, originalHeight);

    // 6. Crop the headshot from the original, high-quality image
    setLoading(true, 'Cropping headshot...');
    const { imageDataUrl: initialCropUrl, mimeType: cropMimeType } = await cropImage(state.originalImageBase64, headshotBox);

    // 7. Remove the background from the cropped image using Gemini API
    setLoading(true, 'Removing background...');
    const transparentImageUrl = await removeImageBackground(initialCropUrl, cropMimeType);

    // 8. Resize the final image to the target dimensions
    setLoading(true, 'Finalizing image...');
    state.croppedImage = await resizeFinalImage(transparentImageUrl, HEADSHOT_WIDTH, HEADSHOT_HEIGHT);

  } catch (err) {
    console.error(err);
    const errorMessage = err instanceof Error ? err.message : "An unexpected error occurred.";
    state.error = `Failed to process image. ${errorMessage}`;
  } finally {
    setLoading(false);
  }
}

/**
 * Resets the application to its initial state.
 */
function handleReset() {
  state.file = null;
  state.originalImageBase64 = null;
  state.croppedImage = null;
  state.error = null;
  state.isLoading = false;
  fileInput.value = ''; // Clear the file input
  render();
}

/**
 * Sets up all necessary event listeners for the application.
 */
function setupEventListeners() {
  fileInput.addEventListener('change', (e) => handleImageSelect(e.target.files[0]));
  
  dropzone.addEventListener('dragover', (e) => e.preventDefault());
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    if (!state.isLoading && !state.modelsLoading) {
      handleImageSelect(e.dataTransfer.files[0]);
    }
  });

  generateButton.addEventListener('click', handleCropRequest);
  resetButton.addEventListener('click', handleReset);
}


// --- INITIALIZATION ---
document.addEventListener('DOMContentLoaded', () => {
  // Cache all DOM element references
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
  
  // Attach event listeners and perform the initial render to show model loading state
  setupEventListeners();
  render();
  // Load the face detection models
  loadModels();
});
