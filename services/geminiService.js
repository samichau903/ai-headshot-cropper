

import { GoogleGenAI, Modality } from "@google/genai";

const PROMPT = `Your task is to act as a precision digital editing tool. 
  You will extract "pure objects" from reference images and apply them to the main subject in Image, 
  or modify the main subject according to text instructions, while strictly adhering to the highest command. 
  1. Isolate the main person from the background. The background MUST be 100% white (alpha channel 0). 
  2. Do not add any color, shadows, or gradients to the background. 
  3. Change the pose of the headshot in the image to Front view (upper body from chest up)
  4. The output must be a PNG image containing only the person on a white background. 
  EXPLICIT PROHIBITIONS:
  1.  **FORBIDDEN** to change the skin color, race, or any facial features of the main subject in Image. 
  2.  **FORBIDDEN** to blend any characteristics of the main subject in Image with people in the reference images. 
  3.  **ABSOLUTELY IGNORE the identity of people in reference images: People in reference images are merely carriers/displays for the materials (hangers, models). 
  It is "STRICTLY FORBIDDEN" to copy, imitate, or reference any of their biological characteristics, including but not limited to: face, facial features, skin color, race, age, gender. 
  Your sole objective is to transplant the "material itself" onto the main subject in Image .`;

/**
 * Removes the background from an image using the Gemini API, making it transparent.
 * @param {string} base64Image - The base64 encoded image data URL.
 * @param {string} mimeType - The MIME type of the image.
 * @returns {Promise<string>} A promise that resolves to the data URL of the image with a transparent background.
 */
export async function removeImageBackground(base64Image, mimeType) {
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash-image-preview',
      contents: {
        parts: [
          { inlineData: { data: base64Image.split(',')[1], mimeType: mimeType } },
          { text: PROMPT },
        ],
      },
      config: {
        responseModalities: [Modality.IMAGE, Modality.TEXT],
        // Note: The 'gemini-2.5-flash-image-preview' model for image editing does not support
        // temperature, topP, or topK settings. These are used to control creativity in
        // text generation models. Including them here may cause errors or have no effect.
        // temperature: 0,
        // topP: 0.95,
        // topK: 1,
      },
    });

    // Find the first image part in the response
    for (const part of response.candidates[0].content.parts) {
      if (part.inlineData) {
        return `data:image/png;base64,${part.inlineData.data}`;
      }
    }
    // If no image is returned, throw an error
    throw new Error("The AI did not return an image. It may have returned text instead: " + response.text);
  } catch (error) {
    console.error("Error removing background:", error);
    throw new Error(`Failed to remove background: ${error.message}`);
  }
}