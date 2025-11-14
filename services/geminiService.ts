import { GoogleGenAI, Type, Modality, GenerateContentResponse } from "@google/genai";
import type { VideoConfig, Scene, ScenePrompt } from '../types';
import { Language, translations } from "../translations";

let apiKeys: string[] = [];
let currentKeyIndex = 0;

export function setApiKeys(keys: string[]) {
  apiKeys = keys;
  currentKeyIndex = 0;
}

function getClient(): GoogleGenAI {
  if (apiKeys.length === 0) {
    throw new Error(translations.en.apiKeyMissingError);
  }
  const key = apiKeys[currentKeyIndex];
  return new GoogleGenAI({ apiKey: key });
}

function rotateKey(): boolean {
  if (apiKeys.length > 1) {
    currentKeyIndex = (currentKeyIndex + 1) % apiKeys.length;
    console.log(`Rotated to API Key index ${currentKeyIndex}`);
    return true;
  }
  return false;
}

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

async function withRetry<T>(
  fn: () => Promise<T>,
  retries = 3,
  initialDelay = 1000,
  context: string
): Promise<T> {
  let lastError: unknown;

  for (let i = 0; i < retries; i++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      const errorMessage = (error instanceof Error ? error.message : String(error)).toLowerCase();

      // Check for retryable server error conditions
      if (errorMessage.includes('503') || errorMessage.includes('overloaded') || errorMessage.includes('unavailable')) {
        const delay = initialDelay * (2 ** i);
        console.warn(`Attempt ${i + 1}/${retries} failed in ${context} with a server error. Retrying in ${delay}ms...`);
        await sleep(delay + Math.random() * 500);
      } else {
        // Not a server error, throw to be handled by key rotation or final error handler
        throw error;
      }
    }
  }

  console.error(`All server retries failed in ${context}.`);
  throw lastError;
}

async function withKeyRotation<T>(
    apiCall: () => Promise<T>,
    context: string
): Promise<T> {
    if (apiKeys.length === 0) {
        throw new Error(translations.en.apiKeyMissingError);
    }

    const initialKeyIndex = currentKeyIndex;
    let attempts = 0;

    while(attempts < apiKeys.length) {
        try {
            return await apiCall();
        } catch (error) {
            const errorMessage = (error instanceof Error ? error.message : String(error)).toLowerCase();
            const isQuotaError = errorMessage.includes('quota') || errorMessage.includes('api key not valid');

            if (isQuotaError) {
                attempts++;
                const rotated = rotateKey();
                
                if (!rotated || currentKeyIndex === initialKeyIndex) {
                    throw new Error(translations.en.allApiKeysFailedError);
                }
                console.warn(`Key failed in ${context}. Retrying with next key (index: ${currentKeyIndex}).`);
            } else {
                throw error; // Not a quota error, rethrow it for the outer handler
            }
        }
    }
    throw new Error(translations.en.allApiKeysFailedError);
}


function getErrorMessage(error: unknown, context: string): string {
    console.error(`Error in ${context}:`, error);
    if (error instanceof Error) {
        const message = error.message;
        if (message === translations.en.apiKeyMissingError || message === translations.en.allApiKeysFailedError) {
            return message;
        }
        if (message.toLowerCase().includes('overloaded') || message.toLowerCase().includes('503') || message.toLowerCase().includes('unavailable')) {
            return `The model is currently busy or unavailable. The request was retried but failed. Please try again in a few moments. (Context: ${context})`;
        }
        return `Error in ${context}: ${message}`;
    }
    return `An unknown error occurred in ${context}.`;
}

const scenePromptSchema = {
  type: Type.OBJECT,
  properties: {
    description: { type: Type.STRING },
    style: { type: Type.STRING },
    camera: { type: Type.STRING },
    lighting: { type: Type.STRING },
    environment: { type: Type.STRING },
    elements: { type: Type.ARRAY, items: { type: Type.STRING } },
    motion: { type: Type.STRING },
    dialogue: { type: Type.STRING },
    audio: { type: Type.STRING },
    ending: { type: Type.STRING },
    text: { type: Type.STRING },
    keywords: { type: Type.ARRAY, items: { type: Type.STRING } },
    aspect_ratio: { type: Type.STRING },
    duration_seconds: { type: Type.INTEGER },
    fps: { type: Type.INTEGER },
    quality: { type: Type.STRING },
    negative_prompts: { type: Type.ARRAY, items: { type: Type.STRING } },
  },
  required: [
    "description", "style", "camera", "lighting", "environment", "elements",
    "motion", "dialogue", "audio", "ending", "text", "keywords", "aspect_ratio",
    "duration_seconds", "fps", "quality", "negative_prompts"
  ],
};

const sceneSchema = {
  type: Type.OBJECT,
  properties: {
    scene_id: { type: Type.INTEGER },
    time: { type: Type.STRING },
    prompt: scenePromptSchema,
  },
  required: ["scene_id", "time", "prompt"],
};

const fullResponseSchema = {
    type: Type.OBJECT,
    properties: {
        scenes: {
            type: Type.ARRAY,
            items: sceneSchema,
        },
    },
    required: ["scenes"],
};

export const generateStoryIdea = async (
  style: string,
  language: Language,
): Promise<string> => {
  const model = 'gemini-2.5-flash';
  
  const systemInstruction = translations[language].systemInstruction_generateStoryIdea(style);

  const apiCall = async () => {
    const ai = getClient();
    const response = await ai.models.generateContent({
      model,
      contents: "Please generate a travel video concept.",
      config: {
        systemInstruction,
        temperature: 0.9,
      },
    });
    return response.text.trim();
  };

  try {
     const performApiCall = () => withRetry(apiCall, 3, 1000, 'generateStoryIdea (server retry)');
     return await withKeyRotation(performApiCall, 'generateStoryIdea');
  } catch (error) {
    throw new Error(getErrorMessage(error, 'generateStoryIdea'));
  }
};


export const generateScript = async (
  storyIdea: string,
  config: VideoConfig,
  language: Language
): Promise<string> => {
  const model = 'gemini-2.5-pro';
  const systemInstruction = translations[language].systemInstruction_generateScript(config);

  const userPrompt = `
    **Travel Idea / Itinerary:**
    ${storyIdea}

    **Video Style:** ${config.style}
  `;

  const apiCall = async () => {
    const ai = getClient();
    const response = await ai.models.generateContent({
      model,
      contents: userPrompt,
      config: {
        systemInstruction,
        temperature: 0.9,
        topP: 0.95,
      },
    });
    
    return response.text.trim();
  };

  try {
    const performApiCall = () => withRetry(apiCall, 3, 1000, 'generateScript (server retry)');
    return await withKeyRotation(performApiCall, 'generateScript');
  } catch (error) {
    throw new Error(getErrorMessage(error, 'generateScript'));
  }
};


export const generateScenePrompts = async (
  generatedScript: string,
  config: VideoConfig,
  language: Language,
  existingScenes: Scene[] = []
): Promise<Scene[]> => {
  const model = 'gemini-2.5-flash';
  
  const isContinuation = existingScenes.length > 0;
  const systemInstruction = translations[language].systemInstruction_generateScenes(config, isContinuation);

  const lastSceneNumber = isContinuation ? Math.max(...existingScenes.map(s => s.scene_id)) : 0;
  
  const continuationPromptPart = isContinuation
    ? `
    You have already generated ${lastSceneNumber} scenes. Please continue generating the storyboard starting from scene number ${lastSceneNumber + 1}.

    **Previously Generated Scenes (for context only, do not repeat them):**
    ${JSON.stringify(existingScenes.slice(-3))} 
    `
    : 'Please generate the video scene prompts based on the following details.';

  const userPrompt = `
    ${continuationPromptPart}

    **Full Shot List to be Visualized:**
    ${generatedScript}

    **Video Configuration:**
    - Total Duration: ${config.duration} seconds
    - Format: ${config.format}
    `;

  const apiCall = async () => {
    const ai = getClient();
    const response = await ai.models.generateContent({
      model,
      contents: userPrompt,
      config: {
        systemInstruction,
        responseMimeType: "application/json",
        responseSchema: fullResponseSchema,
        temperature: 0.8,
        topP: 0.9,
      },
    });

    const rawText = response.text.trim();
    
    const jsonRegex = /```json\s*([\s\S]*?)\s*```|({[\s\S]*})/;
    const match = rawText.match(jsonRegex);
    
    if (!match) {
        throw new Error("Could not find a valid JSON object in the API response.");
    }
    
    const extractedJson = match[1] || match[2];
    
    let parsedJson;
    try {
        parsedJson = JSON.parse(extractedJson);
    } catch (e) {
        console.error("Failed to parse extracted JSON:", e, "Extracted:", extractedJson);
        throw new Error("Invalid JSON format received from API after cleanup.");
    }

    if (parsedJson.scenes && Array.isArray(parsedJson.scenes)) {
        return parsedJson.scenes as Scene[];
    } else {
        console.warn("Received unexpected JSON structure. 'scenes' array not found.", parsedJson);
        return [];
    }
  };

  try {
    const performApiCall = () => withRetry(apiCall, 3, 1500, 'generateScenePrompts (server retry)');
    return await withKeyRotation(performApiCall, 'generateScenePrompts');
  } catch (error) {
    throw new Error(getErrorMessage(error, 'generateScenePrompts'));
  }
};
  
  export const generateSceneImage = async (scenePrompt: ScenePrompt, referenceImageBase64: string): Promise<string> => {
      const model = 'gemini-2.5-flash-image';
      
      const match = referenceImageBase64.match(/^data:(image\/.+);base64,(.+)$/);
      if (!match) {
          throw new Error("Invalid base64 image format provided for reference.");
      }
      const mimeType = match[1];
      const data = match[2];
  
      const imagePart = {
          inlineData: {
              mimeType,
              data,
          },
      };
      const textPart = {
          text: `Using the provided reference image for location and style consistency, create a cinematic image based on the following detailed JSON prompt: ${JSON.stringify(scenePrompt, null, 2)}`
      };
  
      const apiCall = async () => {
          const ai = getClient();
          const response: GenerateContentResponse = await ai.models.generateContent({
              model,
              contents: { parts: [imagePart, textPart] },
              config: {
                  responseModalities: [Modality.IMAGE],
              },
          });
  
          for (const part of response.candidates[0].content.parts) {
              if (part.inlineData) {
                  const base64ImageBytes: string = part.inlineData.data;
                  return `data:image/png;base64,${base64ImageBytes}`;
              }
          }
          throw new Error("No image data found in the response from the model.");
      };
      
      try {
          const performApiCall = () => withRetry(apiCall, 3, 1500, 'generateSceneImage (server retry)');
          return await withKeyRotation(performApiCall, 'generateSceneImage');
      } catch (error) {
        throw new Error(getErrorMessage(error, 'generateSceneImage'));
      }
  };
