import { GoogleGenAI, Modality, GenerateContentResponse, Type } from "@google/genai";
import type { VideoConfig, Scene, ScenePrompt, CharacterVariation } from '../types';
import { Language, translations } from "../translations";
import * as apiKeyManager from './apiKeyManager';

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

const INTERNAL_MODEL = 'gemini-2.5-flash';

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
      const errorMessage = (error instanceof Error ? error.message : String(error));
      
      let delay = 0;
      
      // Do not retry on quota errors here; key rotation will handle it.
      const isQuotaError = errorMessage.includes('429') || errorMessage.includes('RESOURCE_EXHAUSTED');
      const isServerError = errorMessage.includes('503') || errorMessage.includes('overloaded') || errorMessage.includes('unavailable');

      if (isQuotaError) {
          throw error; // Pass quota errors up to the key rotation handler.
      } else if (isServerError) {
          delay = initialDelay * (2 ** i);
          console.warn(`Attempt ${i + 1}/${retries} failed in ${context} with a server error. Retrying in ${delay}ms...`);
      } else {
          // For other errors like invalid key, fail immediately.
          throw error;
      }

      if (i === retries - 1) {
          break;
      }

      await sleep(delay + Math.random() * 500);
    }
  }

  console.error(`All server retries failed in ${context}.`);
  throw lastError;
}

function getErrorMessage(error: unknown, context: string, language: Language): string {
    const t = translations[language];
    console.error(`Error in ${context}:`, error);
    if (error instanceof Error) {
        const message = error.message.toLowerCase();
        if (message.includes('all api keys have hit their quota')) {
            return t.errorAllKeysExhausted;
        }
        if (message.includes('quota') || message.includes('resource_exhausted')) {
            return t.errorQuotaExceeded;
        }
        if (message.includes('api key not valid') || message.includes('api key is invalid')) {
            return t.errorInvalidApiKey(context);
        }
        if (message.includes('overloaded') || message.includes('503') || message.includes('unavailable')) {
            return t.errorServerOverloaded(context);
        }
        if (message.includes('api key is missing')) {
            return t.errorMissingApiKey('Google');
        }
        return t.errorGeneric(context, error.message);
    }
    return t.errorUnknown(context);
}

async function executeWithKeyRotation<T>(
    apiCallFn: (apiKey: string) => Promise<T>,
    language: Language
): Promise<T> {
    const totalKeys = apiKeyManager.getKeyCount();
    if (totalKeys === 0) {
        throw new Error('API key is missing.');
    }

    const startIndex = apiKeyManager.getCurrentIndex();
    let lastError: unknown;

    for (let i = 0; i < totalKeys; i++) {
        const currentKey = apiKeyManager.getCurrentKey();
        
        if (!currentKey) {
            apiKeyManager.moveToNextKey();
            continue;
        }

        try {
            return await apiCallFn(currentKey);
        } catch (error) {
            lastError = error;
            const errorMessage = (error instanceof Error ? error.message : String(error)).toLowerCase();

            if (errorMessage.includes('quota') || errorMessage.includes('resource_exhausted')) {
                console.warn(`API Key ending in ...${currentKey.slice(-4)} hit quota. Switching to the next key.`);
                apiKeyManager.moveToNextKey();
                if (apiKeyManager.getCurrentIndex() === startIndex) {
                    console.error("Cycled through all keys; all are exhausted.");
                    break; 
                }
            } else {
                throw error;
            }
        }
    }
    
    throw new Error("All API keys have hit their quota.");
}

const getAiClient = (apiKey: string) => new GoogleGenAI({ apiKey });

export const generateCharacterPromptFromImage = async (
    imageBase64: string,
    language: Language,
): Promise<string> => {
    const systemInstruction = translations[language].systemInstruction_generateCharacterPrompt;
    const userPromptText = "Please describe the character in this image in detail for an animation project.";

    const match = imageBase64.match(/^data:(image\/.+);base64,(.+)$/);
    if (!match) throw new Error("Invalid image format");
    const mimeType = match[1];
    const data = match[2];

    const apiCall = (apiKey: string) => {
        const fn = async () => {
            const ai = getAiClient(apiKey);
            const response = await ai.models.generateContent({
                model: INTERNAL_MODEL,
                config: { systemInstruction },
                contents: {
                    parts: [
                        { inlineData: { mimeType, data } },
                        { text: userPromptText }
                    ]
                }
            });
            return response.text || "";
        };
        return withRetry(fn, 3, 1000, 'generateCharacterPromptFromImage');
    };

    try {
        return await executeWithKeyRotation(apiCall, language);
    } catch (error) {
        throw new Error(getErrorMessage(error, 'generateCharacterPromptFromImage', language));
    }
};

export const generateCharacterPromptVariations = async (
    characterName: string,
    animationStyle: string,
    storyStyle: string,
    language: Language,
): Promise<CharacterVariation[]> => {
    const systemInstruction = translations[language].systemInstruction_generateCharacterVariations(characterName, animationStyle, storyStyle);

    const apiCall = (apiKey: string) => {
        const fn = async () => {
            const ai = getAiClient(apiKey);
            const response = await ai.models.generateContent({
                model: INTERNAL_MODEL,
                config: {
                    systemInstruction,
                    responseMimeType: 'application/json',
                    responseSchema: {
                        type: Type.OBJECT,
                        properties: {
                            variations: {
                                type: Type.ARRAY,
                                items: {
                                    type: Type.OBJECT,
                                    properties: {
                                        title: { type: Type.STRING },
                                        description: { type: Type.STRING }
                                    },
                                    required: ["title", "description"]
                                }
                            }
                        },
                        required: ["variations"]
                    }
                },
                contents: { text: "Generate variations." }
            });
            
            const parsed = JSON.parse(response.text || "{}");
            if (parsed.variations && Array.isArray(parsed.variations)) {
                return parsed.variations;
            }
            throw new Error("Invalid JSON structure for character variations.");
        };
        return withRetry(fn, 3, 1000, 'generateCharacterPromptVariations');
    };

    try {
        return await executeWithKeyRotation(apiCall, language);
    } catch (error) {
        throw new Error(getErrorMessage(error, 'generateCharacterPromptVariations', language));
    }
};

export const generateStoryIdea = async (
  animationStyle: string,
  storyStyle: string,
  language: Language,
  characterDescriptions: string,
): Promise<string> => {
    const systemInstruction = translations[language].systemInstruction_generateStoryIdea(animationStyle, storyStyle, characterDescriptions);
    const userPrompt = "Please generate an animation story concept for the character(s) provided in the system instruction.";

    const apiCall = (apiKey: string) => {
        const fn = async () => {
            const ai = getAiClient(apiKey);
            const response = await ai.models.generateContent({
                model: INTERNAL_MODEL,
                config: { systemInstruction },
                contents: { text: userPrompt }
            });
            return response.text || "";
        };
        return withRetry(fn, 3, 1000, 'generateStoryIdea');
    };

  try {
     return await executeWithKeyRotation(apiCall, language);
  } catch (error) {
    throw new Error(getErrorMessage(error, 'generateStoryIdea', language));
  }
};

export const generateScript = async (
  storyIdea: string,
  config: VideoConfig,
  language: Language,
  characterDescriptions: string,
): Promise<string> => {
    const systemInstruction = translations[language].systemInstruction_generateScript(config, characterDescriptions);

    const userPrompt = `
        **Animation Story Idea:**
        ${storyIdea}

        **Animation Style:** ${config.style}
    `;

    const apiCall = (apiKey: string) => {
        const fn = async () => {
            const ai = getAiClient(apiKey);
            const response = await ai.models.generateContent({
                model: INTERNAL_MODEL,
                config: { systemInstruction },
                contents: { text: userPrompt }
            });
            return response.text || "";
        };
        return withRetry(fn, 3, 1000, 'generateScript');
    };

  try {
    return await executeWithKeyRotation(apiCall, language);
  } catch (error) {
    throw new Error(getErrorMessage(error, 'generateScript', language));
  }
};

export const generateScenePrompts = async (
  generatedScript: string,
  config: VideoConfig,
  language: Language,
  characterDescriptions: string,
  existingScenesCount: number,
  scenesPerBatch: number,
  lastScene: Scene | null,
): Promise<Scene[]> => {
    const startSceneId = existingScenesCount + 1;
    const systemInstruction = translations[language].systemInstruction_generateScenes(config, characterDescriptions, startSceneId, existingScenesCount, scenesPerBatch, lastScene);
    
    const userPrompt = `
        **Full Animation Script to be Visualized:**
        ${generatedScript}

        **Animation Configuration:**
        - Total Duration: ${config.duration} seconds
        - Format: ${config.format}
        - Scenes to generate in this batch: ${scenesPerBatch}
    `;

    const apiCall = (apiKey: string) => {
        const fn = async () => {
            const ai = getAiClient(apiKey);
            const response = await ai.models.generateContent({
                model: INTERNAL_MODEL,
                config: {
                    systemInstruction,
                    responseMimeType: 'application/json',
                    responseSchema: {
                        type: Type.OBJECT,
                        properties: {
                            scenes: {
                                type: Type.ARRAY,
                                items: {
                                    type: Type.OBJECT,
                                    properties: {
                                        scene_id: { type: Type.INTEGER },
                                        time: { type: Type.STRING },
                                        prompt: {
                                            type: Type.OBJECT,
                                            properties: {
                                                scene_description: { type: Type.STRING },
                                                character_description: { type: Type.STRING },
                                                background_description: { type: Type.STRING },
                                                camera_shot: { type: Type.STRING },
                                                lighting: { type: Type.STRING },
                                                color_palette: { type: Type.STRING },
                                                style: { type: Type.STRING },
                                                composition_notes: { type: Type.STRING },
                                                sound_effects: { type: Type.STRING },
                                                dialogue: { type: Type.STRING },
                                                keywords: { type: Type.ARRAY, items: { type: Type.STRING } },
                                                negative_prompts: { type: Type.ARRAY, items: { type: Type.STRING } },
                                                aspect_ratio: { type: Type.STRING },
                                                duration_seconds: { type: Type.NUMBER },
                                            },
                                            required: ["scene_description", "character_description", "background_description", "style"]
                                        }
                                    },
                                    required: ["scene_id", "time", "prompt"]
                                }
                            }
                        },
                        required: ["scenes"]
                    }
                },
                contents: { text: userPrompt }
            });

            const parsedJson = JSON.parse(response.text || "{}");

            if (parsedJson.scenes && Array.isArray(parsedJson.scenes)) {
                return parsedJson.scenes as Scene[];
            } else {
                console.warn("Received unexpected JSON structure. 'scenes' array not found.", parsedJson);
                return [];
            }
        };
        return withRetry(fn, 3, 1500, 'generateScenePrompts');
    };

    try {
        return await executeWithKeyRotation(apiCall, language);
    } catch (error) {
        throw new Error(getErrorMessage(error, 'generateScenePrompts', language));
    }
};
  
export const generateSceneImage = async (scenePrompt: ScenePrompt, referenceImageBase64: string, language: Language): Promise<string> => {
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
          text: `Using the provided reference image for character and style consistency, create a single animation frame based on the following detailed JSON prompt: ${JSON.stringify(scenePrompt, null, 2)}`
      };
  
      const apiCall = (apiKey: string) => {
          const fn = async () => {
              const ai = getAiClient(apiKey);
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
          return withRetry(fn, 3, 1500, 'generateSceneImage');
      };
      
      try {
        return await executeWithKeyRotation(apiCall, language);
      } catch (error) {
        throw new Error(getErrorMessage(error, 'generateSceneImage', language));
      }
};