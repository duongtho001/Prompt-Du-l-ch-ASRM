export interface CharacterReference {
  id: string;
  name: string;
  prompt: string;
  imageUrl: string | null;
}

export interface CharacterVariation {
  title: string;
  description: string;
}

export interface ScenePrompt {
  scene_description: string;
  character_description: string; // Includes actions and expressions
  background_description: string;
  camera_shot: string;
  lighting: string;
  color_palette: string;
  style: string;
  composition_notes: string;
  sound_effects: string;
  dialogue: string;
  keywords: string[];
  negative_prompts: string[];
  aspect_ratio: string;
  duration_seconds: number;
}


export interface Scene {
  scene_id: number;
  time: string;
  prompt: ScenePrompt;
  imageUrl?: string;
  isGeneratingImage?: boolean;
}

export interface VideoConfig {
  duration: number;
  style: string;
  includeDialogue: boolean;
  dialogueLanguage: string;
  format: 'action_scene' | 'narrative_short' | 'emotional_moment';
}

export interface Project {
  id: string;
  name: string;
  characterReferences: CharacterReference[];
  storyIdea: string;
  generatedScript: string;
  videoConfig: VideoConfig;
  scenes: Scene[];
  apiProvider: string;
  selectedModel: string;
  lastModified: number;
}