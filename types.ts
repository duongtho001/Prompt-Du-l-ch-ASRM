import { ReactNode } from "react";

export interface ReferenceImage {
  id: string;
  name: string;
  imageUrl: string;
}

export interface ScenePrompt {
  description: string;
  style: string;
  camera: string;
  lighting: string;
  environment: string;
  elements: string[];
  motion: string;
  dialogue: string; // The dialogue or voiceover text for the scene
  audio: string;
  ending: string;
  text: string;
  keywords: string[];
  aspect_ratio: string;
  duration_seconds: number;
  fps: number;
  quality: string;
  negative_prompts: string[];
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
  format: 'trailer' | 'short' | 'longform';
}

export interface Project {
  id: string;
  name: string;
  referenceImages: ReferenceImage[];
  storyIdea: string;
  generatedScript: string;
  videoConfig: VideoConfig;
  scenes: Scene[];
  lastModified: number;
}