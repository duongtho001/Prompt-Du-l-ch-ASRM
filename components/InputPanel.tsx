import React, { useState, useEffect, useRef } from 'react';
import type { CharacterReference, VideoConfig, Scene } from '../types';
import { VIDEO_STYLES, VIDEO_FORMATS, STORY_STYLES, DIALOGUE_LANGUAGES } from '../constants';
import SparklesIcon from './icons/SparklesIcon';
import LightBulbIcon from './icons/LightBulbIcon';
import type { TranslationKeys, Language } from '../translations';
import PhotoGroupIcon from './icons/PhotoGroupIcon';
import PlusIcon from './icons/PlusIcon';
import TrashIcon from './icons/TrashIcon';
import WandIcon from './icons/WandIcon';

const SCENE_DURATION_SECONDS = 8;

interface InputPanelProps {
  scenes: Scene[];
  characterReferences: CharacterReference[];
  onUpdateCharacter: (id: string, field: keyof CharacterReference, value: string | null) => void;
  onAddCharacter: () => void;
  onDeleteCharacter: (id: string) => void;
  onGenerateCharacterPrompt: (characterId: string, file: File) => Promise<void>;
  onSuggestCharacterPrompt: (characterId: string) => Promise<void>;
  onImportCharacters: (file: File) => void;
  isGeneratingPromptId: string | null;
  isSuggestingPromptId: string | null;
  storyIdea: string;
  setStoryIdea: React.Dispatch<React.SetStateAction<string>>;
  storyStyle: string;
  setStoryStyle: React.Dispatch<React.SetStateAction<string>>;
  generatedScript: string;
  videoConfig: VideoConfig;
  setVideoConfig: React.Dispatch<React.SetStateAction<VideoConfig>>;
  onPrimaryAction: () => void;
  isLoading: boolean;
  onGenerateStoryIdea: () => void;
  cooldowns: { [key: string]: boolean };
  t: TranslationKeys;
  language: Language;
}

const CharacterCard: React.FC<{
    character: CharacterReference;
    onUpdate: (id: string, field: keyof CharacterReference, value: string | null) => void;
    onDelete: (id: string) => void;
    onGeneratePrompt: (id: string, file: File) => void;
    onSuggestPrompt: (id: string) => void;
    isGeneratingFromImage: boolean;
    isSuggesting: boolean;
    t: TranslationKeys;
}> = ({ character, onUpdate, onDelete, onGeneratePrompt, onSuggestPrompt, isGeneratingFromImage, isSuggesting, t }) => {
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleTriggerFileInput = () => fileInputRef.current?.click();

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            onGeneratePrompt(character.id, file);
        }
        if (event.target) {
            event.target.value = '';
        }
    };

    return (
        <div className="bg-[#0D0D0F] p-4 rounded-lg border border-gray-700 space-y-4">
             <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                className="hidden"
                accept="image/png, image/jpeg, image/webp"
            />
            <div className="flex justify-between items-center">
                <input
                    type="text"
                    placeholder={t.characterNamePlaceholder}
                    value={character.name}
                    onChange={(e) => onUpdate(character.id, 'name', e.target.value)}
                    className="w-full bg-transparent text-lg font-semibold text-gray-200 placeholder-gray-500 border-b-2 border-gray-700 focus:border-[#5BEAFF] focus:outline-none transition-colors"
                />
                <button onClick={() => onDelete(character.id)} className="ml-4 text-gray-500 hover:text-red-400 transition-colors">
                    <TrashIcon className="w-5 h-5" />
                </button>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                    <div 
                        onClick={handleTriggerFileInput}
                        className="relative aspect-square w-full bg-gray-900/50 border-2 border-dashed border-gray-600 rounded-lg flex items-center justify-center overflow-hidden cursor-pointer hover:border-[#5BEAFF] transition-colors group"
                    >
                        {character.imageUrl ? (
                            <img src={character.imageUrl} alt={character.name} className="w-full h-full object-cover" />
                        ) : (
                            <div className="text-center text-gray-500 p-2">
                                <PhotoGroupIcon className="w-8 h-8 mx-auto mb-1" />
                                <p className="text-xs font-semibold">{t.uploadImagePrompt}</p>
                            </div>
                        )}
                    </div>
                     <div className="grid grid-cols-2 gap-2">
                        <button
                            onClick={handleTriggerFileInput}
                            disabled={isGeneratingFromImage || isSuggesting}
                            className="w-full flex items-center justify-center gap-x-1.5 text-xs font-semibold text-center text-cyan-300 hover:text-cyan-100 py-1.5 border-2 border-gray-600 hover:border-[#5BEAFF] rounded-lg transition-colors disabled:text-gray-500 disabled:cursor-not-allowed"
                        >
                            <WandIcon className="w-4 h-4" />
                            {isGeneratingFromImage ? t.generatingPromptButton : t.generatePromptFromImageButton}
                        </button>
                         <button
                            onClick={() => onSuggestPrompt(character.id)}
                            disabled={isGeneratingFromImage || isSuggesting || !character.name.trim()}
                            className="w-full flex items-center justify-center gap-x-1.5 text-xs font-semibold text-center text-cyan-300 hover:text-cyan-100 py-1.5 border-2 border-gray-600 hover:border-[#5BEAFF] rounded-lg transition-colors disabled:text-gray-500 disabled:cursor-not-allowed"
                        >
                            <LightBulbIcon className="w-4 h-4" />
                             {isSuggesting ? t.suggestingPromptButton : t.suggestPromptButton}
                        </button>
                    </div>
                </div>
                <textarea
                    rows={8}
                    className="w-full h-full bg-gray-900/50 text-gray-300 p-2 rounded-md border border-gray-600 focus:ring-2 focus:ring-[#5BEAFF] focus:border-[#5BEAFF] transition text-sm"
                    placeholder={t.characterPromptPlaceholder}
                    value={character.prompt}
                    onChange={(e) => onUpdate(character.id, 'prompt', e.target.value)}
                />
            </div>
        </div>
    );
};


// FIX: Changed component to a named export to resolve module loading issue.
export const InputPanel: React.FC<InputPanelProps> = ({
  scenes,
  characterReferences,
  onUpdateCharacter,
  onAddCharacter,
  onDeleteCharacter,
  onGenerateCharacterPrompt,
  onSuggestCharacterPrompt,
  onImportCharacters,
  isGeneratingPromptId,
  isSuggestingPromptId,
  storyIdea,
  setStoryIdea,
  storyStyle,
  setStoryStyle,
  generatedScript,
  videoConfig,
  setVideoConfig,
  onPrimaryAction,
  isLoading,
  onGenerateStoryIdea,
  cooldowns,
  t,
  language,
}) => {
  const [minutesDisplay, setMinutesDisplay] = useState<string>('');
  const importFileRef = useRef<HTMLInputElement>(null);
  
  useEffect(() => {
    if (videoConfig.duration > 0) {
      const minutes = Math.floor(videoConfig.duration / 60);
      setMinutesDisplay(minutes.toString());
    } else {
      setMinutesDisplay('');
    }
  }, [videoConfig.duration]);


  const handleConfigChange = <K extends keyof VideoConfig>(key: K, value: VideoConfig[K]) => {
    setVideoConfig((prev) => ({ ...prev, [key]: value }));
  };
  
  const handleMinutesInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setMinutesDisplay(value);

    const minutes = parseInt(value, 10);
    if (!isNaN(minutes) && minutes > 0) {
      const requestedSeconds = minutes * 60;
      const numberOfScenes = Math.round(requestedSeconds / SCENE_DURATION_SECONDS);
      const actualSeconds = Math.max(SCENE_DURATION_SECONDS, numberOfScenes * SCENE_DURATION_SECONDS);
      handleConfigChange('duration', actualSeconds);
    } else {
      handleConfigChange('duration', 0); // Invalid input disables generation
    }
  };
  
  const handleImportFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      onImportCharacters(file);
    }
    if (event.target) {
        event.target.value = '';
    }
  };

  const hasAtLeastOneCharacter = characterReferences.length > 0 && characterReferences.some(c => c.prompt.trim() !== '');
  
  const totalScenes = videoConfig.duration > 0 ? Math.round(videoConfig.duration / SCENE_DURATION_SECONDS) : 0;
  const numberOfScenes = totalScenes;
  const displayMinutes = Math.floor(videoConfig.duration / 60);
  const displaySeconds = videoConfig.duration % 60;

  const canGenerateScript = !isLoading && storyIdea.trim() && hasAtLeastOneCharacter && videoConfig.duration > 0;
  const storyboardStarted = scenes.length > 0;
  const storyboardComplete = storyboardStarted && scenes.length >= totalScenes;
  
  const canGenerateStoryboard = canGenerateScript && generatedScript.trim() !== '' && !storyboardComplete;
  
  const isGeneratingScript = isLoading && !generatedScript;
  const isGeneratingStoryboard = isLoading && !!generatedScript;

  const getButtonText = () => {
    if (cooldowns['mainAction']) return t.onCooldownButton;
    if (isGeneratingScript) return t.generatingScriptButton;
    if (isGeneratingStoryboard) return t.generatingStoryboardButton;
    if (storyboardStarted && !storyboardComplete) return t.continueGenerateStoryboardButton;
    if (generatedScript) return t.generateStoryboardButton;
    return t.generateScriptButton;
  };

  return (
    <div className="p-6 bg-[#1E1E22] rounded-lg shadow-lg space-y-8">
       <input
        type="file"
        ref={importFileRef}
        onChange={handleImportFileChange}
        className="hidden"
        accept=".txt"
      />
      <div>
        <h3 className="block text-lg font-semibold text-gray-200 mb-4">
            {t.characterRosterLabel}
        </h3>
        <div className="space-y-4">
            {characterReferences.map(char => (
                <CharacterCard
                    key={char.id}
                    character={char}
                    onUpdate={onUpdateCharacter}
                    onDelete={onDeleteCharacter}
                    onGeneratePrompt={onGenerateCharacterPrompt}
                    onSuggestPrompt={onSuggestCharacterPrompt}
                    isGeneratingFromImage={isGeneratingPromptId === char.id}
                    isSuggesting={isSuggestingPromptId === char.id}
                    t={t}
                />
            ))}
            <div className="flex gap-4">
                 <button
                    onClick={onAddCharacter}
                    className="w-full flex items-center justify-center gap-x-2 text-sm font-semibold text-center text-gray-300 hover:text-white py-2 border-2 border-dashed border-gray-600 hover:border-gray-500 rounded-lg transition-colors"
                >
                    <PlusIcon className="w-5 h-5" />
                    {t.addCharacterButton}
                </button>
                 <button
                    onClick={() => importFileRef.current?.click()}
                    className="w-full flex items-center justify-center gap-x-2 text-sm font-semibold text-center text-[#5BEAFF] hover:text-cyan-200 py-2 border-2 border-dashed border-cyan-800 hover:border-cyan-600 rounded-lg transition-colors"
                >
                    <SparklesIcon className="w-5 h-5" />
                    {t.importFromFileButton}
                </button>
            </div>
        </div>
      </div>


      <div>
        <label htmlFor="storyIdea" className="block text-lg font-semibold text-gray-200 mb-2">
            {t.storyIdeaLabel}
        </label>
        <div className="flex items-end gap-4 mb-4">
            <div className="flex-grow">
                <label htmlFor="storyStyle" className="block text-sm font-medium text-gray-400 mb-2">{t.storyStyleLabel}</label>
                <select
                id="storyStyle"
                className="w-full bg-[#0D0D0F] text-gray-300 p-3 rounded-md border border-gray-700 focus:ring-2 focus:ring-[#5BEAFF] focus:border-[#5BEAFF] transition"
                value={storyStyle}
                onChange={(e) => setStoryStyle(e.target.value)}
                >
                {STORY_STYLES.map(style => <option key={style.key} value={style.key}>{style[language]}</option>)}
                </select>
            </div>
            <button
                onClick={onGenerateStoryIdea}
                disabled={isLoading || cooldowns['storyIdea'] || !hasAtLeastOneCharacter}
                className="flex-shrink-0 h-[50px] flex items-center gap-x-1.5 text-sm font-semibold text-[#5BEAFF] hover:text-cyan-200 disabled:text-gray-500 disabled:cursor-not-allowed transition-colors bg-[#0D0D0F] border border-gray-700 rounded-md px-4 py-3 hover:border-[#5BEAFF]"
            >
                <LightBulbIcon className="w-4 h-4" />
                {cooldowns['storyIdea'] ? t.onCooldownButton : t.suggestIdeaButton}
            </button>
        </div>
        <textarea
          id="storyIdea"
          rows={8}
          className="w-full bg-[#0D0D0F] text-gray-300 p-3 rounded-md border border-gray-700 focus:ring-2 focus:ring-[#5BEAFF] focus:border-[#5BEAFF] transition"
          placeholder={t.storyIdeaPlaceholder}
          value={storyIdea}
          onChange={(e) => setStoryIdea(e.target.value)}
        />
      </div>

      {generatedScript && (
         <div>
          <label htmlFor="generatedScript" className="block text-lg font-semibold text-gray-200 mb-2">
            {t.generatedScriptLabel}
          </label>
          <textarea
            id="generatedScript"
            rows={10}
            readOnly
            className="w-full bg-[#0D0D0F]/50 text-gray-400 p-3 rounded-md border border-gray-700 cursor-default"
            value={generatedScript}
          />
        </div>
      )}
      
      <div>
        <h3 className="text-lg font-semibold text-gray-200 mb-4">{t.videoSettingsLabel}</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label htmlFor="duration" className="block text-sm font-medium text-gray-400 mb-2">{t.durationLabel}</label>
            <input
              id="duration"
              type="number"
              min="1"
              step="1"
              className="w-full bg-[#0D0D0F] text-gray-300 p-3 rounded-md border border-gray-700 focus:ring-2 focus:ring-[#5BEAFF] focus:border-[#5BEAFF] transition"
              value={minutesDisplay}
              onChange={handleMinutesInputChange}
              placeholder={t.durationPlaceholder}
            />
             {videoConfig.duration > 0 && (
              <p className="text-xs text-gray-500 mt-2">
                {t.durationFeedback(numberOfScenes, displayMinutes, displaySeconds)}
              </p>
            )}
          </div>
          <div>
            <label htmlFor="format" className="block text-sm font-medium text-gray-400 mb-2">{t.videoFormatLabel}</label>
            <select
              id="format"
              className="w-full bg-[#0D0D0F] text-gray-300 p-3 rounded-md border border-gray-700 focus:ring-2 focus:ring-[#5BEAFF] focus:border-[#5BEAFF] transition"
              value={videoConfig.format}
              onChange={(e) => handleConfigChange('format', e.target.value as VideoConfig['format'])}
            >
              {VIDEO_FORMATS.map(format => <option key={format.key} value={format.key}>{format[language]}</option>)}
            </select>
          </div>
          <div className="md:col-span-2">
            <label htmlFor="style" className="block text-sm font-medium text-gray-400 mb-2">{t.styleLabel}</label>
            <select
              id="style"
              className="w-full bg-[#0D0D0F] text-gray-300 p-3 rounded-md border border-gray-700 focus:ring-2 focus:ring-[#5BEAFF] focus:border-[#5BEAFF] transition"
              value={videoConfig.style}
              onChange={(e) => handleConfigChange('style', e.target.value)}
            >
              {VIDEO_STYLES.map(style => <option key={style.key} value={style.key}>{style[language]}</option>)}
            </select>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center md:col-span-2">
            <div className="flex items-center">
                <input
                    id="includeDialogue"
                    type="checkbox"
                    className="h-4 w-4 rounded border-gray-500 bg-gray-800 text-cyan-500 focus:ring-cyan-600"
                    checked={videoConfig.includeDialogue}
                    onChange={(e) => handleConfigChange('includeDialogue', e.target.checked)}
                />
                <label htmlFor="includeDialogue" className="ml-3 block text-sm font-medium text-gray-300">
                    {t.includeDialogueLabel}
                </label>
            </div>
            <div>
              <label htmlFor="dialogueLanguage" className="sr-only">{t.dialogueLanguageLabel}</label>
              <select
                id="dialogueLanguage"
                className="w-full bg-[#0D0D0F] text-gray-300 p-3 rounded-md border border-gray-700 focus:ring-2 focus:ring-[#5BEAFF] focus:border-[#5BEAFF] transition disabled:bg-gray-800/50 disabled:text-gray-500"
                value={videoConfig.dialogueLanguage}
                onChange={(e) => handleConfigChange('dialogueLanguage', e.target.value)}
                disabled={!videoConfig.includeDialogue}
              >
                {DIALOGUE_LANGUAGES.map(lang => <option key={lang.key} value={lang.key}>{lang[language]}</option>)}
              </select>
            </div>
          </div>
        </div>
      </div>

      <button
        onClick={onPrimaryAction}
        disabled={isLoading || !!isGeneratingPromptId || !!isSuggestingPromptId || cooldowns['mainAction'] || (generatedScript ? !canGenerateStoryboard : !canGenerateScript)}
        className="w-full flex items-center justify-center gap-x-2 bg-[#5BEAFF] text-[#0D0D0F] font-bold py-3 px-6 rounded-lg hover:bg-opacity-90 transition-all duration-300 disabled:bg-gray-500 disabled:cursor-not-allowed transform hover:scale-105"
      >
        <SparklesIcon className="w-6 h-6" />
        {getButtonText()}
      </button>
    </div>
  );
};