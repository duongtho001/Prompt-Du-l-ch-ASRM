import React, { useState, useEffect, useRef } from 'react';
import type { ReferenceImage, VideoConfig } from '../types';
import { VIDEO_STYLES, VIDEO_FORMATS } from '../constants';
import SparklesIcon from './icons/SparklesIcon';
import LightBulbIcon from './icons/LightBulbIcon';
import CameraIcon from './icons/CameraIcon';
import Loader from './Loader';
import type { TranslationKeys, Language } from '../translations';
import PhotoGroupIcon from './icons/PhotoGroupIcon';
import PlusIcon from './icons/PlusIcon';
import TrashIcon from './icons/TrashIcon';

interface InputPanelProps {
  referenceImages: ReferenceImage[];
  setReferenceImages: React.Dispatch<React.SetStateAction<ReferenceImage[]>>;
  storyIdea: string;
  setStoryIdea: React.Dispatch<React.SetStateAction<string>>;
  generatedScript: string;
  videoConfig: VideoConfig;
  setVideoConfig: React.Dispatch<React.SetStateAction<VideoConfig>>;
  onPrimaryAction: () => void;
  isLoading: boolean;
  onGenerateStoryIdea: () => void;
  isIdeaLoading: boolean;
  t: TranslationKeys;
  language: Language;
}


const InputPanel: React.FC<InputPanelProps> = ({
  referenceImages,
  setReferenceImages,
  storyIdea,
  setStoryIdea,
  generatedScript,
  videoConfig,
  setVideoConfig,
  onPrimaryAction,
  isLoading,
  onGenerateStoryIdea,
  isIdeaLoading,
  t,
  language,
}) => {
  const [minutesDisplay, setMinutesDisplay] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const sceneDuration = 8;

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
      const numberOfScenes = Math.round(requestedSeconds / sceneDuration);
      const actualSeconds = Math.max(sceneDuration, numberOfScenes * sceneDuration);
      handleConfigChange('duration', actualSeconds);
    } else {
      handleConfigChange('duration', 0); // Invalid input disables generation
    }
  };

  const handleAddReferenceImage = () => {
    fileInputRef.current?.click();
  };
  
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        const newImage: ReferenceImage = {
          id: crypto.randomUUID(),
          name: file.name.split('.').slice(0, -1).join('.') || t.newReferenceImageName,
          imageUrl: reader.result as string,
        };
        setReferenceImages(prev => [...prev, newImage]);
      };
      reader.readAsDataURL(file);
    }
    if (event.target) {
        event.target.value = '';
    }
  };

  const handleRemoveReferenceImage = (id: string) => {
    setReferenceImages(prev => prev.filter(img => img.id !== id));
  };
  
  const handleImageNameChange = (id: string, newName: string) => {
    setReferenceImages(prev => prev.map(img => img.id === id ? { ...img, name: newName } : img));
  };


  const numberOfScenes = videoConfig.duration > 0 ? Math.round(videoConfig.duration / sceneDuration) : 0;
  const displayMinutes = Math.floor(videoConfig.duration / 60);
  const displaySeconds = videoConfig.duration % 60;

  const canGenerateScript = !isLoading && !isIdeaLoading && storyIdea.trim() && videoConfig.duration > 0;
  const canGenerateStoryboard = canGenerateScript && generatedScript.trim() !== '';
  
  const isGeneratingScript = isLoading && !generatedScript;
  const isGeneratingStoryboard = isLoading && !!generatedScript;

  return (
    <div className="p-6 bg-[#1E1E22] rounded-lg shadow-lg space-y-8">
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        className="hidden"
        accept="image/png, image/jpeg, image/webp"
      />
      <div>
        <div className="flex justify-between items-center mb-4">
          <label className="block text-lg font-semibold text-gray-200">
            {t.referenceLocationsLabel}
          </label>
        </div>
        <div className="space-y-4">
            {referenceImages.length > 0 && (
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    {referenceImages.map((image) => (
                        <div key={image.id} className="group relative space-y-2">
                             <div className="relative aspect-video w-full bg-gray-900/50 border-2 border-dashed border-gray-700 rounded-lg flex items-center justify-center overflow-hidden">
                                <img src={image.imageUrl} alt={image.name} className="w-full h-full object-cover" />
                             </div>
                             <input 
                                type="text"
                                value={image.name}
                                onChange={(e) => handleImageNameChange(image.id, e.target.value)}
                                className="w-full bg-gray-800/50 text-gray-300 p-1 rounded-md border border-gray-600 focus:ring-1 focus:ring-[#5BEAFF] focus:border-[#5BEAFF] transition text-xs"
                             />
                             <button
                                onClick={() => handleRemoveReferenceImage(image.id)}
                                className="absolute top-2 right-2 p-1.5 bg-black/50 text-white/70 rounded-full opacity-0 group-hover:opacity-100 hover:bg-red-500/80 hover:text-white transition-all"
                                title={t.removeReferenceImageButton}
                             >
                                 <TrashIcon className="w-4 h-4" />
                             </button>
                        </div>
                    ))}
                </div>
            )}
           <button
            onClick={handleAddReferenceImage}
            className="w-full flex items-center justify-center gap-x-2 text-sm font-semibold text-center text-[#5BEAFF] hover:text-cyan-200 py-2 border-2 border-dashed border-gray-600 hover:border-[#5BEAFF] rounded-lg transition-colors"
          >
           <PlusIcon className="w-5 h-5" />
           {t.addReferenceImageButton}
          </button>
        </div>
      </div>

      <div>
        <div className="flex justify-between items-center mb-2">
          <label htmlFor="storyIdea" className="block text-lg font-semibold text-gray-200">
            {t.storyIdeaLabel}
          </label>
           <button
            onClick={onGenerateStoryIdea}
            disabled={isLoading || isIdeaLoading}
            className="flex items-center gap-x-1.5 text-sm font-semibold text-[#5BEAFF] hover:text-cyan-200 disabled:text-gray-500 disabled:cursor-not-allowed transition-colors"
          >
            <LightBulbIcon className="w-4 h-4" />
            {isIdeaLoading ? t.suggestingIdeaButton : t.suggestIdeaButton}
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
        </div>
         <div className="mt-6">
            <label htmlFor="style" className="block text-sm font-medium text-gray-400 mb-2">{t.styleLabel}</label>
            <select
              id="style"
              className="w-full bg-[#0D0D0F] text-gray-300 p-3 rounded-md border border-gray-700 focus:ring-2 focus:ring-[#5BEAFF] focus:border-[#5BEAFF] transition"
              value={videoConfig.style}
              onChange={(e) => handleConfigChange('style', e.target.value)}
            >
              {VIDEO_STYLES.map(style => <option key={style.key} value={style.en}>{style[language]}</option>)}
            </select>
          </div>
      </div>

      <button
        onClick={onPrimaryAction}
        disabled={isLoading || (generatedScript ? !canGenerateStoryboard : !canGenerateScript)}
        className="w-full flex items-center justify-center gap-x-2 bg-[#5BEAFF] text-[#0D0D0F] font-bold py-3 px-6 rounded-lg hover:bg-opacity-90 transition-all duration-300 disabled:bg-gray-500 disabled:cursor-not-allowed transform hover:scale-105"
      >
        <SparklesIcon className="w-6 h-6" />
        {
          isGeneratingScript ? t.generatingScriptButton : 
          isGeneratingStoryboard ? t.generatingStoryboardButton :
          generatedScript ? t.generateStoryboardButton : 
          t.generateScriptButton
        }
      </button>
    </div>
  );
};

export default InputPanel;