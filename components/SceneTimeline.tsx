import React, { useState, useEffect } from 'react';
import type { Scene, ReferenceImage, ScenePrompt } from '../types';
import SceneCard from './SceneCard';
import Loader from './Loader';
import type { TranslationKeys } from '../translations';
import DownloadIcon from './icons/DownloadIcon';
import PhotoGroupIcon from './icons/PhotoGroupIcon';

interface SceneTimelineProps {
  scenes: Scene[];
  onUpdatePrompt: (sceneId: number, newPrompt: ScenePrompt) => void;
  isLoading: boolean;
  isBatchGenerating: boolean;
  isGenerationComplete: boolean;
  generationProgress: { current: number; total: number };
  generationStatusMessage: string;
  onDownloadPrompts: () => void;
  onDownloadAllImages: () => void;
  referenceImages: ReferenceImage[];
  onGenerateSceneImage: (sceneId: number, referenceImageId: string) => void;
  onGenerateAllSceneImages: (referenceImageId: string) => void;
  t: TranslationKeys;
}

const SceneTimeline: React.FC<SceneTimelineProps> = ({ 
  scenes, 
  onUpdatePrompt, 
  isLoading, 
  isBatchGenerating,
  onDownloadPrompts,
  onDownloadAllImages,
  referenceImages,
  onGenerateSceneImage, 
  onGenerateAllSceneImages,
  t,
  isGenerationComplete,
  generationProgress,
  generationStatusMessage
}) => {
  const [primaryReferenceId, setPrimaryReferenceId] = useState<string>('');
  const sortedScenes = [...scenes].sort((a, b) => a.scene_id - b.scene_id);
  
  const hasGeneratedImages = scenes.some(s => s.imageUrl);

  useEffect(() => {
    // Auto-select the first image if available
    if (!primaryReferenceId && referenceImages.length > 0) {
      setPrimaryReferenceId(referenceImages[0].id);
    }
    // If the selected image is removed, reset
    if (primaryReferenceId && !referenceImages.some(c => c.id === primaryReferenceId)) {
      setPrimaryReferenceId(referenceImages.length > 0 ? referenceImages[0].id : '');
    }
  }, [referenceImages, primaryReferenceId]);

  return (
    <div className="p-6 bg-[#0D0D0F] rounded-lg min-h-full">
      <div className="flex flex-col gap-4 mb-6 border-b-2 border-[#1E1E22] pb-4">
        <div className="flex justify-between items-center">
            <h2 className="text-2xl font-bold text-gray-100">
            {t.timelineTitle}
            </h2>
            <button 
                onClick={onDownloadPrompts}
                disabled={isLoading || isBatchGenerating || scenes.length === 0}
                className="flex items-center gap-x-2 bg-[#1E1E22] text-[#5BEAFF] font-semibold py-2 px-4 rounded-lg hover:bg-cyan-900/50 transition-colors disabled:bg-gray-700 disabled:text-gray-500 disabled:cursor-not-allowed"
            >
                <DownloadIcon className="w-5 h-5" />
                {t.downloadButton}
            </button>
        </div>

        {/* Batch Image Tools */}
        {scenes.length > 0 && (
            <div className="bg-[#1E1E22] p-3 rounded-lg border border-gray-700 flex flex-wrap items-end gap-4">
                <div className="flex-grow">
                    <label htmlFor="primary-character" className="block text-xs font-medium text-gray-400 mb-1">{t.primaryReferenceLabel}</label>
                    <select
                        id="primary-character"
                        value={primaryReferenceId}
                        onChange={(e) => setPrimaryReferenceId(e.target.value)}
                        disabled={isLoading || isBatchGenerating || referenceImages.length === 0}
                        className="w-full bg-[#0D0D0F] text-gray-300 p-2 rounded-md border border-gray-600 focus:ring-1 focus:ring-[#5BEAFF] focus:border-[#5BEAFF] transition text-sm disabled:bg-gray-800 disabled:text-gray-500"
                    >
                        <option value="">{t.selectPrimaryReferencePrompt}</option>
                        {referenceImages.map(img => <option key={img.id} value={img.id}>{img.name}</option>)}
                    </select>
                </div>
                <button
                    onClick={() => onGenerateAllSceneImages(primaryReferenceId)}
                    disabled={!primaryReferenceId || isLoading || isBatchGenerating}
                    className="flex items-center gap-x-2 bg-cyan-800/50 text-cyan-200 font-semibold py-2 px-3 rounded-lg hover:bg-cyan-700 hover:text-white transition-colors border border-cyan-700 disabled:bg-gray-700 disabled:text-gray-500 disabled:cursor-not-allowed"
                >
                    <PhotoGroupIcon className="w-5 h-5" />
                    {isBatchGenerating ? t.generatingAllImagesButton : t.generateAllImagesButton}
                </button>
                 <button 
                    onClick={onDownloadAllImages}
                    disabled={isLoading || isBatchGenerating || !hasGeneratedImages}
                    className="flex items-center gap-x-2 bg-[#1E1E22] text-gray-300 font-semibold py-2 px-3 rounded-lg hover:bg-gray-700 hover:text-white transition-colors border border-gray-600 disabled:bg-gray-700 disabled:text-gray-500 disabled:cursor-not-allowed"
                >
                    <DownloadIcon className="w-5 h-5" />
                    {t.downloadAllImagesButton}
                </button>
            </div>
        )}
      </div>
      
      {scenes.length === 0 && !isLoading ? (
        <div className="flex flex-col items-center justify-center h-96 bg-[#1E1E22] rounded-lg text-center p-4">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 text-gray-500 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
          <h3 className="text-xl font-semibold text-gray-300">{t.emptyTimelineTitle}</h3>
          <p className="text-gray-500 mt-2">{t.emptyTimelineDescription}</p>
        </div>
      ) : (
        <div className="space-y-6">
          {sortedScenes.map((scene) => (
            <SceneCard 
              key={scene.scene_id} 
              scene={scene} 
              onUpdatePrompt={onUpdatePrompt} 
              referenceImages={referenceImages}
              onGenerateSceneImage={onGenerateSceneImage}
              isLoading={isLoading}
              isBatchGenerating={isBatchGenerating}
              t={t} 
            />
          ))}
        </div>
      )}

      {/* Centralized loader/status display at the bottom */}
      {(isLoading && !isBatchGenerating) && (
        <div className={`flex justify-center items-center ${scenes.length === 0 ? 'h-96' : 'mt-8'}`}>
          <Loader t={t} progress={generationProgress} statusMessage={generationStatusMessage} />
        </div>
      )}
      {isGenerationComplete && !isLoading && scenes.length > 0 && (
        <div className="flex justify-center items-center mt-8">
          <Loader t={t} isComplete={true} />
        </div>
      )}
    </div>
  );
};

export default SceneTimeline;