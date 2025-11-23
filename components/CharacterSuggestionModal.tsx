import React, { useState } from 'react';
import type { TranslationKeys } from '../translations';
import type { CharacterVariation } from '../types';
import SparklesIcon from './icons/SparklesIcon';
import ArrowPathIcon from './icons/ArrowPathIcon';
import CheckIcon from './icons/CheckIcon';

interface CharacterSuggestionModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAccept: (description: string) => void;
  onRegenerate: () => void;
  isLoading: boolean;
  variations: CharacterVariation[];
  error: string | null;
  cooldown: boolean;
  t: TranslationKeys;
}

const CharacterSuggestionModal: React.FC<CharacterSuggestionModalProps> = ({
  isOpen,
  onClose,
  onAccept,
  onRegenerate,
  isLoading,
  variations,
  error,
  cooldown,
  t,
}) => {
  const [selectedVariation, setSelectedVariation] = useState<CharacterVariation | null>(null);

  if (!isOpen) {
    return null;
  }

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-75 flex justify-center items-center z-50 p-4"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="char-suggestion-modal-title"
    >
      <div
        className="bg-[#1E1E22] rounded-lg shadow-2xl w-full max-w-4xl border-2 border-gray-700 flex flex-col max-h-[90vh]"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="px-6 py-4 flex justify-between items-center border-b border-gray-700">
          <h2 id="char-suggestion-modal-title" className="text-xl font-bold text-gray-100 flex items-center gap-x-3">
            <SparklesIcon className="w-6 h-6 text-[#5BEAFF]" />
            {t.characterSuggestionModalTitle}
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
            aria-label={t.closeButton}
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <div className="p-8 overflow-y-auto flex-grow">
          {isLoading && (
            <div className="flex flex-col items-center justify-center h-full">
              <div className="w-12 h-12 border-4 border-dashed rounded-full animate-spin border-[#5BEAFF]"></div>
              <p className="text-[#5BEAFF] text-lg font-medium mt-4">{t.characterSuggestionLoadingText}</p>
            </div>
          )}
          {error && !isLoading && (
            <div className="bg-red-900/50 border border-red-700 text-red-200 px-4 py-3 rounded-lg text-center">
              <p>{error}</p>
            </div>
          )}
          {!isLoading && !error && variations.length > 0 && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full">
              {variations.map((variation, index) => (
                <div
                  key={index}
                  onClick={() => setSelectedVariation(variation)}
                  className={`bg-[#0D0D0F] rounded-lg p-4 border-2 transition-all duration-200 cursor-pointer flex flex-col ${selectedVariation?.title === variation.title ? 'border-[#5BEAFF] ring-2 ring-[#5BEAFF]/50' : 'border-gray-700 hover:border-gray-500'}`}
                >
                  <h3 className="text-lg font-bold text-cyan-300 mb-2">{variation.title}</h3>
                  <p className="text-sm text-gray-400 whitespace-pre-wrap flex-grow">{variation.description}</p>
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="bg-gray-800/50 px-6 py-4 flex justify-between items-center border-t border-gray-700">
           <button
            type="button"
            className="flex items-center gap-x-2 rounded-md border border-gray-600 shadow-sm px-4 py-2 bg-[#0D0D0F] text-base font-medium text-gray-300 hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500 sm:w-auto sm:text-sm transition-colors disabled:text-gray-500 disabled:cursor-not-allowed"
            onClick={onRegenerate}
            disabled={isLoading || cooldown}
          >
            <ArrowPathIcon className={`w-5 h-5 ${isLoading ? 'animate-spin' : ''}`} />
            {cooldown ? t.onCooldownButton : t.regenerateIdeaButton}
          </button>
          <button
            type="button"
            className="flex items-center gap-x-2 rounded-md border border-transparent shadow-sm px-4 py-2 bg-[#5BEAFF] text-base font-bold text-black hover:bg-opacity-90 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 sm:w-auto sm:text-sm transition-colors disabled:bg-gray-500 disabled:cursor-not-allowed"
            onClick={() => selectedVariation && onAccept(selectedVariation.description)}
            disabled={isLoading || !selectedVariation}
          >
            <CheckIcon className="w-5 h-5"/>
            {t.useThisVariationButton}
          </button>
        </div>
      </div>
    </div>
  );
};

export default CharacterSuggestionModal;