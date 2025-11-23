import React, { useState, useEffect } from 'react';
import type { TranslationKeys } from '../translations';
import SparklesIcon from './icons/SparklesIcon';
import ArrowPathIcon from './icons/ArrowPathIcon';

interface StorySuggestionModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAccept: (idea: string) => void;
  onRegenerate: () => void;
  isLoading: boolean;
  idea: string;
  error: string | null;
  cooldown: boolean;
  t: TranslationKeys;
}

const StorySuggestionModal: React.FC<StorySuggestionModalProps> = ({
  isOpen,
  onClose,
  onAccept,
  onRegenerate,
  isLoading,
  idea,
  error,
  cooldown,
  t,
}) => {
  const [editedIdea, setEditedIdea] = useState('');

  useEffect(() => {
    setEditedIdea(idea);
  }, [idea]);

  if (!isOpen) {
    return null;
  }

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-75 flex justify-center items-center z-50 p-4"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="story-suggestion-modal-title"
    >
      <div
        className="bg-[#1E1E22] rounded-lg shadow-2xl w-full max-w-2xl border-2 border-gray-700 flex flex-col max-h-[90vh]"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="px-6 py-4 flex justify-between items-center border-b border-gray-700">
          <h2 id="story-suggestion-modal-title" className="text-xl font-bold text-gray-100 flex items-center gap-x-3">
            <SparklesIcon className="w-6 h-6 text-[#5BEAFF]" />
            {t.storySuggestionModalTitle}
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
            <div className="flex flex-col items-center justify-center h-48">
              <div className="w-12 h-12 border-4 border-dashed rounded-full animate-spin border-[#5BEAFF]"></div>
              <p className="text-[#5BEAFF] text-lg font-medium mt-4">{t.suggestionLoadingText}</p>
            </div>
          )}
          {error && !isLoading && (
            <div className="bg-red-900/50 border border-red-700 text-red-200 px-4 py-3 rounded-lg text-center">
              <p>{error}</p>
            </div>
          )}
          {!isLoading && !error && (
            <div className="space-y-3 h-full flex flex-col">
              <p className="text-sm text-gray-400">{t.storySuggestionEditHint}</p>
              <textarea
                rows={12}
                className="w-full flex-grow bg-[#0D0D0F] text-gray-300 p-4 rounded-lg border border-gray-600 focus:ring-2 focus:ring-[#5BEAFF] focus:border-[#5BEAFF] transition text-base leading-relaxed"
                value={editedIdea || ''}
                onChange={(e) => setEditedIdea(e.target.value)}
              />
            </div>
          )}
        </div>
        <div className="bg-gray-800/50 px-6 py-4 flex justify-end gap-3 border-t border-gray-700">
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
            className="rounded-md border border-transparent shadow-sm px-4 py-2 bg-[#5BEAFF] text-base font-bold text-black hover:bg-opacity-90 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 sm:w-auto sm:text-sm transition-colors disabled:bg-gray-500"
            onClick={() => onAccept(editedIdea)}
            disabled={isLoading || !editedIdea?.trim() || !!error}
          >
            {t.useThisIdeaButton}
          </button>
        </div>
      </div>
    </div>
  );
};

export default StorySuggestionModal;