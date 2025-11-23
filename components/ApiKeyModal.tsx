import React, { useState, useEffect } from 'react';
import type { TranslationKeys } from '../translations';
import Cog6ToothIcon from './icons/Cog6ToothIcon';
import SaveIcon from './icons/SaveIcon';

interface ApiKeyModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (keys: string[]) => void;
  currentKeys: string[];
  t: TranslationKeys;
}

const ApiKeyModal: React.FC<ApiKeyModalProps> = ({ isOpen, onClose, onSave, currentKeys, t }) => {
  const [keysText, setKeysText] = useState('');

  useEffect(() => {
    // Update textarea only when the modal is opened or currentKeys prop changes
    if (isOpen) {
      setKeysText(currentKeys.join('\n'));
    }
  }, [currentKeys, isOpen]);

  if (!isOpen) {
    return null;
  }

  const handleSave = () => {
    const keysArray = keysText.split('\n').map(k => k.trim()).filter(k => k.length > 0);
    onSave(keysArray);
    onClose();
  };

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-75 flex justify-center items-center z-50 p-4"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="api-key-modal-title"
    >
      <div
        className="bg-[#1E1E22] rounded-lg shadow-2xl w-full max-w-lg border-2 border-gray-700 flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="px-6 py-4 flex justify-between items-center border-b border-gray-700">
          <h2 id="api-key-modal-title" className="text-xl font-bold text-gray-100 flex items-center gap-x-3">
            <Cog6ToothIcon className="w-6 h-6 text-gray-400" />
            {t.apiKeyModalTitle}
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
        <div className="p-8 space-y-4">
          <div>
            <label htmlFor="google-api-keys" className="block text-sm font-medium text-gray-300 mb-2">
              {t.googleApiKeysLabel}
            </label>
            <textarea
              id="google-api-keys"
              rows={8}
              className="w-full bg-[#0D0D0F] text-gray-300 p-3 rounded-md border border-gray-600 focus:ring-2 focus:ring-[#5BEAFF] focus:border-[#5BEAFF] transition font-mono text-sm"
              placeholder={t.apiKeyInputPlaceholder}
              value={keysText}
              onChange={(e) => setKeysText(e.target.value)}
            />
             <p className="text-xs text-gray-500 mt-2">{t.apiKeyInstructions}</p>
          </div>
        </div>
        <div className="bg-gray-800/50 px-6 py-4 flex justify-end gap-3 border-t border-gray-700">
          <button
            type="button"
            className="flex items-center gap-x-2 rounded-md border border-transparent shadow-sm px-4 py-2 bg-[#5BEAFF] text-base font-bold text-black hover:bg-opacity-90 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 sm:w-auto sm:text-sm transition-colors"
            onClick={handleSave}
          >
            <SaveIcon className="w-5 h-5" />
            {t.saveKeysButton}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ApiKeyModal;