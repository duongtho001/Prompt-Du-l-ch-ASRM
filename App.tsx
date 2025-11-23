import React, { useState, useEffect } from 'react';
import Header from './components/Header';
// FIX: Changed to a named import to resolve module loading error.
import { InputPanel } from './components/InputPanel';
import SceneTimeline from './components/SceneTimeline';
import type { Scene, CharacterReference, VideoConfig, ScenePrompt, CharacterVariation } from './types';
import { generateScenePrompts, generateScript, generateStoryIdea, generateSceneImage, generateCharacterPromptFromImage, generateCharacterPromptVariations } from './services/geminiService';
import { translations, type Language } from './translations';
import GuideModal from './components/GuideModal';
import ConfirmationModal from './components/ConfirmationModal';
import ExclamationTriangleIcon from './components/icons/ExclamationTriangleIcon';
import StorySuggestionModal from './components/StorySuggestionModal';
import ContinueGenerationModal from './components/ContinueGenerationModal';
import CharacterSuggestionModal from './components/CharacterSuggestionModal';
import ApiKeyModal from './components/ApiKeyModal';
import { getKeys, saveKeys } from './services/apiKeyManager';


declare var JSZip: any;

const SCENES_PER_BATCH = 10;
const SCENE_DURATION_SECONDS = 8;

const isRetryableError = (errorMessage: string): boolean => {
  const lowerCaseError = errorMessage.toLowerCase();
  return lowerCaseError.includes('quota') 
      || lowerCaseError.includes('resource_exhausted') 
      || lowerCaseError.includes('overloaded') 
      || lowerCaseError.includes('unavailable') 
      || lowerCaseError.includes('503');
};

const App: React.FC = () => {
  const [language, setLanguage] = useState<Language>('vi');
  const t = translations[language];

  // Project State
  const [projectName, setProjectName] = useState<string>(t.untitledProject);
  const [characterReferences, setCharacterReferences] = useState<CharacterReference[]>([]);
  const [storyIdea, setStoryIdea] = useState<string>('');
  const [storyStyle, setStoryStyle] = useState<string>('adventure');
  const [generatedScript, setGeneratedScript] = useState<string>('');
  const [videoConfig, setVideoConfig] = useState<VideoConfig>({ 
    duration: 0, 
    style: 'ghibli',
    includeDialogue: false,
    dialogueLanguage: 'vi',
    format: 'narrative_short',
  });
  const [scenes, setScenes] = useState<Scene[]>([]);
  
  // UI & Loading State
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isGeneratingPromptId, setIsGeneratingPromptId] = useState<string | null>(null);
  const [isSuggestingPromptId, setIsSuggestingPromptId] = useState<string | null>(null);
  const [isBatchGenerating, setIsBatchGenerating] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [apiKeys, setApiKeys] = useState<string[]>([]);

  // Modals State
  const [isGuideVisible, setIsGuideVisible] = useState<boolean>(false);
  const [isNewProjectConfirmVisible, setIsNewProjectConfirmVisible] = useState<boolean>(false);
  const [isResumeModalVisible, setIsResumeModalVisible] = useState<boolean>(false);
  const [isSuggestionModalVisible, setIsSuggestionModalVisible] = useState<boolean>(false);
  const [isContinueModalVisible, setIsContinueModalVisible] = useState(false);
  const [isCharacterSuggestionModalVisible, setIsCharacterSuggestionModalVisible] = useState(false);
  const [isApiKeyModalVisible, setIsApiKeyModalVisible] = useState(false);

  // Generation Progress & Control State
  const [generationProgress, setGenerationProgress] = useState<{ current: number; total: number }>({ current: 0, total: 0 });
  const [isGenerationComplete, setIsGenerationComplete] = useState<boolean>(false);
  const [generationStatusMessage, setGenerationStatusMessage] = useState<string>('');
  const [cooldowns, setCooldowns] = useState<{ [key: string]: boolean }>({});
  const [retryAction, setRetryAction] = useState<(() => void) | null>(null);
  
  // Suggestion Modals Content State
  const [isSuggesting, setIsSuggesting] = useState(false);
  const [suggestionCooldown, setSuggestionCooldown] = useState(false);
  const [suggestedStoryIdea, setSuggestedStoryIdea] = useState<string>('');
  const [suggestionError, setSuggestionError] = useState<string | null>(null);
  const [characterSuggestions, setCharacterSuggestions] = useState<CharacterVariation[]>([]);
  const [charSuggestionError, setCharSuggestionError] = useState<string | null>(null);
  const [suggestionTargetCharacterId, setSuggestionTargetCharacterId] = useState<string | null>(null);

  useEffect(() => {
    const loadedKeys = getKeys();
    setApiKeys(loadedKeys);
    if (loadedKeys.length === 0) {
      setIsApiKeyModalVisible(true);
    }
  }, []);

  useEffect(() => {
    const firstLine = storyIdea.split('\n')[0].trim();
    if (firstLine.length > 0 && firstLine.length < 50) {
      setProjectName(firstLine);
    } else {
      setProjectName(t.untitledProject);
    }
  }, [storyIdea, t.untitledProject]);

  const startCooldown = (key: string, duration: number) => {
    setCooldowns(prev => ({...prev, [key]: true}));
    setTimeout(() => {
      setCooldowns(prev => ({...prev, [key]: false}));
    }, duration);
  };
  
  const handleError = (err: unknown, retryFn: () => void) => {
    const errorMessage = err instanceof Error ? err.message : t.errorUnknown('N/A');
    setIsLoading(false); 
    setIsSuggesting(false);
    setIsSuggestingPromptId(null);
    setIsGeneratingPromptId(null);

    if (isRetryableError(errorMessage)) {
      setError(errorMessage);
      setRetryAction(() => retryFn);
      setIsResumeModalVisible(true);
    } else {
      setError(errorMessage);
    }
  };

  const resetState = () => {
    setProjectName(t.untitledProject);
    setCharacterReferences([]);
    setStoryIdea('');
    setStoryStyle('adventure');
    setGeneratedScript('');
    setScenes([]);
    setVideoConfig({ 
        duration: 0, 
        style: 'ghibli',
        includeDialogue: false,
        dialogueLanguage: 'vi',
        format: 'narrative_short',
    });
    setError(null);
    setIsLoading(false);
    setIsSuggesting(false);
    setIsGeneratingPromptId(null);
    setIsSuggestingPromptId(null);
    setIsBatchGenerating(false);
    setIsGenerationComplete(false);
    setGenerationProgress({ current: 0, total: 0 });
    setGenerationStatusMessage('');
    setCooldowns({});
    setRetryAction(null);
  };

  const handleNewProjectRequest = () => {
    if (storyIdea || characterReferences.length > 0 || scenes.length > 0 || generatedScript) {
        setIsNewProjectConfirmVisible(true);
    } else {
        resetState();
    }
  };

  const handleConfirmNewProject = () => {
      resetState();
      setIsNewProjectConfirmVisible(false);
  };
  
  const clearGeneratedContent = () => {
    if (generatedScript || scenes.length > 0) {
      setGeneratedScript('');
      setScenes([]);
      setIsGenerationComplete(false);
    }
  };

  const handleAddCharacter = () => {
    const newCharacter: CharacterReference = {
        id: crypto.randomUUID(),
        name: `${t.newCharacterName} ${characterReferences.length + 1}`,
        prompt: '',
        imageUrl: null,
    };
    setCharacterReferences(prev => [...prev, newCharacter]);
    clearGeneratedContent();
  };

  const handleUpdateCharacter = (id: string, field: keyof CharacterReference, value: string | null) => {
    setCharacterReferences(prev => prev.map(char => char.id === id ? { ...char, [field]: value } : char));
    if (field === 'prompt') return; // Don't clear script if only editing prompt text manually
    clearGeneratedContent();
  };
  
  const handleDeleteCharacter = (id: string) => {
    setCharacterReferences(prev => prev.filter(char => char.id !== id));
    clearGeneratedContent();
  };

  const handleImportCharacters = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
        const content = e.target?.result as string;
        if (!content) return;

        const newCharacters: CharacterReference[] = [];
        const blocks = content.split(/\[Character Name:/).filter(block => block.trim() !== '');

        blocks.forEach(block => {
            const lines = block.split('\n');
            const name = lines[0]?.replace(']', '').trim();
            const prompt = lines.slice(1).join('\n').trim();

            if (name && prompt) {
                newCharacters.push({
                    id: crypto.randomUUID(),
                    name,
                    prompt,
                    imageUrl: null
                });
            }
        });

        if (newCharacters.length > 0) {
            setCharacterReferences(prev => [...prev, ...newCharacters]);
            clearGeneratedContent();
        }
    };
    reader.readAsText(file);
  };

  const handleStoryIdeaChange = (value: string) => {
    setStoryIdea(value);
    clearGeneratedContent();
  };

  const handleVideoConfigChange = (value: React.SetStateAction<VideoConfig>) => {
    setVideoConfig(value);
    clearGeneratedContent();
  };

  const handleGenerateCharacterPrompt = async (characterId: string, file: File) => {
    setIsGeneratingPromptId(characterId);
    setError(null);
    try {
      const reader = new FileReader();
      reader.onloadend = async () => {
        const imageUrl = reader.result as string;
        handleUpdateCharacter(characterId, 'imageUrl', imageUrl);
        try {
          const prompt = await generateCharacterPromptFromImage(imageUrl, language);
          handleUpdateCharacter(characterId, 'prompt', prompt);
        } catch (err) {
          handleError(err, () => handleGenerateCharacterPrompt(characterId, file));
        } finally {
          setIsGeneratingPromptId(null);
        }
      };
      reader.readAsDataURL(file);
    } catch (err) {
      handleError(err, () => handleGenerateCharacterPrompt(characterId, file));
      setIsGeneratingPromptId(null);
    }
  };

  const handleSuggestCharacterPrompt = async (characterId: string) => {
    setSuggestionTargetCharacterId(characterId);
    setIsCharacterSuggestionModalVisible(true);
    await handleRegenerateCharacterVariations();
  };

  const handleRegenerateCharacterVariations = async () => {
    const character = characterReferences.find(c => c.id === suggestionTargetCharacterId);
    if (!character) return;
    
    setIsSuggesting(true);
    setCharSuggestionError(null);
    setSuggestionCooldown(true);
    
    try {
      const variations = await generateCharacterPromptVariations(
        character.name,
        videoConfig.style,
        storyStyle,
        language
      );
      setCharacterSuggestions(variations);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : t.errorUnknown('N/A');
      setCharSuggestionError(errorMessage);
    } finally {
      setIsSuggesting(false);
      setTimeout(() => setSuggestionCooldown(false), 5000);
    }
  };

  const handleAcceptCharacterVariation = (description: string) => {
    if (suggestionTargetCharacterId) {
      handleUpdateCharacter(suggestionTargetCharacterId, 'prompt', description);
    }
    setIsCharacterSuggestionModalVisible(false);
    setCharacterSuggestions([]);
    setSuggestionTargetCharacterId(null);
  };
  
  const handleGenerateStoryIdea = async () => {
    setIsSuggestionModalVisible(true);
    await handleRegenerateStory();
  };

  const handleRegenerateStory = async () => {
    setIsSuggesting(true);
    setSuggestionError(null);
    setSuggestionCooldown(true);
    
    try {
      const characterDescriptions = characterReferences
        .map(c => `[${c.name}]: ${c.prompt}`)
        .join('\n');
      const idea = await generateStoryIdea(videoConfig.style, storyStyle, language, characterDescriptions);
      setSuggestedStoryIdea(idea);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : t.errorUnknown('N/A');
      setSuggestionError(errorMessage);
    } finally {
      setIsSuggesting(false);
      setTimeout(() => setSuggestionCooldown(false), 5000);
    }
  };

  const handleAcceptSuggestedStory = (idea: string) => {
    setStoryIdea(idea);
    setIsSuggestionModalVisible(false);
    setSuggestedStoryIdea('');
  };

  const handlePrimaryAction = async () => {
    if (generatedScript) {
      await handleGenerateStoryboard();
    } else {
      await handleGenerateScript();
    }
  };

  const handleGenerateScript = async () => {
    setIsLoading(true);
    setError(null);
    startCooldown('mainAction', 5000);
    try {
      const characterDescriptions = characterReferences
        .map(c => `[${c.name}]: ${c.prompt}`)
        .join('\n\n');
      const script = await generateScript(storyIdea, videoConfig, language, characterDescriptions);
      setGeneratedScript(script);
    } catch (err) {
      handleError(err, handleGenerateScript);
    } finally {
      setIsLoading(false);
    }
  };

  const handleGenerateStoryboard = async () => {
    setIsLoading(true);
    setError(null);
    setIsContinueModalVisible(false);
    startCooldown('mainAction', 5000);

    const totalScenes = Math.round(videoConfig.duration / SCENE_DURATION_SECONDS);
    setGenerationProgress({ current: scenes.length, total: totalScenes });
    setIsGenerationComplete(false);
    setGenerationStatusMessage(t.generationStatusPreparing);

    try {
      const lastScene = scenes.length > 0 ? scenes[scenes.length - 1] : null;

      const characterDescriptions = characterReferences
        .map(c => `[${c.name}]: ${c.prompt}`)
        .join('\n\n');
      
      setGenerationStatusMessage(t.generationStatusRequesting(Math.floor(scenes.length / SCENES_PER_BATCH) + 1));
      
      const newScenes = await generateScenePrompts(
        generatedScript,
        videoConfig,
        language,
        characterDescriptions,
        scenes.length,
        SCENES_PER_BATCH,
        lastScene
      );
      
      setScenes(prev => [...prev, ...newScenes]);
      
      const newTotalGenerated = scenes.length + newScenes.length;
      setGenerationProgress({ current: newTotalGenerated, total: totalScenes });
      
      if (newTotalGenerated < totalScenes) {
        setIsContinueModalVisible(true);
      } else {
        setIsGenerationComplete(true);
      }
      
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : t.errorUnknown('generateScenePrompts');
      setError(t.generationFailedCanResume(errorMessage));
      setRetryAction(() => handleGenerateStoryboard);
    } finally {
      setIsLoading(false);
    }
  };

  const handleGenerateSceneImage = async (sceneId: number, referenceCharacterId: string) => {
    const sceneIndex = scenes.findIndex(s => s.scene_id === sceneId);
    if (sceneIndex === -1) return;
    
    const reference = characterReferences.find(c => c.id === referenceCharacterId);
    if (!reference?.imageUrl) return;

    setScenes(prev => prev.map(s => s.scene_id === sceneId ? { ...s, isGeneratingImage: true } : s));
    try {
      const imageUrl = await generateSceneImage(scenes[sceneIndex].prompt, reference.imageUrl, language);
      setScenes(prev => prev.map(s => s.scene_id === sceneId ? { ...s, imageUrl, isGeneratingImage: false } : s));
    } catch (error) {
      console.error(error);
      setError(t.errorGeneratingImage);
    } finally {
      setScenes(prev => prev.map(s => s.scene_id === sceneId ? { ...s, isGeneratingImage: false } : s));
    }
  };

  const handleGenerateAllSceneImages = async (referenceCharacterId: string) => {
    const reference = characterReferences.find(c => c.id === referenceCharacterId);
    if (!reference?.imageUrl) return;

    setIsBatchGenerating(true);
    const scenesToGenerate = scenes.filter(s => !s.imageUrl);

    for (const scene of scenesToGenerate) {
      if(isBatchGenerating) { // check if still generating
         await handleGenerateSceneImage(scene.scene_id, referenceCharacterId);
      }
    }
    setIsBatchGenerating(false);
  };
  
  const handleDownloadPrompts = async () => {
    const sceneStrings = scenes.map(scene => {
      const promptText = `Phân cảnh ${scene.scene_id}: ${JSON.stringify(scene.prompt)}`;
      return promptText;
    });
    const content = sceneStrings.join('\n');
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `${projectName}_prompts.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleDownloadAllImages = async () => {
    const zip = new JSZip();
    const imageScenes = scenes.filter(s => s.imageUrl);

    for (const scene of imageScenes) {
        if (scene.imageUrl) {
            const response = await fetch(scene.imageUrl);
            const blob = await response.blob();
            zip.file(`scene_${scene.scene_id}.png`, blob);
        }
    }

    zip.generateAsync({ type: 'blob' }).then((content: Blob) => {
        const link = document.createElement('a');
        link.href = URL.createObjectURL(content);
        link.download = `${projectName}_images.zip`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });
  };

  const handleResume = () => {
    setIsResumeModalVisible(false);
    if (retryAction) {
      retryAction();
      setRetryAction(null);
    }
  };
  
  const handleSaveApiKeys = (keys: string[]) => {
    saveKeys(keys);
    setApiKeys(keys);
  };

  return (
    <div className="min-h-screen bg-[#0D0D0F] text-gray-200">
      <Header 
        language={language} 
        setLanguage={setLanguage} 
        t={t} 
        onOpenGuide={() => setIsGuideVisible(true)}
        onNewProject={handleNewProjectRequest}
        onOpenApiSettings={() => setIsApiKeyModalVisible(true)}
      />
      <main className="grid grid-cols-1 lg:grid-cols-2 gap-8 p-8 max-w-[1600px] mx-auto">
        <InputPanel
          scenes={scenes}
          characterReferences={characterReferences}
          onUpdateCharacter={handleUpdateCharacter}
          onAddCharacter={handleAddCharacter}
          onDeleteCharacter={handleDeleteCharacter}
          onGenerateCharacterPrompt={handleGenerateCharacterPrompt}
          onSuggestCharacterPrompt={handleSuggestCharacterPrompt}
          onImportCharacters={handleImportCharacters}
          isGeneratingPromptId={isGeneratingPromptId}
          isSuggestingPromptId={isSuggestingPromptId}
          storyIdea={storyIdea}
          setStoryIdea={handleStoryIdeaChange}
          storyStyle={storyStyle}
          setStoryStyle={setStoryStyle}
          generatedScript={generatedScript}
          videoConfig={videoConfig}
          setVideoConfig={handleVideoConfigChange}
          onPrimaryAction={handlePrimaryAction}
          isLoading={isLoading}
          onGenerateStoryIdea={handleGenerateStoryIdea}
          cooldowns={cooldowns}
          t={t}
          language={language}
        />
        <SceneTimeline
          scenes={scenes}
          onUpdatePrompt={(sceneId, newPrompt) => setScenes(prev => prev.map(s => s.scene_id === sceneId ? {...s, prompt: newPrompt} : s))}
          isLoading={isLoading}
          isBatchGenerating={isBatchGenerating}
          onDownloadPrompts={handleDownloadPrompts}
          onDownloadAllImages={handleDownloadAllImages}
          characterReferences={characterReferences}
          onGenerateSceneImage={handleGenerateSceneImage}
          onGenerateAllSceneImages={handleGenerateAllSceneImages}
          t={t}
          isGenerationComplete={isGenerationComplete}
          generationProgress={generationProgress}
          generationStatusMessage={generationStatusMessage}
        />
      </main>

      {error && !isResumeModalVisible && (
        <div className="fixed bottom-5 right-5 bg-red-800/90 text-white p-4 rounded-lg shadow-lg flex items-center gap-x-3 border border-red-600">
          <ExclamationTriangleIcon className="w-6 h-6" />
          <p className="text-sm font-medium">{error}</p>
          <button onClick={() => setError(null)} className="ml-4 text-red-200 hover:text-white">&times;</button>
        </div>
      )}

      <GuideModal isOpen={isGuideVisible} onClose={() => setIsGuideVisible(false)} t={t} />

      <ConfirmationModal
        isOpen={isNewProjectConfirmVisible}
        onClose={() => setIsNewProjectConfirmVisible(false)}
        onConfirm={handleConfirmNewProject}
        title={t.newProjectConfirmationTitle}
        message={t.newProjectConfirmationMessage}
        confirmText={t.confirmButton}
        cancelText={t.cancelButton}
        icon={<ExclamationTriangleIcon className="w-16 h-16 text-yellow-400" />}
      />
      
      <ConfirmationModal
        isOpen={isResumeModalVisible}
        onClose={() => {
          setIsResumeModalVisible(false);
          setError(null);
        }}
        onConfirm={handleResume}
        title={t.resumeGenerationTitle}
        message={error || ''}
        confirmText={t.resumeButton}
        cancelText={t.finishForNowButton}
        icon={<ExclamationTriangleIcon className="w-16 h-16 text-yellow-400" />}
        confirmButtonClass="bg-yellow-600 hover:bg-yellow-700 focus:ring-yellow-500"
      />

      <StorySuggestionModal
        isOpen={isSuggestionModalVisible}
        onClose={() => {
          setIsSuggestionModalVisible(false);
          setSuggestionError(null);
        }}
        onAccept={handleAcceptSuggestedStory}
        onRegenerate={handleRegenerateStory}
        isLoading={isSuggesting}
        idea={suggestedStoryIdea}
        error={suggestionError}
        cooldown={suggestionCooldown}
        t={t}
      />
      
      <CharacterSuggestionModal
        isOpen={isCharacterSuggestionModalVisible}
        onClose={() => {
          setIsCharacterSuggestionModalVisible(false);
          setCharSuggestionError(null);
        }}
        onAccept={handleAcceptCharacterVariation}
        onRegenerate={handleRegenerateCharacterVariations}
        isLoading={isSuggesting}
        variations={characterSuggestions}
        error={charSuggestionError}
        cooldown={suggestionCooldown}
        t={t}
      />
      
      <ContinueGenerationModal
        isOpen={isContinueModalVisible}
        onClose={() => setIsContinueModalVisible(false)}
        onConfirm={handleGenerateStoryboard}
        generatedCount={scenes.length}
        totalCount={Math.round(videoConfig.duration / SCENE_DURATION_SECONDS)}
        t={t}
      />

      <ApiKeyModal
        isOpen={isApiKeyModalVisible}
        onClose={() => setIsApiKeyModalVisible(false)}
        onSave={handleSaveApiKeys}
        currentKeys={apiKeys}
        t={t}
      />
    </div>
  );
};

export default App;