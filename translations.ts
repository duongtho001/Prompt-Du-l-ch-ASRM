import type { VideoConfig, Scene } from './types';
import { DIALOGUE_LANGUAGES } from './constants';

export type Language = 'en' | 'vi';

const en = {
    // App.tsx
    untitledProject: "Untitled Animation",
    generationStatusPreparing: "Preparing to generate scenes...",
    generationStatusRequesting: (batch: number) => `Requesting scene batch #${batch}...`,
    generationIncompleteError: (current: number, total: number) => `Generation incomplete. Only ${current} out of ${total} scenes were generated. You can try to resume.`,
    errorGeneratingImage: "An error occurred while generating the image.",
    errorQuotaExceeded: "API quota exceeded. You've made too many requests in a short period. Please wait a moment and try again. You can monitor your usage here: https://ai.dev/usage",
    errorInvalidApiKey: (context: string) => `The API key is not valid. Please check your key in the settings. (Context: ${context})`,
    errorServerOverloaded: (context: string) => `The model is currently busy or overloaded. Please try again in a few moments. (Context: ${context})`,
    errorGeneric: (context: string, message: string) => `Error in ${context}: ${message}`,
    errorUnknown: (context: string) => `An unknown error occurred in ${context}.`,
    generationFailedCanResume: (errorMsg: string) => `Scene generation failed: ${errorMsg}. You can try to resume the process.`,
    errorGeneratingPrompt: "An error occurred while generating the character prompt from the image.",
    newCharacterName: "Character",
    errorMissingApiKey: (provider: string) => `API Key for ${provider} is missing. Please add at least one key in the API Settings.`,
    errorAllKeysExhausted: "All provided API keys have reached their usage quota. Please add a new key or wait for the quota to reset.",

    // Header.tsx
    appTitle: "Chuyên Hoạt Hình",
    appDescription: "AI-Powered Animation Studio",
    newProjectButton: "New Project",
    guideButtonTooltip: "Show User Guide",
    apiSettingsTooltip: "API Settings",
    languageLabel: "Language",

    // InputPanel.tsx
    characterRosterLabel: "Character Roster",
    addCharacterButton: "Add Character",
    importFromFileButton: "Import from .txt",
    characterNamePlaceholder: "Character Name...",
    characterImageLabel: "Reference Image",
    uploadImagePrompt: "Upload Image",
    generatePromptFromImageButton: "From Image",
    generatingPromptButton: "Generating...",
    suggestPromptButton: "Suggest Prompt",
    suggestingPromptButton: "Suggesting...",
    characterPromptLabel: "Character DNA Prompt",
    characterPromptPlaceholder: "Describe your character's appearance, clothing, and style in detail, or generate a prompt from an image.",
    storyIdeaLabel: "Story / Script",
    suggestIdeaButton: "Suggest a Story",
    onCooldownButton: "On Cooldown...",
    suggestingIdeaButton: "Suggesting...",
    storyIdeaPlaceholder: "Describe your animation idea or paste a short script. Or, first create a character prompt, select a story style, and click 'Suggest a Story' for an AI-generated idea based on your character.",
    generatedScriptLabel: "Generated Animation Script",
    videoSettingsLabel: "Animation Settings",
    durationLabel: "Approximate Animation Duration (minutes)",
    durationPlaceholder: "e.g., 1",
    durationFeedback: (scenes: number, mins: number, secs: number) => `~${scenes} scenes (~${mins}m ${secs}s).`,
    videoFormatLabel: "Animation Pace / Format",
    styleLabel: "Animation Style",
    storyStyleLabel: "Story Style",
    aiProviderLabel: "AI Provider",
    aiModelLabel: "AI Model",
    generateScriptButton: "Generate Animation Script",
    generatingScriptButton: "Generating Script...",
    generateStoryboardButton: "Generate Storyboard",
    generatingStoryboardButton: "Generating Storyboard...",
    continueGenerateStoryboardButton: "Continue Generating Storyboard",
    includeDialogueLabel: "Include Dialogue",
    dialogueLanguageLabel: "Dialogue Language",

    // SceneTimeline.tsx
    timelineTitle: "Storyboard Timeline",
    downloadButton: "Download Prompts",
    primaryReferenceLabel: "Primary Reference for Batch Generation",
    selectPrimaryReferencePrompt: "Select a character/style reference",
    generateAllImagesButton: "Generate All Missing Images",
    generatingAllImagesButton: "Generating All...",
    downloadAllImagesButton: "Download All Images (.zip)",
    emptyTimelineTitle: "Your storyboard is empty.",
    emptyTimelineDescription: "Provide a story and generate a script to see your animation scenes here.",

    // SceneCard.tsx
    sceneLabel: "Scene",
    timeLabel: "Time",
    promptLabel: "Prompt (JSON)",
    promptHelperTooltip: "Toggle prompt helper",
    invalidJsonError: "Invalid JSON format. Please correct it.",
    sceneImageLabel: "Generated Scene Image",
    selectReferenceImageLabel: "Reference Image",
    noImageGenerated: "No image generated.",
    generateImageButton: "Generate Image",
    generatingImageButton: "Generating...",
    noReferenceImagesAvailable: "No reference images available",

    // Loader.tsx
    generationComplete: "Generation Complete!",
    generatingScene: (current: number, total: number) => `Generating Scene ${current} of ${total}`,
    loaderText: "Generating...",

    // ConfirmationModal.tsx & App.tsx
    newProjectConfirmationTitle: "Start a New Project?",
    newProjectConfirmationMessage: "This will clear all character references, stories, and scenes. Are you sure you want to continue?",
    confirmButton: "Confirm",
    cancelButton: "Cancel",
    resumeGenerationTitle: "Action Paused",
    resumeButton: "Resume",
    finishForNowButton: "Finish for Now",

    // GuideModal.tsx
    guideModalTitle: "How to Use Chuyên Hoạt Hình",
    guideSteps: [
        { title: "Step 1: Set Your API Keys", description: "Before you begin, click the gear icon in the header to open the API Settings. Add one or more Google AI API keys. If you provide multiple, the app will automatically switch to the next one if a key runs out of quota." },
        { title: "Step 2: Define Your Characters", description: "This is the most important step. Add one or more characters. For each character, provide a name and a detailed description in their 'prompt' box. You can also upload a reference image and click 'Generate Prompt from Image' to have the AI create a description for you. This roster of characters ensures consistency." },
        { title: "Step 3 (Optional): Import Characters", description: "To work faster, you can import multiple characters from a `.txt` file. Click 'Import from .txt' and choose your file. The file must follow this format (blank lines between characters are important):\n[Character Name: Character A]\nA detailed description of Character A...\n\n[Character Name: Character B]\nA detailed description of Character B..." },
        { title: "Step 4: Get a Story Idea", description: "Once your characters are ready, select a 'Story Style' and click 'Suggest a Story'. The AI will generate a story concept featuring your characters in that genre. You can edit this story as much as you like." },
        { title: "Step 5: Set Animation Style & Duration", description: "Choose an animation style and the desired length of your video in minutes. This sets the overall look and feel." },
        { title: "Step 6: Generate Animation Script", description: "Click 'Generate Animation Script'. The AI will expand your story into a structured script, keeping your characters' descriptions in mind." },
        { title: "Step 7: Generate Storyboard", description: "With the script ready, click 'Generate Storyboard' to create detailed, AI-ready prompts for each scene that are consistent with your characters." },
        { title: "Step 8: Generate Scene Images", description: "In the timeline, use your uploaded reference images to generate visuals for each scene, either one-by-one or all at once." },
        { title: "Step 9: Review and Export", description: "Review your storyboard. You can edit the JSON prompts for any scene to fine-tune the visuals, then download all prompts or images." },
    ],
    guideProTipsTitle: "Pro-Tips",
    guideProTips: [
        { title: "Detailed Character Prompts are Key", description: "The more detail you provide for each character (clothing, hair, distinctive features, style), the more consistent your results will be." },
        { title: "Use a Clear Reference Image", description: "For both prompt generation and image generation, use a clear, front-facing image of your character for the best results." },
        { title: "Name Characters in the Script", description: "When editing the script or story idea, use the character names you defined. The AI will use these to create more accurate scene descriptions." },
        { title: "Iterate and Refine", description: "The generated JSON prompts are fully editable. Change character expressions, camera shots, or lighting to perfect your vision." },
    ],

    // PromptHelper.tsx
    promptHelperTitle: "Prompt Quick-Adds",
    promptHelperTags: {
        camera_shots: {
            group: "Camera Shots",
            tags: [
                { tag: "establishing shot", desc: "A very wide shot to show the location and set the scene." },
                { tag: "full shot", desc: "Shows the full character from head to toe." },
                { tag: "medium shot", desc: "Shows character from the waist up, good for dialogue." },
                { tag: "close-up shot", desc: "Focuses on the character's face for emotion." },
                { tag: "extreme close-up", desc: "Focuses on a small detail, like an eye or an object." },
                { tag: "over-the-shoulder shot", desc: "View from behind one character looking at another." },
                { tag: "insert shot", desc: "A close-up of a specific object or detail relevant to the story." },
                { tag: "POV shot", desc: "From the character's point of view." },
            ],
        },
        camera_angles: {
            group: "Camera Angles",
            tags: [
                { tag: "eye-level angle", desc: "Neutral and standard perspective." },
                { tag: "low-angle", desc: "Makes subject look powerful or imposing." },
                { tag: "high-angle", desc: "Makes subject look small or vulnerable." },
                { tag: "dutch angle / tilted shot", desc: "Tilted camera, creates tension, unease, or adventure." },
                { tag: "bird's-eye view / top-down shot", desc: "Directly above the scene, shows overall layout." },
            ],
        },
        camera_movement: {
            group: "Camera Movement",
            tags: [
                { tag: "static shot", desc: "Camera is completely still." },
                { tag: "pan shot (left/right)", desc: "Camera pivots horizontally on a fixed axis." },
                { tag: "tilt shot (up/down)", desc: "Camera pivots vertically on a fixed axis." },
                { tag: "tracking shot / dolly shot", desc: "Camera moves alongside the subject." },
                { tag: "crane shot / pedestal shot", desc: "Camera moves vertically up or down." },
                { tag: "dolly zoom / vertigo effect", desc: "Zooming while moving the camera in the opposite direction." },
                { tag: "orbit shot / 360-degree shot", desc: "Camera circles around the subject." },
            ]
        },
        action_shots: {
            group: "Action & Dynamic Shots",
            tags: [
              { tag: "slow motion", desc: "Slows down the action to emphasize movement or impact." },
              { tag: "hand tracking shot", desc: "Camera follows the movement of a character's hand." },
              { tag: "foot tracking shot", desc: "Camera follows the movement of a character's feet." },
              { tag: "object tracking shot", desc: "Camera follows a moving object, like a weapon or vehicle." },
              { tag: "motion-follow camera", desc: "Camera moves at the same speed as the character, often used in fight scenes." },
              { tag: "hero shot", desc: "Slow motion combined with a camera follow to make a character look epic." },
              { tag: "insert action shot", desc: "A close-up of a specific action detail, like a hand gripping a sword." },
            ]
        },
        styles: {
            group: "Art Styles",
            tags: [
                { tag: "cel-shaded", desc: "Classic 2D anime look." },
                { tag: "watercolor", desc: "Soft, painted backgrounds." },
                { tag: "painterly", desc: "Looks like an oil painting." },
                { tag: "concept art", desc: "Detailed, illustrative style." },
                { tag: "vector art", desc: "Clean lines, solid colors." },
                { tag: "8-bit pixel art", desc: "Retro video game style." },
            ],
        },
        lighting: {
            group: "Lighting",
            tags: [
                { tag: "cinematic lighting", desc: "Dramatic, high-contrast." },
                { tag: "soft, ambient light", desc: "Even, gentle lighting." },
                { tag: "dramatic rim lighting", desc: "Highlights character edges." },
                { tag: "golden hour light", desc: "Warm, magical lighting." },
                { tag: "neon lighting", desc: "Vibrant, cyberpunk feel." },
            ],
        }
    },

    // StorySuggestionModal.tsx
    storySuggestionModalTitle: "AI Story Suggestion",
    useThisIdeaButton: "Use This Idea",
    regenerateIdeaButton: "Regenerate",
    closeButton: "Close",
    suggestionLoadingText: "AI is crafting a story for your characters...",
    storySuggestionEditHint: "You can edit the text below before accepting.",

    // ContinueGenerationModal.tsx
    continueGenerationTitle: "Storyboard Batch Complete",
    continueGenerationMessage: (generated: number, total: number) => `Successfully generated ${generated} out of ${total} total scenes. Do you want to generate the next batch?`,
    continueGenerationButton: "Continue Generation",

    // CharacterSuggestionModal.tsx
    characterSuggestionModalTitle: "Character Prompt Suggestions",
    characterSuggestionLoadingText: "AI is imagining character variations...",
    useThisVariationButton: "Use This Variation",

    // ApiKeyModal.tsx
    apiKeyModalTitle: "API Key Settings",
    googleApiKeysLabel: "Google API Keys (One per line)",
    apiKeyInputPlaceholder: "Enter your API keys, one on each line...",
    apiKeyInstructions: "The app will automatically use the next key if the current one hits its quota.",
    saveKeysButton: "Save Keys",
    
    // Gemini Service System Instructions
    systemInstruction_generateCharacterPrompt: `You are an expert art director specializing in character design for animation.
Your task is to analyze an image of a character and generate a detailed, descriptive text prompt that can be used to recreate this character consistently with an AI image generator.
The description must be in English.
Focus on capturing every key visual detail.

Structure your response as follows:
- **Overall Style:** Briefly describe the art style (e.g., "A charming female character in a vibrant, cel-shaded anime style").
- **Face & Hair:** Describe facial features, expression, eye color, hair color, and hairstyle in detail.
- **Clothing & Attire:** Describe the full outfit from head to toe, including type of clothing, colors, patterns, and accessories.
- **Key Features:** Mention any unique or defining characteristics like scars, tattoos, special items they are holding, or unique design elements.

The response must be only the descriptive text, with no extra formatting or introductory phrases.`,

    systemInstruction_generateCharacterVariations: (characterName: string, animationStyle: string, storyStyle: string) => `You are an expert art director and creative writer specializing in character design for animation.
Your task is to generate THREE distinct and creative variations for a new character based on their name and the project's styles. Each variation should offer a unique take on the character.
The output MUST be a single, valid JSON object.

**Character Name:** ${characterName}
**Animation Style:** ${animationStyle}
**Story Genre:** ${storyStyle}

Based on this information, create three rich and imaginative character descriptions. Each description must be detailed enough to be used as a prompt for an AI image generator. The descriptions must be in **English**.

For each variation, provide:
- A short, evocative \`title\` that captures the essence of that version (e.g., "The Rogue Inventor", "The Solemn Guardian").
- A detailed \`description\` that covers the character's overall style, face, hair, clothing, and key features.

The final output MUST be a single JSON object containing a "variations" array, with no other text before or after it.`,

    systemInstruction_generateStoryIdea: (animationStyle: string, storyStyle: string, characterDescriptions: string) => `You are a creative storyteller for animation. Your task is to generate a short, compelling story concept or synopsis with a clear three-act structure (Setup, Confrontation, Resolution).
The story must be in the **${storyStyle}** genre.
The story must feature the characters described below.
The idea should be concise but well-structured, suitable for the animation style of "${animationStyle}".

**Cast of Characters:**
${characterDescriptions}

Structure your response with these headings:
**Setup:** Introduce the main character(s), their world, and the initial situation.
**Confrontation:** Describe the main conflict, challenge, or goal the character(s) must face.
**Resolution:** Briefly explain how the story concludes and the outcome for the character(s).

The response must be only the story idea text, with no extra formatting or introductory phrases.`,

    systemInstruction_generateScript: (config: VideoConfig, characterDescriptions: string) => {
        const languageName = DIALOGUE_LANGUAGES.find(lang => lang.key === config.dialogueLanguage)?.en || config.dialogueLanguage;
        const dialogueInstruction = config.includeDialogue 
            ? `The script MUST include dialogue where appropriate. It is absolutely critical that all dialogue is written ONLY in the specified language: **${languageName} (${config.dialogueLanguage})**. Do NOT use English for dialogue unless the specified language is English.`
            : `The script should be purely descriptive and contain NO dialogue whatsoever.`;

        const totalScenes = Math.round(config.duration / 8);

        return `You are a professional animation scriptwriter. Your task is to faithfully expand a story idea into a descriptive scene-by-scene script.
The final animation will be approximately ${config.duration} seconds long, which corresponds to EXACTLY ${totalScenes} scenes.

**CRITICAL - SEAMLESS STORY STRUCTURE:** 
- The script MUST consist of EXACTLY ${totalScenes} scenes, numbered sequentially (e.g., "Scene 1:...", "Scene 2:...").
- You MUST tell a complete, seamless story that progresses logically from beginning to end within these ${totalScenes} scenes.
- **DO NOT REPEAT** events or dialogue to fill time. Every scene must move the story forward.
- Divide the narrative arc proportionally:
  - ~20% Setup (Scenes 1-${Math.ceil(totalScenes * 0.2)})
  - ~60% Rising Action & Climax (Scenes ${Math.ceil(totalScenes * 0.2) + 1}-${Math.ceil(totalScenes * 0.8)})
  - ~20% Resolution (Scenes ${Math.ceil(totalScenes * 0.8) + 1}-${totalScenes})
- The story MUST end definitively at Scene ${totalScenes}. Do not loop or cutoff.

The characters in this script MUST be consistent with the detailed descriptions provided below.
${dialogueInstruction}

**Cast of Characters:**
${characterDescriptions}

**Output Format:**
You must output a numbered list of scenes. Use this exact format for every scene:
Scene [Number]: [Action/Description] [Dialogue if enabled]

Example:
Scene 1: Character A stands on a cliff looking at the sunset.
Scene 2: Character A turns around and sees...

The output must be ONLY the script text. Do not include any introductory phrases, summaries, or explanations.`;
    },

    systemInstruction_generateScenes: (config: VideoConfig, characterDescriptions: string, startSceneId: number, existingScenesCount: number, scenesPerBatch: number, lastScene: Scene | null) => {
        const sceneDuration = 8;
        const languageName = DIALOGUE_LANGUAGES.find(lang => lang.key === config.dialogueLanguage)?.en || config.dialogueLanguage;
        const dialogueFieldInstruction = config.includeDialogue 
            ? `This is a CRITICAL field. You MUST extract the character's dialogue for this scene from the script. The dialogue MUST be in the specified language: **${languageName} (${config.dialogueLanguage})**. If there is no dialogue in this specific scene, you MUST use an empty string "". Do not write dialogue in English unless English is the selected language.`
            : `An empty string "". Dialogue is disabled for this project.`;
        
        const endSceneId = startSceneId + scenesPerBatch - 1;

        const continuityInstruction = lastScene 
            ? `
**CRITICAL CONTINUITY INSTRUCTION (STATE TRANSITION & CROSS-ANCHORING):**
You are continuing an existing storyboard. The previous scene (Scene #${existingScenesCount}) ended with this state:
\`\`\`json
${JSON.stringify(lastScene.prompt, null, 2)}
\`\`\`
To create a seamless "one-shot" transition, you MUST follow these strict rules for the first scene you generate (Scene #${startSceneId}):

1.  **State Matching (Cross-Anchor):** The beginning of Scene #${startSceneId} MUST be an immediate and direct continuation of the end of Scene #${existingScenesCount}. The scene description, character actions, and background must explicitly acknowledge and flow from the previous scene's final state. This is a "cross-anchor" - the end of one scene is the anchor for the start of the next.

2.  **Camera & Lighting Lock:** Unless the script dictates a dramatic cut or change, you MUST start Scene #${startSceneId} with the *exact same camera shot, angle, and lighting* as the previous scene. For example, if the last scene ended with a 'medium shot' and 'soft warm ambient' lighting, the new scene MUST begin with those same values before any potential transition within the scene. This maintains visual continuity.

3.  **Motif & Timeline Consistency:** Maintain consistent motifs (recurring visual elements, sounds, character states) and timeline logic. Do not abruptly change the time of day, weather, or environment unless explicitly required by the script. The transition must be believable.`
            : `**CRITICAL CONTINUITY INSTRUCTION:** This is the first batch of scenes. You must establish the setting and characters clearly as described in the script.`;


        return `You are an AI animation director specializing in dynamic, "one-shot" style storyboarding where scenes flow seamlessly into one another. Your task is to visualize a specific batch of scenes from an animation script into detailed prompts for an image generation model.
You will receive a full script. You must ONLY process scenes #${startSceneId} through #${endSceneId} from that script.

${continuityInstruction}

**GENERAL INSTRUCTIONS:**
- **SCOPE:** Focus EXCLUSIVELY on the action and events described in **Scene ${startSceneId}** to **Scene ${endSceneId}** of the provided script.
- **LANGUAGE:** With the EXCEPTION of the \`dialogue\` field, ALL other string values in the entire JSON output MUST be written in **English**.
- **CHARACTER CONSISTENCY:** The characters MUST strictly adhere to the detailed "Cast of Characters" provided below. In the 'character_description' field, identify characters by name.
- **JSON OUTPUT:** The final output MUST be a single, perfectly formatted JSON object starting with \`{\` and ending with \`}\`, containing the 'scenes' array with exactly ${scenesPerBatch} items (or fewer if the script ends).

**Cast of Characters:**
${characterDescriptions}

Each scene object must have the following structure:
- \`scene_id\`: (Integer) The sequential number of the scene. The first scene in your response MUST have this ID: ${startSceneId}.
- \`time\`: (String) The timestamp for the scene in "MM:SS" format.
- \`prompt\`: (Object) A detailed prompt containing:
  - \`scene_description\`: (String) A one-sentence description of the key moment in **English**.
  - \`character_description\`: (String) Description of pose, action, and expression in **English**.
  - \`background_description\`: (String) Description of setting in **English**.
  - \`camera_shot\`: (String) Shot type in **English**.
  - \`lighting\`: (String) Lighting style in **English**.
  - \`color_palette\`: (String) Colors and mood in **English**.
  - \`style\`: (String) "${config.style}".
  - \`composition_notes\`: (String) Framing notes in **English**.
  - \`sound_effects\`: (String) Sound FX in **English**.
  - \`dialogue\`: (String) ${dialogueFieldInstruction}
  - \`keywords\`: (Array of Strings) 5-10 keywords in **English**.
  - \`negative_prompts\`: (Array of Strings) Things to avoid in **English**.
  - \`aspect_ratio\`: (String) "16:9".
  - \`duration_seconds\`: (Integer) ${sceneDuration}.`;
    },
};

const vi: TranslationKeys = {
    // App.tsx
    untitledProject: "Hoạt hình chưa có tên",
    generationStatusPreparing: "Đang chuẩn bị tạo các phân cảnh...",
    generationStatusRequesting: (batch: number) => `Đang yêu cầu lô phân cảnh #${batch}...`,
    generationIncompleteError: (current: number, total: number) => `Tạo chưa hoàn tất. Chỉ có ${current} trên tổng số ${total} phân cảnh được tạo. Bạn có thể thử tiếp tục.`,
    errorGeneratingImage: "Đã xảy ra lỗi khi tạo hình ảnh.",
    errorQuotaExceeded: "Đã vượt quá hạn ngạch API. Bạn đã thực hiện quá nhiều yêu cầu trong một thời gian ngắn. Vui lòng đợi một lát và thử lại. Bạn có thể theo dõi việc sử dụng tại đây: https://ai.dev/usage",
    errorInvalidApiKey: (context: string) => `API key không hợp lệ. Vui lòng kiểm tra lại key trong phần cài đặt. (Bối cảnh: ${context})`,
    errorServerOverloaded: (context: string) => `Mô hình hiện đang bận hoặc quá tải. Vui lòng thử lại sau giây lát. (Bối cảnh: ${context})`,
    errorGeneric: (context: string, message: string) => `Lỗi trong ${context}: ${message}`,
    errorUnknown: (context: string) => `Đã xảy ra lỗi không xác định trong ${context}.`,
    generationFailedCanResume: (errorMsg: string) => `Tạo phân cảnh thất bại: ${errorMsg}. Bạn có thể thử tiếp tục quá trình.`,
    errorGeneratingPrompt: "Đã xảy ra lỗi khi tạo prompt nhân vật từ hình ảnh.",
    newCharacterName: "Nhân vật",
    errorMissingApiKey: (provider: string) => `Thiếu API Key cho ${provider}. Vui lòng thêm ít nhất một key trong Cài đặt API.`,
    errorAllKeysExhausted: "Tất cả các API key bạn cung cấp đều đã hết hạn ngạch sử dụng. Vui lòng thêm key mới hoặc chờ hạn ngạch được đặt lại.",

    // Header.tsx
    appTitle: "Chuyên Hoạt Hình",
    appDescription: "Studio Hoạt hình AI",
    newProjectButton: "Dự án mới",
    guideButtonTooltip: "Hiển thị hướng dẫn",
    apiSettingsTooltip: "Cài đặt API",
    languageLabel: "Ngôn ngữ",

    // InputPanel.tsx
    characterRosterLabel: "Danh sách nhân vật",
    addCharacterButton: "Thêm nhân vật",
    importFromFileButton: "Nhập từ tệp .txt",
    characterNamePlaceholder: "Tên nhân vật...",
    characterImageLabel: "Ảnh tham chiếu",
    uploadImagePrompt: "Tải ảnh",
    generatePromptFromImageButton: "Từ Ảnh",
    generatingPromptButton: "Đang tạo...",
    suggestPromptButton: "Gợi ý Prompt",
    suggestingPromptButton: "Đang gợi ý...",
    characterPromptLabel: "Prompt DNA Nhân vật",
    characterPromptPlaceholder: "Mô tả chi tiết ngoại hình, quần áo, và phong cách của nhân vật, hoặc tạo prompt từ một hình ảnh.",
    storyIdeaLabel: "Câu chuyện / Kịch bản",
    suggestIdeaButton: "Gợi ý câu chuyện",
    onCooldownButton: "Đang chờ...",
    suggestingIdeaButton: "Đang gợi ý...",
    storyIdeaPlaceholder: "Mô tả ý tưởng hoạt hình của bạn. Hoặc, hãy tạo một prompt nhân vật, chọn phong cách câu chuyện, và nhấp 'Gợi ý câu chuyện' để AI tạo ý tưởng dựa trên nhân vật của bạn.",
    generatedScriptLabel: "Kịch bản hoạt hình đã tạo",
    videoSettingsLabel: "Cài đặt hoạt hình",
    durationLabel: "Thời lượng hoạt hình ước tính (phút)",
    durationPlaceholder: "VD: 1",
    durationFeedback: (scenes: number, mins: number, secs: number) => `~${scenes} phân cảnh (~${mins} phút ${secs} giây).`,
    videoFormatLabel: "Nhịp độ / Định dạng hoạt hình",
    styleLabel: "Phong cách hoạt hình",
    storyStyleLabel: "Phong cách câu chuyện",
    aiProviderLabel: "Nhà cung cấp AI",
    aiModelLabel: "Model AI",
    generateScriptButton: "Tạo kịch bản hoạt hình",
    generatingScriptButton: "Đang tạo kịch bản...",
    generateStoryboardButton: "Tạo bảng phân cảnh",
    generatingStoryboardButton: "Đang tạo bảng phân cảnh...",
    continueGenerateStoryboardButton: "Tiếp tục tạo Bảng phân cảnh",
    includeDialogueLabel: "Bao gồm Lời thoại",
    dialogueLanguageLabel: "Ngôn ngữ Lời thoại",

    // SceneTimeline.tsx
    timelineTitle: "Dòng thời gian phân cảnh",
    downloadButton: "Tải xuống prompt",
    primaryReferenceLabel: "Tham chiếu chính để tạo hàng loạt",
    selectPrimaryReferencePrompt: "Chọn tham chiếu nhân vật/phong cách",
    generateAllImagesButton: "Tạo tất cả ảnh còn thiếu",
    generatingAllImagesButton: "Đang tạo tất cả...",
    downloadAllImagesButton: "Tải tất cả ảnh (.zip)",
    emptyTimelineTitle: "Bảng phân cảnh của bạn trống.",
    emptyTimelineDescription: "Cung cấp một câu chuyện và tạo kịch bản để xem các phân cảnh hoạt hình của bạn ở đây.",

    // SceneCard.tsx
    sceneLabel: "Phân cảnh",
    timeLabel: "Thời gian",
    promptLabel: "Prompt (JSON)",
    promptHelperTooltip: "Bật/tắt trợ giúp prompt",
    invalidJsonError: "Định dạng JSON không hợp lệ. Vui lòng sửa lại.",
    sceneImageLabel: "Hình ảnh phân cảnh đã tạo",
    selectReferenceImageLabel: "Ảnh tham chiếu",
    noImageGenerated: "Chưa có ảnh nào được tạo.",
    generateImageButton: "Tạo ảnh",
    generatingImageButton: "Đang tạo...",
    noReferenceImagesAvailable: "Không có ảnh tham chiếu",

    // Loader.tsx
    generationComplete: "Tạo hoàn tất!",
    generatingScene: (current: number, total: number) => `Đang tạo phân cảnh ${current} trên ${total}`,
    loaderText: "Đang tạo...",

    // ConfirmationModal.tsx & App.tsx
    newProjectConfirmationTitle: "Bắt đầu dự án mới?",
    newProjectConfirmationMessage: "Hành động này sẽ xóa tất cả tham chiếu nhân vật, câu chuyện và phân cảnh. Bạn có chắc chắn muốn tiếp tục không?",
    confirmButton: "Xác nhận",
    cancelButton: "Hủy bỏ",
    resumeGenerationTitle: "Hành động bị tạm dừng",
    resumeButton: "Tiếp tục",
    finishForNowButton: "Để sau",

    // GuideModal.tsx
    guideModalTitle: "Hướng dẫn sử dụng Chuyên Hoạt Hình",
     guideSteps: [
        { title: "Bước 1: Cài đặt API Keys", description: "Trước khi bắt đầu, hãy nhấp vào biểu tượng bánh răng ở đầu trang để mở Cài đặt API. Thêm một hoặc nhiều Google AI API key. Nếu bạn cung cấp nhiều key, ứng dụng sẽ tự động chuyển sang key tiếp theo nếu key hiện tại hết hạn ngạch." },
        { title: "Bước 2: Xác định các Nhân vật", description: "Đây là bước quan trọng nhất. Thêm một hoặc nhiều nhân vật. Với mỗi nhân vật, cung cấp tên và mô tả chi tiết trong ô 'prompt' của họ. Bạn cũng có thể tải lên ảnh tham chiếu và nhấp vào 'Tạo Prompt từ Ảnh' để AI tạo mô tả cho bạn. Danh sách này đảm bảo tính nhất quán." },
        { title: "Bước 3 (Tùy chọn): Nhập Nhân vật", description: "Để làm việc nhanh hơn, bạn có thể nhập nhiều nhân vật từ tệp `.txt`. Nhấp vào 'Nhập từ tệp .txt' và chọn tệp của bạn. Tệp phải tuân theo định dạng sau (dòng trống giữa các nhân vật là quan trọng):\n[Character Name: Nhân vật A]\nMô tả chi tiết về Nhân vật A...\n\n[Character Name: Nhân vật B]\nMô tả chi tiết về Nhân vật B..." },
        { title: "Bước 4: Lấy ý tưởng câu chuyện", description: "Khi các nhân vật của bạn đã sẵn sàng, hãy chọn 'Phong cách câu chuyện' và nhấp vào 'Gợi ý câu chuyện'. AI sẽ tạo ra một ý tưởng câu chuyện có sự tham gia của các nhân vật của bạn theo thể loại đó. Bạn có thể chỉnh sửa câu chuyện này tùy thích." },
        { title: "Bước 5: Cài đặt Hoạt hình", description: "Chọn một phong cách hoạt hình và thời lượng mong muốn cho video của bạn. Điều này thiết lập giao diện và cảm nhận tổng thể." },
        { title: "Bước 6: Tạo Kịch bản Hoạt hình", description: "Nhấp vào 'Tạo kịch bản hoạt hình'. AI sẽ mở rộng câu chuyện của bạn thành một kịch bản có cấu trúc, ghi nhớ mô tả các nhân vật của bạn." },
        { title: "Bước 7: Tạo Bảng phân cảnh", description: "Khi kịch bản đã sẵn sàng, hãy nhấp vào 'Tạo Bảng phân cảnh'. Quá trình sẽ diễn ra theo từng lô để đảm bảo ổn định. Bạn có thể tiếp tục tạo cho đến khi hoàn tất." },
        { title: "Bước 8: Tạo Hình ảnh Phân cảnh", description: "Trong dòng thời gian, sử dụng hình ảnh tham chiếu bạn đã tải lên để tạo hình ảnh cho mỗi cảnh, có thể tạo từng cái một hoặc tất cả cùng một lúc." },
        { title: "Bước 9: Xem lại và Xuất", description: "Xem lại bảng phân cảnh của bạn. Bạn có thể chỉnh sửa các prompt JSON cho bất kỳ cảnh nào để tinh chỉnh hình ảnh, sau đó tải xuống tất cả các prompt hoặc hình ảnh." },
    ],
    guideProTipsTitle: "Mẹo chuyên nghiệp",
    guideProTips: [
        { title: "Prompt nhân vật chi tiết là chìa khóa", description: "Bạn càng cung cấp nhiều chi tiết cho mỗi nhân vật (quần áo, tóc, đặc điểm riêng, phong cách), kết quả của bạn sẽ càng nhất quán." },
        { title: "Sử dụng ảnh tham chiếu rõ nét", description: "Để tạo prompt và tạo hình ảnh, hãy sử dụng hình ảnh rõ ràng, chính diện của nhân vật để có kết quả tốt nhất." },
        { title: "Dùng tên nhân vật trong kịch bản", description: "Khi chỉnh sửa kịch bản hoặc ý tưởng câu chuyện, hãy sử dụng tên nhân vật bạn đã xác định. AI sẽ sử dụng chúng để tạo mô tả cảnh chính xác hơn." },
        { title: "Lặp lại và Tinh chỉnh", description: "Các prompt JSON được tạo ra hoàn toàn có thể chỉnh sửa. Thay đổi biểu cảm của nhân vật, góc máy hoặc ánh sáng để hoàn thiện tầm nhìn của bạn." },
    ],
    
    promptHelperTitle: "Thêm nhanh Prompt",
    promptHelperTags: {
        camera_shots: {
            group: "Các loại góc quay",
            tags: [
                { tag: "establishing shot", desc: "Góc siêu rộng để giới thiệu bối cảnh và thiết lập phân cảnh." },
                { tag: "full shot", desc: "Hiển thị toàn bộ nhân vật từ đầu đến chân." },
                { tag: "medium shot", desc: "Khung hình từ thắt lưng trở lên, tốt cho hội thoại." },
                { tag: "close-up shot", desc: "Tập trung vào khuôn mặt để thể hiện cảm xúc." },
                { tag: "extreme close-up", desc: "Tập trung vào một chi tiết nhỏ, như mắt hoặc đồ vật." },
                { tag: "over-the-shoulder shot", desc: "Góc nhìn từ phía sau vai một nhân vật nhìn sang nhân vật khác." },
                { tag: "insert shot", desc: "Cận cảnh một vật thể hoặc chi tiết cụ thể liên quan đến câu chuyện." },
                { tag: "POV shot", desc: "Góc nhìn từ quan điểm của nhân vật." },
            ],
        },
        camera_angles: {
            group: "Góc đặt máy quay",
            tags: [
                { tag: "eye-level angle", desc: "Góc nhìn trung tính và tiêu chuẩn." },
                { tag: "low-angle", desc: "Làm cho chủ thể trông quyền lực hoặc uy nghi." },
                { tag: "high-angle", desc: "Làm cho chủ thể trông nhỏ bé hoặc yếu đuối." },
                { tag: "dutch angle / tilted shot", desc: "Máy quay nghiêng, tạo cảm giác căng thẳng, bất ổn hoặc phiêu lưu." },
                { tag: "bird's-eye view / top-down shot", desc: "Nhìn thẳng từ trên cao xuống, cho thấy bố cục toàn cảnh." },
            ],
        },
        camera_movement: {
            group: "Chuyển động máy quay",
            tags: [
                { tag: "static shot", desc: "Máy quay hoàn toàn đứng yên." },
                { tag: "pan shot (left/right)", desc: "Máy quay xoay ngang trên một trục cố định." },
                { tag: "tilt shot (up/down)", desc: "Máy quay xoay dọc trên một trục cố định." },
                { tag: "tracking shot / dolly shot", desc: "Máy quay di chuyển theo nhân vật." },
                { tag: "crane shot / pedestal shot", desc: "Máy quay di chuyển lên hoặc xuống theo phương thẳng đứng." },
                { tag: "dolly zoom / vertigo effect", desc: "Phóng to/thu nhỏ trong khi di chuyển máy quay theo hướng ngược lại." },
                { tag: "orbit shot / 360-degree shot", desc: "Máy quay di chuyển vòng quanh chủ thể." },
            ]
        },
        action_shots: {
            group: "Kỹ thuật quay hành động",
            tags: [
              { tag: "slow motion", desc: "Làm chậm hành động để nhấn mạnh chuyển động hoặc tác động." },
              { tag: "hand tracking shot", desc: "Máy quay theo dõi chuyển động của bàn tay nhân vật." },
              { tag: "foot tracking shot", desc: "Máy quay theo dõi chuyển động của bàn chân nhân vật." },
              { tag: "object tracking shot", desc: "Máy quay theo dõi một vật thể đang di chuyển, như vũ khí hoặc xe cộ." },
              { tag: "motion-follow camera", desc: "Máy quay di chuyển với tốc độ tương tự nhân vật, thường dùng trong các cảnh chiến đấu." },
              { tag: "hero shot", desc: "Chuyển động chậm kết hợp với máy quay theo dõi để làm cho nhân vật trông hoành tráng." },
              { tag: "insert action shot", desc: "Cận cảnh một chi tiết hành động cụ thể, như bàn tay nắm chặt thanh kiếm." },
            ]
        },
        styles: {
            group: "Phong cách nghệ thuật",
            tags: [
                { tag: "cel-shaded", desc: "Phong cách anime 2D cổ điển." },
                { tag: "watercolor", desc: "Phông nền mềm mại, giống tranh màu nước." },
                { tag: "painterly", desc: "Trông giống như một bức tranh sơn dầu." },
                { tag: "concept art", desc: "Phong cách minh họa, chi tiết." },
                { tag: "vector art", desc: "Nét vẽ sạch sẽ, màu sắc đơn khối." },
                { tag: "8-bit pixel art", desc: "Phong cách game retro." },
            ],
        },
        lighting: {
            group: "Ánh sáng",
            tags: [
                { tag: "cinematic lighting", desc: "Kịch tính, độ tương phản cao." },
                { tag: "soft, ambient light", desc: "Ánh sáng đều, dịu nhẹ." },
                { tag: "dramatic rim lighting", desc: "Làm nổi bật các cạnh của nhân vật." },
                { tag: "golden hour light", desc: "Ánh sáng ấm áp, huyền ảo." },
                { tag: "neon lighting", desc: "Sống động, cảm giác cyberpunk." },
            ],
        }
    },

    // StorySuggestionModal.tsx
    storySuggestionModalTitle: "Gợi ý câu chuyện từ AI",
    useThisIdeaButton: "Dùng ý tưởng này",
    regenerateIdeaButton: "Tạo lại",
    closeButton: "Đóng",
    suggestionLoadingText: "AI đang sáng tạo câu chuyện cho nhân vật của bạn...",
    storySuggestionEditHint: "Bạn có thể chỉnh sửa văn bản dưới đây trước khi chấp nhận.",

    // ContinueGenerationModal.tsx
    continueGenerationTitle: "Hoàn thành lô phân cảnh",
    continueGenerationMessage: (generated: number, total: number) => `Đã tạo thành công ${generated} trên tổng số ${total} phân cảnh. Bạn có muốn tạo lô tiếp theo không?`,
    continueGenerationButton: "Tiếp tục tạo",
    
    // CharacterSuggestionModal.tsx
    characterSuggestionModalTitle: "Gợi ý Prompt Nhân vật",
    characterSuggestionLoadingText: "AI đang tưởng tượng các phiên bản nhân vật...",
    useThisVariationButton: "Dùng phiên bản này",

    // ApiKeyModal.tsx
    apiKeyModalTitle: "Cài đặt API Key",
    googleApiKeysLabel: "Google API Keys (Mỗi key một dòng)",
    apiKeyInputPlaceholder: "Nhập các API key của bạn, mỗi key trên một dòng...",
    apiKeyInstructions: "Ứng dụng sẽ tự động chuyển sang key tiếp theo nếu key hiện tại hết hạn ngạch.",
    saveKeysButton: "Lưu Keys",

    // Gemini Service System Instructions (VI - points to EN version)
    systemInstruction_generateCharacterPrompt: en.systemInstruction_generateCharacterPrompt,
    systemInstruction_generateCharacterVariations: en.systemInstruction_generateCharacterVariations,
    systemInstruction_generateStoryIdea: en.systemInstruction_generateStoryIdea,
    systemInstruction_generateScript: en.systemInstruction_generateScript,
    systemInstruction_generateScenes: en.systemInstruction_generateScenes,
};

export const translations = {
  en,
  vi,
};

export type TranslationKeys = typeof en;