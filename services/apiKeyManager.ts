const API_KEYS_STORAGE_KEY = 'chuyen-hoat-hinh-google-api-keys';
const CURRENT_KEY_INDEX_KEY = 'chuyen-hoat-hinh-current-key-index';

let keys: string[] = [];
let currentIndex = 0;

// Load keys and index from localStorage on module initialization
try {
    const storedKeys = localStorage.getItem(API_KEYS_STORAGE_KEY);
    if (storedKeys) {
        const parsedKeys = JSON.parse(storedKeys);
        if (Array.isArray(parsedKeys)) {
            keys = parsedKeys;
        }
    }
    const storedIndex = localStorage.getItem(CURRENT_KEY_INDEX_KEY);
    if (storedIndex) {
        const parsedIndex = parseInt(storedIndex, 10);
        if (!isNaN(parsedIndex) && parsedIndex < keys.length) {
            currentIndex = parsedIndex;
        }
    }
} catch (error) {
    console.error("Failed to load API keys from localStorage:", error);
    keys = [];
    currentIndex = 0;
}

export const getKeys = (): string[] => {
    return keys;
};

export const saveKeys = (newKeys: string[]): void => {
    keys = newKeys.map(k => k.trim()).filter(k => k.length > 0);
    currentIndex = 0;
    try {
        localStorage.setItem(API_KEYS_STORAGE_KEY, JSON.stringify(keys));
        localStorage.setItem(CURRENT_KEY_INDEX_KEY, '0');
    } catch (error) {
        console.error("Failed to save API keys to localStorage:", error);
    }
};

export const getCurrentKey = (): string | undefined => {
    if (keys.length === 0) {
        return undefined;
    }
    // Ensure index is always valid
    if (currentIndex >= keys.length) {
        currentIndex = 0;
    }
    return keys[currentIndex];
};

export const moveToNextKey = (): void => {
    if (keys.length === 0) {
        return;
    }
    currentIndex = (currentIndex + 1) % keys.length;
     try {
        localStorage.setItem(CURRENT_KEY_INDEX_KEY, String(currentIndex));
    } catch (error) {
        console.error("Failed to save current key index to localStorage:", error);
    }
};

export const getKeyCount = (): number => {
    return keys.length;
};

export const getCurrentIndex = (): number => {
    return currentIndex;
};
