export const VIDEO_STYLES = [
  { key: 'ghibli', en: 'Ghibli Inspired', vi: 'Phong cách Ghibli' },
  { key: 'pixar', en: 'Pixar Style 3D', vi: 'Phong cách 3D Pixar' },
  { key: 'disney_2d', en: 'Classic Disney 2D', vi: 'Hoạt hình 2D Disney cổ điển' },
  { key: 'anime_shonen', en: 'Japanese Anime (Shonen)', vi: 'Anime Nhật Bản (Shonen)' },
  { key: 'lofi_anime', en: 'Lo-fi Anime', vi: 'Anime Lo-fi' },
  { key: 'claymation', en: 'Claymation / Stop Motion', vi: 'Hoạt hình đất sét / Stop Motion' },
  { key: 'rubber_hose', en: '1930s Rubber Hose', vi: 'Hoạt hình ống cao su 1930s' },
  { key: 'chibi', en: 'Chibi Style', vi: 'Phong cách Chibi' },
];

export const VIDEO_FORMATS = [
  { key: 'action_scene', en: 'Action Scene (Fast-paced)', vi: 'Cảnh hành động (Nhịp độ nhanh)' },
  { key: 'narrative_short', en: 'Narrative Short (Standard pace)', vi: 'Phim ngắn tự sự (Nhịp độ vừa)' },
  { key: 'emotional_moment', en: 'Emotional Moment (Slow pace)', vi: 'Khoảnh khắc cảm xúc (Nhịp độ chậm)' },
];

export const STORY_STYLES = [
  { key: 'adventure', en: 'Adventure', vi: 'Phiêu lưu' },
  { key: 'comedy', en: 'Comedy', vi: 'Hài hước' },
  { key: 'mystery', en: 'Mystery', vi: 'Bí ẩn' },
  { key: 'slice_of_life', en: 'Slice of Life', vi: 'Đời thường' },
  { key: 'fantasy', en: 'Fantasy', vi: 'Giả tưởng' },
  { key: 'sci_fi', en: 'Sci-Fi', vi: 'Khoa học viễn tưởng' },
  { key: 'drama', en: 'Drama', vi: 'Kịch tính' },
];

export const DIALOGUE_LANGUAGES = [
  { key: 'vi', en: 'Vietnamese', vi: 'Tiếng Việt' },
  { key: "en", en: "English", vi: "Tiếng Anh" },
  { key: "ja", en: "Japanese", vi: "Tiếng Nhật" },
  { key: "ko", en: "Korean", vi: "Tiếng Hàn" },
  { key: "fr", en: "French", vi: "Tiếng Pháp" },
  { key: "es", en: "Spanish", vi: "Tiếng Tây Ban Nha" },
];

export const API_PROVIDERS = [
  {
    key: 'proxy',
    name: 'LLM (Proxy)',
    models: [
      { key: 'gemini:gemini-2.5-pro-preview-06-05', name: 'Gemini 2.5 Pro (Preview 06-05)' },
      { key: 'openai:gpt-4o-mini', name: 'GPT-4o Mini' },
    ],
  },
];