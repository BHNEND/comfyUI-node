from .qwen3_tts import (
    Qwen3TTSCustomVoice,
    Qwen3TTSCreateVoicePrompt,
    Qwen3TTSModelLoader,
    Qwen3TTSVoiceClone,
    Qwen3TTSVoiceCloneWithPrompt,
    Qwen3TTSVoiceDesign,
)
from .size import ResolutionByAspectRatio
from .url import UrlToLocalPath

NODE_CLASS_MAPPINGS = {
    "LKLKAI_ResolutionByAspectRatio": ResolutionByAspectRatio,
    "LKLKAI_UrlToLocalPath": UrlToLocalPath,
    "Qwen3TTSModelLoader": Qwen3TTSModelLoader,
    "Qwen3TTSVoiceClone": Qwen3TTSVoiceClone,
    "Qwen3TTSCreateVoicePrompt": Qwen3TTSCreateVoicePrompt,
    "Qwen3TTSVoiceCloneWithPrompt": Qwen3TTSVoiceCloneWithPrompt,
    "Qwen3TTSCustomVoice": Qwen3TTSCustomVoice,
    "Qwen3TTSVoiceDesign": Qwen3TTSVoiceDesign,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LKLKAI_ResolutionByAspectRatio": "LK-尺寸转换",
    "LKLKAI_UrlToLocalPath": "LK-URL 2 File",
    "Qwen3TTSModelLoader": "LK Qwen3 TTS 模型加载",
    "Qwen3TTSVoiceClone": "LK Qwen3 TTS 语音克隆",
    "Qwen3TTSCreateVoicePrompt": "LK Qwen3 TTS 创建角色预设",
    "Qwen3TTSVoiceCloneWithPrompt": "LK Qwen3 TTS 角色预设克隆",
    "Qwen3TTSCustomVoice": "LK Qwen3 TTS 预设音色",
    "Qwen3TTSVoiceDesign": "LK Qwen3 TTS 声音设计",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
