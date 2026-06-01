import gc
import os
import time
from dataclasses import dataclass

try:
    import folder_paths
except ImportError:
    folder_paths = None


MODEL_NAMES = [
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
]

DEVICES = ["cuda", "cuda:0", "cuda:1", "cpu"]
DTYPES = ["fp16", "bf16", "fp32"]
LANGUAGES = [
    "自动",
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
]
SPEAKERS = [
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee",
]


@dataclass
class Qwen3TTSModelBundle:
    model: object
    model_name: str
    device: str
    dtype: str
    keep_loaded: bool


@dataclass
class Qwen3TTSVoicePrompt:
    prompt: object
    reference_text: str


class Qwen3TTSModelLoader:
    _cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "模型名称": (MODEL_NAMES,),
                "运行设备": (DEVICES,),
                "精度": (DTYPES,),
            },
            "optional": {
                "保持模型常驻": ("BOOLEAN", {"default": True}),
                "使用FlashAttention": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("QWEN3_TTS_MODEL",)
    RETURN_NAMES = ("模型",)
    FUNCTION = "load_model"
    CATEGORY = "LKLKAI Nodes/Qwen3 TTS"

    def load_model(self, **kwargs):
        model_name = kwargs["模型名称"]
        device = _normalize_device(kwargs["运行设备"])
        dtype_name = kwargs["精度"]
        keep_loaded = bool(kwargs.get("保持模型常驻", True))
        use_flash_attention = bool(kwargs.get("使用FlashAttention", True))

        torch, qwen_model_cls = _load_qwen_dependencies()
        dtype = _torch_dtype(torch, dtype_name)
        model_path = _resolve_model_path(model_name)
        cache_key = (model_path, device, dtype_name, use_flash_attention)

        if keep_loaded and cache_key in self._cache:
            bundle = self._cache[cache_key]
            _ensure_model_on_device(bundle.model, bundle.device)
            return (bundle,)

        if not keep_loaded and cache_key in self._cache:
            del self._cache[cache_key]
            _clear_memory()

        load_kwargs = {
            "device_map": device,
            "dtype": dtype,
            "attn_implementation": "flash_attention_2" if use_flash_attention else "eager",
        }

        print(f"[LKLKAI Qwen3 TTS] Loading model: {model_path}")
        model = qwen_model_cls.from_pretrained(model_path, **load_kwargs)
        bundle = Qwen3TTSModelBundle(
            model=model,
            model_name=model_name,
            device=device,
            dtype=dtype_name,
            keep_loaded=keep_loaded,
        )

        if keep_loaded:
            self._cache[cache_key] = bundle

        return (bundle,)


class Qwen3TTSVoiceClone:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "模型": ("QWEN3_TTS_MODEL",),
                "参考音频": ("AUDIO",),
                "文本": ("STRING", {"default": "", "multiline": True}),
                "参考文本": ("STRING", {"default": "", "multiline": True}),
                "语言": (LANGUAGES,),
                "自动卸载模型": ("BOOLEAN", {"default": False}),
                "最大生成Token数": ("INT", {"default": 2048, "min": 1, "max": 8192}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "语速": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05}),
                "批量模式": ("BOOLEAN", {"default": False}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 500}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.01, "max": 2.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.1, "max": 3.0, "step": 0.01}),
                "启用高级采样配置": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "仅使用声纹": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("AUDIO", "QWEN3_TTS_VOICE_PROMPT")
    RETURN_NAMES = ("音频", "角色预设")
    FUNCTION = "generate"
    CATEGORY = "LKLKAI Nodes/Qwen3 TTS"

    def generate(self, **kwargs):
        bundle = _require_bundle(kwargs["模型"])
        model = bundle.model
        _set_seed(kwargs["seed"])
        _ensure_model_on_device(model, bundle.device)

        text = kwargs["文本"].strip()
        ref_text = kwargs["参考文本"].strip()
        if not text:
            raise ValueError("文本 is required.")

        x_vector_only = bool(kwargs.get("仅使用声纹", False))
        if not x_vector_only and not ref_text:
            raise ValueError("参考文本 is required unless 仅使用声纹 is enabled.")

        generation_kwargs = _generation_kwargs(kwargs)
        ref_audio = _comfy_audio_to_qwen_audio(kwargs["参考音频"])
        language = _normalize_language(kwargs["语言"])

        start = time.time()
        wavs, sample_rate = model.generate_voice_clone(
            text=_split_batch_text(text) if kwargs["批量模式"] else text,
            language=language,
            ref_audio=ref_audio,
            ref_text=None if x_vector_only else ref_text,
            x_vector_only_mode=x_vector_only,
            **generation_kwargs,
        )
        audio = _wavs_to_comfy_audio(wavs, sample_rate, speed=kwargs["语速"])
        prompt = model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=None if x_vector_only else ref_text,
            x_vector_only_mode=x_vector_only,
        )
        _log_generation("VoiceClone", wavs, sample_rate, start)

        if kwargs["自动卸载模型"] or not bundle.keep_loaded:
            _offload_model(model)

        return (audio, Qwen3TTSVoicePrompt(prompt=prompt, reference_text=ref_text))


class Qwen3TTSCreateVoicePrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "模型": ("QWEN3_TTS_MODEL",),
                "参考音频": ("AUDIO",),
                "参考文本": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "仅使用声纹": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("QWEN3_TTS_VOICE_PROMPT",)
    RETURN_NAMES = ("角色预设",)
    FUNCTION = "create"
    CATEGORY = "LKLKAI Nodes/Qwen3 TTS"

    def create(self, **kwargs):
        bundle = _require_bundle(kwargs["模型"])
        model = bundle.model
        ref_text = kwargs["参考文本"].strip()
        x_vector_only = bool(kwargs.get("仅使用声纹", False))
        if not x_vector_only and not ref_text:
            raise ValueError("参考文本 is required unless 仅使用声纹 is enabled.")

        prompt = model.create_voice_clone_prompt(
            ref_audio=_comfy_audio_to_qwen_audio(kwargs["参考音频"]),
            ref_text=None if x_vector_only else ref_text,
            x_vector_only_mode=x_vector_only,
        )
        return (Qwen3TTSVoicePrompt(prompt=prompt, reference_text=ref_text),)


class Qwen3TTSVoiceCloneWithPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "模型": ("QWEN3_TTS_MODEL",),
                "角色预设": ("QWEN3_TTS_VOICE_PROMPT",),
                "文本": ("STRING", {"default": "", "multiline": True}),
                "语言": (LANGUAGES,),
                "最大生成Token数": ("INT", {"default": 2048, "min": 1, "max": 8192}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "语速": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 500}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.01, "max": 2.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.1, "max": 3.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("音频",)
    FUNCTION = "generate"
    CATEGORY = "LKLKAI Nodes/Qwen3 TTS"

    def generate(self, **kwargs):
        bundle = _require_bundle(kwargs["模型"])
        prompt = _require_voice_prompt(kwargs["角色预设"])
        _set_seed(kwargs["seed"])
        _ensure_model_on_device(bundle.model, bundle.device)

        text = kwargs["文本"].strip()
        if not text:
            raise ValueError("文本 is required.")

        start = time.time()
        wavs, sample_rate = bundle.model.generate_voice_clone(
            text=text,
            language=_normalize_language(kwargs["语言"]),
            voice_clone_prompt=prompt.prompt,
            **_generation_kwargs(kwargs),
        )
        _log_generation("VoiceCloneWithPrompt", wavs, sample_rate, start)
        return (_wavs_to_comfy_audio(wavs, sample_rate, speed=kwargs["语速"]),)


class Qwen3TTSCustomVoice:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "模型": ("QWEN3_TTS_MODEL",),
                "文本": ("STRING", {"default": "", "multiline": True}),
                "说话人": (SPEAKERS,),
                "语言": (LANGUAGES,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "风格指令": ("STRING", {"default": "", "multiline": True}),
                "最大生成Token数": ("INT", {"default": 2048, "min": 1, "max": 8192}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 500}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.01, "max": 2.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.1, "max": 3.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("音频",)
    FUNCTION = "generate"
    CATEGORY = "LKLKAI Nodes/Qwen3 TTS"

    def generate(self, **kwargs):
        bundle = _require_bundle(kwargs["模型"])
        _set_seed(kwargs["seed"])
        _ensure_model_on_device(bundle.model, bundle.device)

        start = time.time()
        wavs, sample_rate = bundle.model.generate_custom_voice(
            text=kwargs["文本"].strip(),
            language=_normalize_language(kwargs["语言"]),
            speaker=kwargs["说话人"],
            instruct=(kwargs.get("风格指令") or "").strip() or None,
            **_generation_kwargs(kwargs),
        )
        _log_generation("CustomVoice", wavs, sample_rate, start)
        return (_wavs_to_comfy_audio(wavs, sample_rate),)


class Qwen3TTSVoiceDesign:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "模型": ("QWEN3_TTS_MODEL",),
                "文本": ("STRING", {"default": "", "multiline": True}),
                "声音描述": ("STRING", {"default": "", "multiline": True}),
                "语言": (LANGUAGES,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "最大生成Token数": ("INT", {"default": 2048, "min": 1, "max": 8192}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 500}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.01, "max": 2.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.1, "max": 3.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("音频",)
    FUNCTION = "generate"
    CATEGORY = "LKLKAI Nodes/Qwen3 TTS"

    def generate(self, **kwargs):
        bundle = _require_bundle(kwargs["模型"])
        _set_seed(kwargs["seed"])
        _ensure_model_on_device(bundle.model, bundle.device)

        start = time.time()
        wavs, sample_rate = bundle.model.generate_voice_design(
            text=kwargs["文本"].strip(),
            language=_normalize_language(kwargs["语言"]),
            instruct=kwargs["声音描述"].strip(),
            **_generation_kwargs(kwargs),
        )
        _log_generation("VoiceDesign", wavs, sample_rate, start)
        return (_wavs_to_comfy_audio(wavs, sample_rate),)


def _load_qwen_dependencies():
    try:
        import torch
    except ImportError as error:
        raise ImportError("Qwen3 TTS nodes require torch in the ComfyUI environment.") from error

    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError as error:
        raise ImportError("Qwen3 TTS nodes require qwen-tts. Install with: pip install qwen-tts") from error

    return torch, Qwen3TTSModel


def _resolve_model_path(model_name):
    model_name = model_name.strip()
    if not model_name:
        raise ValueError("模型名称 is required.")

    if os.path.isdir(model_name):
        return model_name

    if folder_paths is None:
        return model_name

    local_path = os.path.join(folder_paths.models_dir, "TTS", model_name.split("/")[-1])
    if os.path.isdir(local_path) and os.listdir(local_path):
        return local_path

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return model_name

    os.makedirs(local_path, exist_ok=True)
    print(f"[LKLKAI Qwen3 TTS] Downloading {model_name} to {local_path}")
    snapshot_download(repo_id=model_name, local_dir=local_path)
    return local_path


def _torch_dtype(torch, dtype_name):
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    return mapping[dtype_name]


def _normalize_device(device):
    if device == "cuda":
        return "cuda:0"
    return device


def _normalize_language(language):
    return "Auto" if language == "自动" else language


def _require_bundle(value):
    if not isinstance(value, Qwen3TTSModelBundle):
        raise TypeError("模型 must be a QWEN3_TTS_MODEL produced by Qwen3TTSModelLoader.")
    return value


def _require_voice_prompt(value):
    if not isinstance(value, Qwen3TTSVoicePrompt):
        raise TypeError("角色预设 must be produced by Qwen3TTSCreateVoicePrompt or Qwen3TTSVoiceClone.")
    return value


def _set_seed(seed):
    torch, _ = _load_qwen_dependencies()
    try:
        import numpy as np
    except ImportError as error:
        raise ImportError("Qwen3 TTS nodes require numpy.") from error

    seed = int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))


def _generation_kwargs(kwargs):
    max_tokens = kwargs.get("最大生成Token数", kwargs.get("max_new_tokens", 2048))
    top_p = kwargs.get("top_p", 1.0)
    top_k = kwargs.get("top_k", 50)
    temperature = kwargs.get("temperature", 0.9)
    repetition_penalty = kwargs.get("repetition_penalty", 1.05)

    return {
        "max_new_tokens": int(max_tokens),
        "do_sample": True,
        "top_p": float(top_p),
        "top_k": int(top_k),
        "temperature": float(temperature),
        "repetition_penalty": float(repetition_penalty),
        "subtalker_dosample": True,
        "subtalker_top_p": float(top_p),
        "subtalker_top_k": int(top_k),
        "subtalker_temperature": float(temperature),
    }


def _comfy_audio_to_qwen_audio(audio):
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("参考音频 must be a ComfyUI AUDIO object.")

    try:
        import numpy as np
    except ImportError as error:
        raise ImportError("Qwen3 TTS nodes require numpy.") from error

    waveform = audio["waveform"]
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu()
    waveform = np.asarray(waveform, dtype=np.float32)

    while waveform.ndim > 1:
        waveform = waveform[0] if waveform.shape[0] == 1 else waveform.mean(axis=0)

    return waveform.astype(np.float32), int(audio["sample_rate"])


def _wavs_to_comfy_audio(wavs, sample_rate, speed=1.0):
    torch, _ = _load_qwen_dependencies()
    try:
        import numpy as np
    except ImportError as error:
        raise ImportError("Qwen3 TTS nodes require numpy.") from error

    if not wavs:
        raise ValueError("Qwen3 TTS returned no audio.")

    waveform = np.asarray(wavs[0], dtype=np.float32)
    if float(speed) != 1.0:
        waveform = _change_speed(waveform, float(speed))

    tensor = torch.from_numpy(waveform).float()
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)

    return {"waveform": tensor, "sample_rate": int(sample_rate)}


def _change_speed(waveform, speed):
    if speed <= 0:
        raise ValueError("语速 must be greater than 0.")

    if abs(speed - 1.0) < 0.0001:
        return waveform

    try:
        import numpy as np
    except ImportError as error:
        raise ImportError("Qwen3 TTS nodes require numpy.") from error

    target_len = max(1, int(round(len(waveform) / speed)))
    x_old = np.linspace(0.0, 1.0, num=len(waveform), endpoint=True)
    x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=True)
    return np.interp(x_new, x_old, waveform).astype(np.float32)


def _split_batch_text(text):
    parts = [part.strip() for part in text.split("\n") if part.strip()]
    if not parts:
        raise ValueError("批量模式 requires at least one non-empty text line.")
    return parts


def _ensure_model_on_device(model, device):
    try:
        if hasattr(model, "model") and hasattr(model.model, "to"):
            model.model.to(device)
    except Exception as error:
        raise RuntimeError(f"Failed to move Qwen3 TTS model to {device}: {error}") from error


def _offload_model(model):
    try:
        if hasattr(model, "model") and hasattr(model.model, "to"):
            model.model.to("cpu")
    finally:
        _clear_memory()


def _clear_memory():
    gc.collect()
    try:
        import torch
    except ImportError:
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _log_generation(label, wavs, sample_rate, start):
    duration = sum(len(wav) for wav in wavs) / max(int(sample_rate), 1)
    elapsed = time.time() - start
    print(f"[LKLKAI Qwen3 TTS] {label}: {duration:.2f}s audio generated in {elapsed:.2f}s")

