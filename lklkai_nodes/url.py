import hashlib
import mimetypes
import os
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

try:
    import folder_paths
except ImportError:
    folder_paths = None


DEFAULT_SUBFOLDER = "lklkai_url"
DEFAULT_PREFIX = "url_asset"
USER_AGENT = "lklkai_nodes/1.0"
OUTPUT_TYPES = ("IMAGE", "VIDEO", "AUDIO")


class UrlToLocalPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "", "multiline": False}),
                "output_type": (list(OUTPUT_TYPES),),
                "filename_prefix": ("STRING", {"default": DEFAULT_PREFIX, "multiline": False}),
                "overwrite": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "VIDEO", "AUDIO", "STRING")
    RETURN_NAMES = ("image", "video", "audio", "local_path")
    FUNCTION = "run"
    CATEGORY = "LKLKAI Nodes"

    def run(self, url, output_type, filename_prefix=DEFAULT_PREFIX, overwrite=False):
        local_path = download_url_to_local_path(url, filename_prefix, overwrite)
        image = None
        video = None
        audio = None

        if output_type == "IMAGE":
            image = load_image(local_path)
        elif output_type == "VIDEO":
            video = local_path
        elif output_type == "AUDIO":
            audio = load_audio(local_path)
        else:
            raise ValueError("output_type must be IMAGE, VIDEO, or AUDIO.")

        return (image, video, audio, local_path)


def download_url_to_local_path(url, filename_prefix=DEFAULT_PREFIX, overwrite=False):
    parsed = _validate_url(url)
    output_dir = _get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        content_type = response.headers.get_content_type()
        filename = _build_filename(url, parsed, content_type, filename_prefix)
        local_path = output_dir / filename

        if local_path.exists() and not overwrite:
            return str(local_path)

        with local_path.open("wb") as file:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                file.write(chunk)

    return str(local_path)


def _validate_url(url):
    normalized_url = url.strip()
    if not normalized_url:
        raise ValueError("url is required.")

    parsed = urlparse(normalized_url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("url must start with http:// or https://.")

    if not parsed.netloc:
        raise ValueError("url must include a host.")

    return parsed


def _get_output_dir():
    if folder_paths is not None:
        return Path(folder_paths.get_input_directory()) / DEFAULT_SUBFOLDER

    return Path.cwd() / "input" / DEFAULT_SUBFOLDER


def _build_filename(url, parsed, content_type, filename_prefix):
    safe_prefix = _sanitize_filename(filename_prefix.strip() or DEFAULT_PREFIX)
    extension = _get_extension(parsed, content_type)
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    return f"{safe_prefix}_{url_hash}{extension}"


def _get_extension(parsed, content_type):
    path_name = os.path.basename(unquote(parsed.path))
    path_extension = Path(path_name).suffix
    if _is_safe_extension(path_extension):
        return path_extension.lower()

    guessed_extension = mimetypes.guess_extension(content_type or "")
    if guessed_extension:
        return guessed_extension

    return ".bin"


def _is_safe_extension(extension):
    return extension.startswith(".") and 1 < len(extension) <= 16 and "/" not in extension and "\\" not in extension


def _sanitize_filename(value):
    safe = []
    for character in value:
        if character.isalnum() or character in ("-", "_"):
            safe.append(character)
        else:
            safe.append("_")

    filename = "".join(safe).strip("_")
    return filename or DEFAULT_PREFIX


def load_image(local_path):
    try:
        import numpy as np
        import torch
        from PIL import Image, ImageOps
    except ImportError as error:
        raise ImportError("IMAGE output requires pillow, numpy, and torch in the ComfyUI environment.") from error

    image = Image.open(local_path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image_array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(image_array)[None,]


def load_audio(local_path):
    try:
        import torchaudio
    except ImportError as error:
        raise ImportError("AUDIO output requires torchaudio in the ComfyUI environment.") from error

    waveform, sample_rate = torchaudio.load(local_path)
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)

    return {
        "waveform": waveform,
        "sample_rate": sample_rate,
    }
