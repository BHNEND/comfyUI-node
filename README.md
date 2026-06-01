# lklkai_nodes

Personal ComfyUI helper nodes for workflow automation.

## Install

Clone this repository into ComfyUI's `custom_nodes` directory:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/BHNEND/comfyUI-node.git
cd comfyUI-node
pip install -r requirements.txt
```

Restart ComfyUI, then search for `LK-尺寸转换` or `LK-Qwen3 TTS`.

For local development, link this package folder into ComfyUI's `custom_nodes` directory:

```bash
ln -s /Users/wangyang/Documents/comfyUI-node /path/to/ComfyUI/custom_nodes/comfyUI-node
```

## Update

If the repository was installed with `git clone`, update it with:

```bash
cd /path/to/ComfyUI/custom_nodes/comfyUI-node
git pull
pip install -r requirements.txt
```

Restart ComfyUI after updating.

## Nodes

### LK-尺寸转换

Category: `LKLKAI Nodes`

Calculates image dimensions from two dropdowns:

- `resolution_preset`: `480p`, `720p`, `1080p`, `1K`, `2K`, `4K`
- `aspect_ratio`: `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `4:5`, `5:4`, `16:9`, `9:16`, `21:9`, `9:21`

It returns:

- `width`
- `height`

The `1K`, `2K`, and `4K` presets are treated as the long edge. For example, `1K` with `16:9` returns `1024 x 576`.

The `480p`, `720p`, and `1080p` presets are video-size presets, and all outputs are rounded to multiples of 64 for AI generation compatibility. For example, `720p` with `16:9` returns `1280 x 704`.

### LK-URL 2 File

Category: `LKLKAI Nodes`

Downloads a remote `http` or `https` URL into ComfyUI's input directory, loads the selected media type, and returns the local file path as a fallback.

Inputs:

- `url`: Remote file URL.
- `output_type`: `IMAGE`, `VIDEO`, or `AUDIO`.
- `filename_prefix`: Local filename prefix. Defaults to `url_asset`.
- `overwrite`: Re-download and replace the local file when enabled.

It returns:

- `image`
- `video`
- `audio`
- `local_path`

Files are saved under `input/lklkai_url` with a stable filename based on the URL hash. The selected media output is populated, and the other media outputs are left empty.

### Qwen3 TTS nodes

Category: `LKLKAI Nodes/Qwen3 TTS`

Adds Qwen3 TTS nodes based on the official `qwen-tts` Python package:

- `Qwen3TTSModelLoader`
- `Qwen3TTSVoiceClone`
- `Qwen3TTSCreateVoicePrompt`
- `Qwen3TTSVoiceCloneWithPrompt`
- `Qwen3TTSCustomVoice`
- `Qwen3TTSVoiceDesign`

These nodes are displayed with the `LK-Qwen3 TTS` prefix in ComfyUI. The model loader downloads Hugging Face Qwen3 TTS models into `ComfyUI/models/TTS` when `huggingface_hub` is available. The voice clone node keeps the class name and Chinese port names used by imported `Qwen3 TTS 语音克隆` workflows.

Install runtime dependencies in the ComfyUI Python environment:

```bash
pip install qwen-tts soundfile transformers accelerate huggingface_hub numpy
```
