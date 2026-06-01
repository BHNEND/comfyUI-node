# Changelog

## 2026-06-01

- Created the local Codex plugin scaffold `comfyui-workflow-control` with a personal marketplace entry.
- Added ComfyUI workflow control guidance for connecting to local or cloud ComfyUI instances from Codex.
- Added a dependency-free Node MCP server and CLI client for ComfyUI system stats, queue, history, validation, patching, submission, and interrupt operations.
- Added local proxy bypass environment variables to the ComfyUI MCP server config so localhost API calls are not routed through an HTTP proxy.
- Updated the local ComfyUI plugin default endpoint to `http://127.0.0.1:8000` after verifying the desktop ComfyUI app is listening there.
- Registered the local `personal` plugin marketplace root with Codex so `comfyui-workflow-control` can appear in the Codex app plugin catalog.
- Updated the ComfyUI workflow control plugin default endpoint to the configured cloud ComfyUI URL.
- Added compatibility nodes for imported Qwen3 TTS voice-cloning workflows: `LoadAudioFromUrl` and `HeartMuLa_Transcribe`.
- Removed the experimental Qwen3 TTS compatibility nodes to keep missing workflow dependencies explicit during development.
- Added a first-party Qwen3 TTS node set with model loading, voice cloning, reusable voice prompts, custom voice generation, and voice design based on the official `qwen-tts` package.
- Added repository-level ComfyUI entrypoint, dependency list, and GitHub clone/update instructions for easier online updates.
- Added the `LK-` prefix to Qwen3 TTS node display names while keeping internal class names compatible with imported workflows.

## 2026-05-25

- Added the local `LK-URL 2 File` node for downloading an `http` or `https` URL into ComfyUI's input directory and returning the local file path.
- Added `IMAGE`, `VIDEO`, and `AUDIO` output selection to `LK-URL 2 File`, while keeping `local_path` as a fallback output.
- Added `480p`, `720p`, and `1080p` options to the local `LK-尺寸转换` node.
- Kept `1K`, `2K`, and `4K` as long-edge presets while treating the new `p` presets as standard video-size presets.
- Restored 64-pixel rounding so generated dimensions are AI-generation friendly.
- Updated local node documentation for the expanded resolution presets.

## 2026-05-22

- Added a standalone cloud API code path for an `ltx23` worker that calls the official LTX-2 pipeline without using the ComfyUI runtime.
- Added remote API support for `image_url` download, `negative_prompt` request data, and LTX-2.3 model/upscaler configuration through environment variables.
- Tested LTX-2.3/19B-style official pipeline startup on the A10G instance and found the current 30GB system memory is insufficient before GPU inference begins.
- Verified the official LTX-2.3 distilled pipeline can generate a minimal 128x128, 9-frame MP4 on the current A10G instance when called directly through `ltx_pipelines.distilled`.
- Generated a 1280x704, 121-frame, 24fps official LTX-2.3 distilled test video and exposed it through the public API outputs path.
- Tested official LTX-2.3 HQ two-stage generation with `distilled-lora-384`; FP8 LoRA fusion failed on A10G due unsupported FP8 kernel, and the non-quantized LoRA path was killed by resource limits even at 128x128.
- Restored the public API service to the stable standalone `ltx2b` backend after the large-model validation attempt.

## 2026-05-21

- Brought up a cloud inference API prototype and replaced the mock video worker with a real LTX-Video 2B distilled worker.
- Renamed the displayed size node to `LK-尺寸转换`.
- Flattened the package layout so node code lives directly under `lklkai_nodes`.
- Simplified `Resolution By Aspect Ratio` to two dropdown inputs: resolution and aspect ratio.
- Limited resolution presets to `1K`, `2K`, and `4K`.
- Limited outputs to `width` and `height`.
- Created the `lklkai_nodes` ComfyUI custom node package skeleton.
- Added the first size helper node: `Resolution By Aspect Ratio`.
