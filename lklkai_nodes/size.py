RESOLUTION_PRESETS = {
    "480p": {"edge": 480, "mode": "short"},
    "720p": {"edge": 720, "mode": "short"},
    "1080p": {"edge": 1080, "mode": "short"},
    "1K": {"edge": 1024, "mode": "long"},
    "2K": {"edge": 2048, "mode": "long"},
    "4K": {"edge": 4096, "mode": "long"},
}

ASPECT_RATIOS = {
    "1:1": (1, 1),
    "2:3": (2, 3),
    "3:2": (3, 2),
    "3:4": (3, 4),
    "4:3": (4, 3),
    "4:5": (4, 5),
    "5:4": (5, 4),
    "16:9": (16, 9),
    "9:16": (9, 16),
    "21:9": (21, 9),
    "9:21": (9, 21),
}

ROUND_TO = 64


class ResolutionByAspectRatio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resolution_preset": (list(RESOLUTION_PRESETS.keys()),),
                "aspect_ratio": (list(ASPECT_RATIOS.keys()),),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "run"
    CATEGORY = "LKLKAI Nodes"

    def run(self, resolution_preset, aspect_ratio):
        preset = RESOLUTION_PRESETS[resolution_preset]
        width, height = calculate_size(
            edge=preset["edge"],
            edge_mode=preset["mode"],
            aspect_ratio=ASPECT_RATIOS[aspect_ratio],
        )
        return (width, height)


def calculate_size(edge, aspect_ratio, edge_mode="long"):
    if edge <= 0:
        raise ValueError("edge must be greater than zero.")

    if edge_mode not in ("long", "short"):
        raise ValueError("edge_mode must be either 'long' or 'short'.")

    width_ratio, height_ratio = aspect_ratio
    if width_ratio <= 0 or height_ratio <= 0:
        raise ValueError("Aspect ratio values must be greater than zero.")

    if edge_mode == "long":
        if width_ratio >= height_ratio:
            width = edge
            height = edge * height_ratio / width_ratio
        else:
            width = edge * width_ratio / height_ratio
            height = edge
    else:
        if width_ratio >= height_ratio:
            width = edge * width_ratio / height_ratio
            height = edge
        else:
            width = edge
            height = edge * height_ratio / width_ratio

    return _round_to_multiple(width), _round_to_multiple(height)


def _round_to_multiple(value):
    return max(ROUND_TO, int(round(value / ROUND_TO) * ROUND_TO))
