# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.1.47"

# NOTE: Keep ultralytics import lightweight.
# Some submodules (e.g. Explorer) depend on optional packages like cv2.
try:
    from ultralytics.data.explorer.explorer import Explorer  # noqa: F401
except Exception:  # pragma: no cover
    Explorer = None

try:
    from ultralytics.models import RTDETR, SAM, YOLO, YOLOWorld  # noqa: F401
    from ultralytics.models.fastsam import FastSAM  # noqa: F401
    from ultralytics.models.nas import NAS  # noqa: F401
except Exception:  # pragma: no cover
    RTDETR = SAM = YOLO = YOLOWorld = FastSAM = NAS = None
from ultralytics.utils import ASSETS, SETTINGS
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

settings = SETTINGS
__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
    "Explorer",
)
