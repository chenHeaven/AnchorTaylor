from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG
import ultralytics.nn.tasks

import traceback
from copy import deepcopy

import torch

try:
    from thop import profile
except Exception as e:
    profile = None
    print(f"Failed to import thop: {e}")


def debug_thop_flops(model, imgsz=640, device="cuda:0"):
    """Try to compute full GFLOPs with thop and print the real failure reason if it breaks."""
    if profile is None:
        raise RuntimeError("thop is not available")

    model = deepcopy(model).to(device).eval()
    p = next(model.parameters())
    dtype = p.dtype

    # Match ultralytics' custom 6-channel multimodal input
    im = torch.empty((1, 6, imgsz, imgsz), device=device, dtype=dtype)

    try:
        flops = profile(model, inputs=(im,), verbose=False)[0] / 1e9 * 2
        return flops
    except Exception as e:
        print("[thop debug] first attempt failed:")
        print(repr(e))
        traceback.print_exc()

    # Second attempt: use stride-sized input like ultralytics' internal fallback
    try:
        stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32
        im = torch.empty((1, 6, stride, stride), device=device, dtype=dtype)
        flops = profile(model, inputs=(im,), verbose=False)[0] / 1e9 * 2
        return flops * imgsz / stride * imgsz / stride
    except Exception as e:
        print("[thop debug] second attempt failed:")
        print(repr(e))
        traceback.print_exc()
        raise


model = YOLO("/home/asefu/project/WaveMamba-main/runs/flir_align/middle_p1_p2_p33_p4/weights/best.pt")

# 打印参数量
model.info(verbose=True)

# 用 thop 统计完整 FLOPs，并在失败时输出真实异常
try:
    gflops = debug_thop_flops(model.model, imgsz=640, device="cuda:0")
    print(f"Full GFLOPs (thop): {gflops:.2f}")
except Exception as e:
    print(f"Full GFLOPs failed: {e}")

results = model.val(
    data="/home/asefu/project/WaveMamba-main/data/flir_align.yaml",
    batch=16,
    imgsz=640,
    device=0,
    project="/home/asefu/project/WaveMamba-main/ceshi",
)
