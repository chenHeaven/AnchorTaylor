from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import SETTINGS


ROOT = Path(__file__).resolve().parent
MODEL_CFG = ROOT / "yaml/fusehd_talornet_ir.yaml"
DATA_CFG = ROOT / "data/M3FD.yaml"
TRAIN_CFG = ROOT / "ultralytics/cfg/defaultm3fd.yaml"


def main():
    SETTINGS["wandb"] = False

    model = YOLO(str(MODEL_CFG))
    model.train(
        cfg=str(TRAIN_CFG),
        data=str(DATA_CFG),
    )


if __name__ == "__main__":
    main()
