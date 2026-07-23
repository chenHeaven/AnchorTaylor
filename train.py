#训练 - 从 train17 继续训练
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
import ultralytics.nn.tasks

SETTINGS["wandb"] = False

last_ckpt = "/home/asefu/project/WaveMamba-main/runs/M3FD_xiaorong/ablation_last2/weights/last.pt"

model = YOLO(last_ckpt)
# model = YOLO("/home/asefu/project/WaveMamba-main/WaveMamba-main/runs/m3fd/train10/weights/best.pt")
# results = model.train(data="/home/asefu/project/WaveMamba-main/WaveMamba-main/data/flir_aligned.yaml",batch=12,epochs=300,imgsz=640,device=0,project="/home/asefu/project/WaveMamba-main/WaveMamba-main/runs/m3fd/")
results = model.train(
    resume=True,
    batch=8,
    imgsz=640,
    device=0,
)
