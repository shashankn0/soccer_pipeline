from ultralytics import YOLO
import cv2
import os
import gc

import torch
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect import DetectionTrainer
add_safe_globals([DetectionModel])
# Manual loading
ckpt = torch.load("models/player.pt", map_location='cpu', weights_only=False)
# If you really need YOLO object:
# should work now after add_safe_globals
model_players = YOLO("models/player.pt")


# Set video path
video_path = "match_videos/videoplaybacktest.mp4"

'''
# Open video and skip first 25 seconds
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
skip_frames = int(fps * 25)

print(f"Skipping first {skip_frames} frames (~25 seconds)...")
for _ in range(skip_frames):
    cap.read()
'''

# Players
print("Detecting players...")
model_players = YOLO("models/player.pt")
for r in model_players.predict(source=video_path, save=True, conf=0.4, stream=True):
    pass
del model_players
print("Player detection done.")

# Balls
print("Detecting balls...")
model_balls = YOLO("models/ball.pt")
for r in model_balls.predict(source=video_path, save=True, conf=0.4, stream=True):
    pass
del model_balls
print("Ball detection done.")

# Field
print("Detecting field...")
model_field = YOLO("models/field.pt")
for r in model_field.predict(source=video_path, save=True, conf=0.4, stream=True):
    pass
del model_field
print("Field detection done.")
