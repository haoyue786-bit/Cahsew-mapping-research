"""Produce numbered YOLO field proposals for a single full-size RGB image."""

import argparse
from hashlib import sha256
import json
import math
import sys
from pathlib import Path
from time import monotonic

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO

WEIGHTS = Path("/home/ubuntu/.cache/huggingface/hub/models--MykolaL--DelineateAnything/snapshots/"
               "46f98c05ece1d57047e36952a8b440583de38f0d/DelineateAnythingv2.pt")
WEIGHT_SHA = "46700b8a279b07922953a11adaeb5e658d9a2384b6334c8e0a3090886218915a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rgb", type=Path)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()

    if sha256(WEIGHTS.read_bytes()).hexdigest() != WEIGHT_SHA:
        raise ValueError("Field detection requires the pinned public checkpoint")
    with Image.open(args.rgb) as source:
        scene = source.convert("RGB")
    started = monotonic()
    model = YOLO(str(WEIGHTS), task="segment")
    predicted = model.predict(scene, imgsz=scene.width, conf=.15, device=0, half=True,
                              max_det=sys.maxsize, verbose=False)[0]
    torch.cuda.synchronize()
    boxes = []
    canvas = scene.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    palette = ("#fbd945", "#9bdfff", "#e5a9ff", "#79f2b3", "#ff9d77")
    args.folder.mkdir(parents=True, exist_ok=True)
    for bbox, score in zip(predicted.boxes.xyxy.cpu().tolist(), predicted.boxes.conf.cpu().tolist()):
        x0, y0, x1, y1 = [max(0, min(limit, adjust(value))) for value, adjust, limit in
                          zip(bbox, (math.floor, math.floor, math.ceil, math.ceil), scene.size * 2)]
        if x1 - x0 < 16 or y1 - y0 < 16:
            continue
        identity = f"R{len(boxes) + 1:02d}"
        color = palette[len(boxes) % len(palette)]
        boxes.append(dict(id=identity, box=[x0, y0, x1, y1], score=round(score, 4)))
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=color, width=3)
        corner = (min(x0 + 2, scene.width - 52), min(y0 + 2, scene.height - 30))
        draw.rectangle(draw.textbbox(corner, identity, font=font), fill="#16201d")
        draw.text(corner, identity, font=font, fill=color)
    canvas.save(args.folder / "numbered_boxes.png")
    data = dict(tool="Delineate Anything v2", rgb_sha256=sha256(args.rgb.read_bytes()).hexdigest(),
                load_inference_and_render_seconds=round(monotonic() - started, 3), boxes=boxes)
    (args.folder / "PROPOSALS.json").write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
