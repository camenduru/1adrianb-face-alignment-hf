#!/usr/bin/env python

from __future__ import annotations

import functools
import pathlib

import cv2
import face_alignment
import gradio as gr
import numpy as np
import torch

TITLE = 'face-alignment'
DESCRIPTION = 'https://github.com/1adrianb/face-alignment'

MAX_IMAGE_SIZE = 1800


def detect(
    image: np.ndarray,
    detector,
    device: torch.device,
) -> np.ndarray:
    landmarks, _, boxes = detector.get_landmarks(image, return_bboxes=True)
    if landmarks is None:
        return image

    res = image.copy()
    for pts, box in zip(landmarks, boxes):
        box = np.round(box[:4]).astype(int)
        cv2.rectangle(res, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 2)
        tl = pts.min(axis=0)
        br = pts.max(axis=0)
        size = (br - tl).max()
        radius = max(2, int(3 * size / 256))
        for pt in np.round(pts).astype(int):
            cv2.circle(res, tuple(pt), radius, (0, 255, 0), cv2.FILLED)
    return res


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,
                                        device=device.type)
fn = functools.partial(detect, detector=detector, device=device)

image_paths = sorted(pathlib.Path('images').glob('*.jpg'))
examples = [[path.as_posix()] for path in image_paths]

gr.Interface(
    fn=fn,
    inputs=gr.Image(label='Input', type='numpy'),
    outputs=gr.Image(label='Output', type='numpy'),
    examples=examples,
    title=TITLE,
    description=DESCRIPTION,
).queue().launch()
