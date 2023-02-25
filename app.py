#!/usr/bin/env python

from __future__ import annotations

import functools
import os
import pathlib
import subprocess

if os.environ.get('SYSTEM') == 'spaces':
    subprocess.call('pip uninstall -y opencv-python'.split())
    subprocess.call('pip uninstall -y opencv-python-headless'.split())
    subprocess.call('pip install opencv-python-headless==4.5.5.62'.split())

import cv2
import face_alignment
import gradio as gr
import numpy as np
import torch

TITLE = 'face-alignment'
DESCRIPTION = 'This is an unofficial demo for https://github.com/1adrianb/face-alignment.'


def detect(
    image: np.ndarray,
    detector,
    device: torch.device,
) -> np.ndarray:
    preds = detector.get_landmarks(image)
    if len(preds) == 0:
        raise RuntimeError('No face was found')

    res = image.copy()
    for pts in preds:
        tl = pts.min(axis=0)
        br = pts.max(axis=0)
        size = (br - tl).max()
        radius = max(2, int(3 * size / 256))
        for pt in np.round(pts).astype(int):
            cv2.circle(res, tuple(pt), radius, (0, 255, 0), cv2.FILLED)
    return res


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                        device=device.type)
func = functools.partial(detect, detector=detector, device=device)

image_paths = sorted(pathlib.Path('images').glob('*.jpg'))
examples = [[path.as_posix()] for path in image_paths]

gr.Interface(
    fn=func,
    inputs=gr.Image(label='Input', type='numpy'),
    outputs=gr.Image(label='Output', type='numpy'),
    examples=examples,
    title=TITLE,
    description=DESCRIPTION,
).launch(show_api=False)
