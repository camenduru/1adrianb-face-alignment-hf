#!/usr/bin/env python

from __future__ import annotations

import argparse
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

ORIGINAL_REPO_URL = 'https://github.com/1adrianb/face-alignment'
TITLE = '1adrianb/face-alignment'
DESCRIPTION = f'This is an unofficial demo for {ORIGINAL_REPO_URL}.'
ARTICLE = ''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    parser.add_argument('--allow-flagging', type=str, default='never')
    parser.add_argument('--allow-screenshot', action='store_true')
    return parser.parse_args()


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


def main():
    gr.close_all()

    args = parse_args()
    device = torch.device(args.device)

    detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                            device=device.type)

    func = functools.partial(detect, detector=detector, device=device)
    func = functools.update_wrapper(func, detect)

    image_paths = sorted(pathlib.Path('images').glob('*.jpg'))
    examples = [[path.as_posix()] for path in image_paths]

    gr.Interface(
        func,
        gr.inputs.Image(type='numpy', label='Input'),
        gr.outputs.Image(type='numpy', label='Output'),
        examples=examples,
        title=TITLE,
        description=DESCRIPTION,
        article=ARTICLE,
        theme=args.theme,
        allow_screenshot=args.allow_screenshot,
        allow_flagging=args.allow_flagging,
        live=args.live,
    ).launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
