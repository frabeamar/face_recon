from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from insightface.app import FaceAnalysis
from typer import Typer

from data import Detection

class FaceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedder = FaceAnalysis()
        self.embedder.prepare(ctx_id=0, det_size=(640, 640))

    def forward(self, x: np.ndarray) -> list[Detection] | None:
        faces = self.embedder.get(x)
        if len(faces) == 0:
            return None
        return [Detection(**f) for f in faces]
