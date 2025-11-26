from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from time import time, perf_counter
from matplotlib.pylab import f
import numpy as np
from sklearn.neighbors import KDTree
import torch.nn as nn
from insightface.app import FaceAnalysis
import pandas as pd
from skfda.exploratory.stats import geometric_median
from functools import wraps

from typer import Typer

from data import Detection, Paths
from deepface import DeepFace
app = Typer(pretty_exceptions_show_locals=False)
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




@dataclass
class NNClassifier:
    cluster_centers: KDTree
    human_label: pd.DataFrame


    @classmethod
    def from_mean(cls, embeddings: np.ndarray, labels: pd.DataFrame):
        means = []
        for i in range(0, max(labels.labels)):
            means.append(embeddings[labels.labels == i].mean(axis=0))
        tree = KDTree(np.stack(means))
        return NNClassifier(tree, labels)
    
    @classmethod
    def from_median(cls, embeddings: np.ndarray, labels: pd.DataFrame):
        median = []
        for i in range(0, max(labels.labels)):
            gm = geometric_median( embeddings[labels.labels == i])
            median.append(gm)
        return NNClassifier(KDTree(np.stack(median)), labels)


    def predict(self, x: np.ndarray) -> tuple[str, int]:
        distance, index = self.cluster_centers.query(x)
        return self.human_label[index].person, distance







