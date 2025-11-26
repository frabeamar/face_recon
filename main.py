from pathlib import Path
import pickle

import cv2
from utils import timeit
from  deepface import DeepFace
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from typer import Typer

from data import (
    ClusterMethod,
    Detection,
    ImageFrame,
    Paths,
    VideoFrame,
    crop_image,
    image_reader,
    load_all_images,
    make_dirs,
    read_image,
    video_reader,
)
from model import FaceModel, NNClassifier

app = Typer(pretty_exceptions_show_locals=False)


@app.command()
def create_dataset(folder: str):
    model = FaceModel()
    paths = Paths(Path(folder).parent /( Path(folder).stem + "_dataset"))
    make_dirs(paths)
    images = []
    for image_file in image_reader(folder):
        prediction: list[Detection] | None = model(read_image(image_file))
        if prediction:
            frame = ImageFrame(prediction, image_file)
            images.append(frame)
    (paths.images).write_bytes(pickle.dumps(images))

    images = []
    for image_file, frame_num in video_reader(folder):
        prediction: list[Detection] | None = model(read_image(image_file))
        if prediction:
            frame = VideoFrame(prediction, image_file, frame_num)
            images.append(frame)
    (Paths().video_frame).write_bytes(pickle.dumps(images))


@app.command()
def cluster(min_samples: int = 50, metric: str = "cosine"):
    dbscan = DBSCAN(metric="cosine", min_samples=50)
    method = ClusterMethod(metric=metric, min_samples=min_samples)
    images = load_all_images()
    embeddings = np.stack([det.embedding for img in images for det in img.detections])
    labels = dbscan.fit_predict(embeddings)
    df = pd.DataFrame({"labels": labels})
    df.to_csv(Paths().clustering(method), index=False)
    df = pd.DataFrame(
        {
            "labels": [i for i in range(max(labels))],
            "person": ["unknown" for i in range(max(labels))],
        }
    )
    df.to_csv(Paths().human_labels(method), index=False)


@app.command()
def visualize_clusters(metric: str = "cosine", min_samples: int = 50, M: int = 5):
    method = ClusterMethod(metric=metric, min_samples=min_samples)
    human_labels = pd.read_csv(Paths().human_labels(method))
    images = load_all_images()
    df = pd.read_csv(Paths().clustering(method))
    N = len(human_labels)

    bboxes = [det.bbox for img in images for det in img.detections]
    filenames = [img.filename for img in images for det in img.detections]
    fig, axs = plt.subplots(N, M, figsize=(M, N))
    
    for i in range(N):
        sub_df = df[df.labels == i]
        sub_df = sub_df.sample(min(M, len(sub_df)))
        for t, (index, sample) in enumerate(sub_df.iterrows()):
            index = int(index)
            img = read_image(filenames[index])
            bbox = np.array(bboxes[index])
            crop = crop_image(img, bbox)
            axs[i, t].imshow(crop)
            axs[i, t].axis("off")
            person = human_labels.loc[i].person 
            ident = person if person !="unknown" else i
            axs[i, 0].annotate(ident, xy=(-0.5, 1))

    plt.tight_layout()
    plt.savefig("clusters.png", dpi=100)
    plt.close()


@app.command()
def benchmark():
    models = [
    "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace",
    "DeepID", "ArcFace", "SFace", "GhostFaceNet",
    "Buffalo_L",
]
    images = [read_image(im.filename) for im in load_all_images()[:10]]
    images = [cv2.resize(im, (753, 1440)) for im in images]
    
    @timeit
    def run_deepface(model, N:int = 10):
        embedding_objs = DeepFace.represent(img_path = images, model_name = model, enforce_detection=False)
        return embedding_objs

 

    results = []
    for m in models:
        _, time = run_deepface(m, N=1)
        results.append({"model": m, "time": time})
        print(results)
    pd.DataFrame(results).to_csv("benchmark.csv", index=False)


@app.command()
def predict(folder:str):
    feature_extractor = FaceModel()
    images = load_all_images()
    embeddings = np.stack([det.embedding for img in images for det in img.detections])
    labels = pd.read_csv(Paths().clustering(ClusterMethod(metric="cosine", min_samples=50)))
    median = NNClassifier.from_median(embeddings, labels)
    mean = NNClassifier.from_mean(embeddings, labels)
    
    results = []
    for img_file in image_reader(folder):
        img = read_image(img_file)
        prediction: list[Detection] | None = feature_extractor(img)
        if prediction:
            for det in prediction:
                person, distance = median.predict(det.embedding.reshape(1, -1))
                results.append({"distance":distance, "person":person, "method": "median"})
                person, distance = mean.predict(det.embedding.reshape(1, -1))
                results.append({"distance":distance, "person":person, "method": "mean"})
    
    sns.relplot(pd.DataFrame(results).groupby("person").mean(), x="person", y="distance").savefig("mean_distance.png")
        


app()
