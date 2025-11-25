import glob
import logging
import pickle
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tqdm

from utils import Paths

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Face:
    bbox: tuple[float]
    ident: str
    image_path: str


@dataclass(frozen=True)
class FramedFace:
    face: Face
    frame_num: int


@dataclass
class Stats:
    distance_from_custer: float


@dataclass
class Detection:
    bbox: np.ndarray
    kps: np.ndarray
    det_score: np.float32
    landmark_3d_68: np.ndarray
    pose: np.ndarray
    landmark_2d_106: np.ndarray
    gender: np.int64
    age: int
    embedding: np.ndarray


@dataclass(frozen=True)
class ImageFrame:
    detections: list[Detection]
    filename: Path


@dataclass(frozen=True)
class VideoFrame:
    detections: list[Detection]
    filename: Path
    frame_num: int


def load_data():
    edis = pickle.loads(Path("edis.pkl").read_bytes())
    embeddings = np.array([d.embedding for e in edis for d in e.people])
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    bbox = np.array([d.bbox for e in edis for d in e.people])

    filenames = [e.filename for e in edis for d in e.people]
    return embeddings, bbox, filenames


def extract_frames_from_video(video_path: Path):
    out_path = Paths.top_folder / "videos"
    video_folder = out_path / Path(video_path).stem
    video_folder.mkdir(parents=True, exist_ok=True)
    for j, res in enumerate(gen_frames_from_video(video_path)):
        cv2.imwrite(str(video_folder / f"{j:05}.jpg"), res)
    return video_folder


def gen_frames_from_video(video_path: Path, every_n: int = 1):
    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        ret, res = cap.read()
        if not ret:
            break
        if count % every_n == 0:
            yield res
        count += 1
        yield res


def video_reader(base_path: str) -> Iterator[tuple[Path, int]]:
    """
    Extracts all frame from each video present in base path, then yields the path for each extracted image
    """
    videos = sorted(glob.glob(f"{base_path}/*.mp4"))
    for i, video in enumerate(videos):
        image_folder = extract_frames_from_video(Path(video))
        images = sorted(Path(image_folder).iterdir())
        for img_file in tqdm.tqdm(images, desc=f"Processing video {i}/{len(videos)}"):
            yield Path(img_file), i


def image_reader(base_path: str) -> Iterator[Path]:
    files = sorted(list(Path(base_path).glob("*.jpg")))
    for file in tqdm.tqdm(files, desc="Reading_images", total=len(files)):
        yield Path(file)


def crop_image(img: np.ndarray, bbox: np.ndarray):
    x, y, xx, yy = map(int, bbox)
    w = xx - x
    h = yy - y
    y = max(y - h * 0.2, 0)
    x = max(x - w * 0.2, 0)
    xx = min(xx + w * 0.2, img.shape[1])
    yy = min(yy + h * 0.2, img.shape[0])
    x, y, xx, yy = map(int, [x, y, xx, yy])
    return img[y:yy, x:xx]


def read_image(filename: str | Path) -> np.ndarray:
    image = cv2.imread(str(filename), cv2.IMREAD_COLOR)  # pyright: ignore[reportCallIssue]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image is None:
        raise IOError("Failed to load image: {}".format(filename))
    return image
