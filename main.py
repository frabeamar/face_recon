import pickle

from typer import Typer

from data import (
    Detection,
    ImageFrame,
    VideoFrame,
    image_reader,
    read_image,
    video_reader,
)
from model import FaceModel
from utils import Paths

app = Typer(pretty_exceptions_show_locals=False)


@app.command()
def create_dataset(folder: str):
    model = FaceModel()

    images = []
    for image_file in image_reader(folder):
        prediction: list[Detection] | None = model(read_image(image_file))
        if prediction:
            frame = ImageFrame(prediction, image_file)
            images.append(frame)
    (Paths().images).write_bytes(pickle.dumps(images))

    images = []
    for image_file, frame_num in video_reader(folder):
        prediction: list[Detection] | None = model(read_image(image_file))
        if prediction:
            frame = VideoFrame(prediction, image_file, frame_num)
            images.append(frame)
    (Paths().video_frame).write_bytes(pickle.dumps(images))


app()
