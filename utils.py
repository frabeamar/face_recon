
from pathlib import Path
from dataclasses import dataclass
from functools import cached_property
DATA_DIR = Path.home() / "data"



@dataclass
class Paths:
    top_folder:Path = DATA_DIR / "edis"

    @cached_property
    def human_labels(self):
        return  (self.top_folder / f"edis/{method.as_string()}_human_labels.json")
    
    @cached_property
    def clustering(self)->Path:
        return self.top_folder / f"{method.as_string()}.csv"

    @cached_property
    def cluster_folder(self)->Path:
        return self.top_folder / "clustered"
    
    @cached_property
    def dataset(self)->Path:
        return self.top_folder / "dataset"
    
    @cached_property
    def video_frame(self)->Path:
        return self.top_folder / "video_frames.pkl"
    
    @cached_property
    def images(self)->Path:
        return self.top_folder / "images.pkl"

def make_dirs():
    for path in [Paths().top_folder, Paths().cluster_folder]:
        path.mkdir(exist_ok=True)

make_dirs()
