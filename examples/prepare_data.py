import open3d as o3d
import shutil
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SAMPLE_DIR = os.path.join(_REPO_ROOT, "data", "sample_scene")

print("Downloading RGB-D sample...")
dataset = o3d.data.SampleRedwoodRGBDImages()

os.makedirs(_SAMPLE_DIR, exist_ok=True)

shutil.copy(dataset.color_paths[0], os.path.join(_SAMPLE_DIR, "color.jpg"))
shutil.copy(dataset.depth_paths[0], os.path.join(_SAMPLE_DIR, "depth.png"))

print("Success! Data prepared in data/sample_scene/")
