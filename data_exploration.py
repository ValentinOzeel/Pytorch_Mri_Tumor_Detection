import os
import random 
from PIL import Image
from typing import List
from pathlib import Path

from secondary_module import read_yaml


project_root_path = os.path.abspath(os.path.dirname(__file__))
config = read_yaml(os.path.join(project_root_path, 'conf', 'config.yml'))
random.seed(config['RANDOM_SEED'])


def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"In {dir_path}, there are:\n- {len(dirnames)} Directories\n- {len(filenames)} Files")
        
def get_all_image_paths(data_path:Path) -> List:
    return list(data_path.glob("*/*/*.jpg"))

if __name__ == "__main__":
    
    # Walk through our data directory
    walk_through_dir(os.path.join(project_root_path, 'data'))
    # Get all image (train/test) paths
    all_image_paths = get_all_image_paths(Path(os.path.join(project_root_path, 'data')))
    # Pick a random image path
    random_image_path = random.choice(all_image_paths)
    # Get image class (name of parent directory)
    image_class = random_image_path.parent.stem
    # Open the random image
    random_img = Image.open(random_image_path)
    # Print random image's metadata
    print(f"Random image's path: {random_image_path}")
    print(f"Random image's class: {image_class}")
    print(f"Random image's height and width: {random_img.height} / {random_img.width}")
    random_img.show()