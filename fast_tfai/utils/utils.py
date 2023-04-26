from pathlib import Path
from typing import List, Union


def get_all_images(folder: Union[str, Path]) -> List[Path]:
    img_fmts = ["*.png", "*.jpg", "*.tif", "*.tiff"]
    if isinstance(folder, str):
        folder = Path(folder)

    images = []
    for img_fmt in img_fmts:
        images.extend(folder.rglob(img_fmt))

    return images
