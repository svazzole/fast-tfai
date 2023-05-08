from pathlib import Path
from typing import Dict, List, Union

import cv2
import numpy as np
from rich.progress import track

from fast_tfai.utils.console import parse_cli
from fast_tfai.utils.utils import get_all_images


def canny_crop(
    img: np.ndarray,
    width: int,
    height: int,
    threshold1: int = 100,
    threshold2: int = 255,
) -> np.ndarray:
    canny_output = cv2.Canny(img, threshold1, threshold2)
    v_sum = np.sum(canny_output, axis=0)

    min_hix = 0
    p = np.nonzero(v_sum)
    if np.any(p):
        min_hix = np.min(p)

    left = min_hix
    right = min_hix + width
    if right > img.shape[1]:
        left = np.max([img.shape[1] - width, 0])
        right = np.min([left + width, img.shape[1]])

    h_sum = np.sum(canny_output[:, left:right], axis=1)

    min_vix = 0
    p = np.nonzero(h_sum)
    if np.any(p):
        min_vix = np.min(p)

    top = min_vix
    bottom = min_vix + height
    if bottom > img.shape[0]:
        top = np.max([img.shape[0] - height, 0])
        bottom = np.min([top + height, img.shape[0]])

    return img[top:bottom, left:right]


def main(cfg: Dict[str, Union[List, str]]):
    input_data = cfg.get("input_data")
    if not input_data or not isinstance(input_data, str):
        raise ValueError("Wrong 'input_data' key.")
    images_path = Path(input_data)

    output_dir = cfg.get("output_path")
    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("Wrong 'output_path' key.")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    crop_shape = cfg.get("crop_shape")
    if not isinstance(crop_shape, list) or not crop_shape:
        raise ValueError("Wrong 'crop_shape' key.")
    height, width = crop_shape

    images_list = get_all_images(images_path)
    for multi in track(images_list):
        _, images = cv2.imreadmulti(str(multi), [], cv2.IMREAD_GRAYSCALE)
        for ix, img in enumerate(images):
            if ix == 0:
                cropped_img = canny_crop(img, width, height)
                multipl = 224 / np.max(crop_shape)
                new_width = int(width * multipl)
                new_height = int(height * multipl)
                cropped_img = cv2.resize(cropped_img, (new_width, new_height))
                fn = str(output_path / f"{multi.stem}_frame_{ix:03d}.png")
                cv2.imwrite(fn, cropped_img)


if __name__ == "__main__":
    import yaml

    args = parse_cli()
    cfg = yaml.safe_load(Path(args.conf_path).open())
    main(cfg["preprocess"])
