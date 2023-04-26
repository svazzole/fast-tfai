from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from fast_tfai.utils.console import console


class Dataset(ABC):
    @property
    @abstractmethod
    def df(self):
        ...

    @df.setter
    @abstractmethod
    def df(self):
        ...

    @property
    @abstractmethod
    def train_df(self):
        ...

    @property
    @abstractmethod
    def val_df(self):
        ...

    @property
    @abstractmethod
    def test_df(self):
        ...

    @abstractmethod
    def split(self, validation_size: float, test_size: float, seed: int):
        ...

    @abstractmethod
    def summary(self):
        ...

    @classmethod
    @abstractmethod
    def from_folder(self):
        ...


@dataclass
class ClassificationDataset(Dataset):

    _df: pd.DataFrame
    _train_df: pd.DataFrame = field(default=None)
    _val_df: pd.DataFrame = field(default=None)
    _test_df: pd.DataFrame = field(default=None)

    def __post_init__(self):

        # Check if the DataFrame has the right columns
        if not set(["label", "image"]).issubset(self.df.columns):
            raise ValueError("The input dataframe has not 'image' or 'label' column.")

        # Extract sample
        self.extract_ids()

        # Get class names and number of classes
        self.class_names = sorted(self.df["label"].unique())
        self.num_classes = len(self.class_names)

        # Maps class2int and int2class
        label_int = [str(x) for x in range(len(self.class_names))]
        self.class2int_map = dict(zip(self.class_names, label_int))
        self.int2class_map = {v: k for k, v in self.class2int_map.items()}

        # Add integer version of the 'label' col
        self.df["label_int"] = self.df["label"].replace(self.class2int_map)

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def train_df(self) -> pd.DataFrame:
        return self._train_df

    @train_df.setter
    def train_df(self, new_df: pd.DataFrame):
        self._train_df = new_df

    @property
    def val_df(self) -> pd.DataFrame:
        return self._val_df

    @val_df.setter
    def val_df(self, new_df: pd.DataFrame):
        self._val_df = new_df

    @property
    def test_df(self) -> pd.DataFrame:
        return self._test_df

    @test_df.setter
    def test_df(self, new_df: pd.DataFrame):
        self._test_df = new_df

    def extract_ids(self):
        sample = ["_".join(Path(x).name.split("_")[:-1]) for x in self.df["image"]]
        self.df["sample"] = sample

    def _find_class_weights(self):
        # Scaling by total/num_classes helps keep the loss to a similar magnitude.
        # The sum of the weights of all examples stays the same.

        counts = self.train_df["label_int"].value_counts()
        total = np.sum(counts)
        weights = [1 / k * (total / len(counts)) for k in counts]

        class_weights = dict(zip([int(x) for x in counts.index], weights))

        console.rule("[bold red] :warning: Weights")
        console.print(
            "Weights to balance the loss function:",
            style="bold red",
        )

        for c in counts.index:
            console.print(
                "Weight for class {} -> {}: {:.2f}".format(
                    self.class_names[int(c)], c, class_weights[int(c)]
                ),
                style="cyan1",
            )

        return class_weights

    def summary(self):
        """Print a summary with some stats about the dataset."""
        console.rule("[bold red] Dataset Summary")
        console.print()
        console.print(f"Dataset size: {self.df.shape[0]}")
        console.print("Dataset distribution:")
        console.print(self.df.label.value_counts(normalize=True), "\n")

        if (
            (self.train_df is not None)
            and (self.val_df is not None)
            and (self.test_df is not None)
        ):
            console.print(f"Train size: {self.train_df.shape[0]}")
            console.print("Train distribution:")
            console.print(self.train_df.label.value_counts(normalize=True), "\n")
            console.print(f"Validation size: {self.val_df.shape[0]}")
            console.print("Validation distribution:")
            console.print(self.val_df.label.value_counts(normalize=True), "\n")
            console.print(f"Test size: {self.test_df.shape[0]}")
            console.print("Test distribution:")
            console.print(self.test_df.label.value_counts(normalize=True), "\n")
        console.rule()

    def split(
        self,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        method: str = "std",
        seed: Optional[int] = None,
    ):
        """Split the dataset into train, val and test. Only stratified splitting
            based on the 'label' column.

        Args:
            test_size (float, optional): test size in percentage in (0, 1). Defaults to 0.2.
            validation_size (float, optional): validation size in percentage in (0, 1).
                Defaults to 0.1.
            seed (Optional[int], optional): set the random number generator seed.
                Defaults to None.
        """
        if method == "std":
            self._stratify(
                test_size=test_size, validation_size=validation_size, seed=seed
            )
        elif method == "group":
            self._stratify_by_group(
                test_size=test_size, validation_size=validation_size, seed=seed
            )

        self.class_weights = self._find_class_weights()

    def _stratify(
        self,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        seed: Optional[int] = None,
    ):
        train_df, self.test_df = train_test_split(
            self.df,
            test_size=test_size,
            stratify=self.df["label"],
            random_state=seed,
        )

        self.train_df, self.val_df = train_test_split(
            train_df,
            test_size=validation_size,
            stratify=train_df["label"],
            random_state=seed,
        )

    def _stratify_by_group(
        self,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        seed: Optional[int] = None,
    ):

        X, y = self.df["image"].values, self.df["label"].values
        groups = self.df["sample"].values

        k = int(1 / test_size)
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        train_index, test_index = next(sgkf.split(X, y, groups))

        self._train_df = self._df.iloc[train_index]
        self._test_df = self._df.iloc[test_index]
        self._train_df.reset_index(inplace=True)
        self._test_df.reset_index(inplace=True)

        k = int(1 / validation_size)
        sgkf2 = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)

        X, y = self.train_df["image"].values, self.train_df["label"].values
        groups = self.train_df["sample"].values
        train_index, val_index = next(sgkf2.split(X, y, groups))

        self._val_df = self._train_df.iloc[val_index]
        self._train_df = self._train_df.iloc[train_index]

    @classmethod
    def from_folder(cls, folder: Union[Path, str]):
        folder = Path(folder)
        csv_file = list(folder.rglob("*.csv"))[0]
        df: pd.DataFrame = pd.read_csv(csv_file)
        dataset = ClassificationDataset(df)
        return dataset


class MultiLabelDataset(Dataset):
    ...


class ObjDetectionDataset(Dataset):
    ...


@dataclass
class ImageFile:
    image_filename: Path
    annotation_filename: Path

    def __post_init__(self):
        self.image_ext = self.image_filename.suffix
        self.annotation_ext = self.annotation_filename.suffiximage.resize((480, 250))
        if not self.isvalid():
            raise ValueError("Image filename != Annotation filename")

    def isvalid(self):
        return self.annotation_filename.name == self.image_filename.name


@dataclass
class Image:
    _image: np.ndarray

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, new_image: np.ndarray):
        self._image = new_image.astype(np.uint8)
        self.__post_init__()

    def __post_init__(self):
        self._image = self._image.astype(np.uint8)
        if len(self.image.shape) == 2:
            self.h, self.w = self.image.shape
            self.c = 1
        else:
            self.h, self.w, self.c = self.image.shape

        self.shape = (self.h, self.w, self.c)

    def resize(self, shape: Tuple[int, int]):
        interpolation = cv2.INTER_AREA
        if (shape[0] > self.h) or (shape[1] > self.w):
            interpolation = cv2.INTER_LINEAR
        self.image = cv2.resize(self.image, shape[::-1], interpolation=interpolation)

    def crop(self, pt_min: Tuple[int, int], pt_max: Tuple[int, int]):
        self.image = self.image[pt_min[0] : pt_max[0], pt_min[1] : pt_max[1]]

    def to_rgb(self):
        if self.c == 1:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)

    def to_grayscale(self):
        if self.c == 3:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

    @classmethod
    def load(cls, filename: Union[str, Path], mode: str = "rgb"):
        color_format = cv2.IMREAD_COLOR
        if mode == "grayscale":
            color_format = cv2.IMREAD_GRAYSCALE
        image = cv2.imread(str(filename), color_format)
        if mode == "rgb":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image(image)


if __name__ == "__main__":

    folder_path = "/path/to/dataset/"

    clf_dataset = ClassificationDataset.from_folder(folder_path)
    clf_dataset.split()
    clf_dataset.summary()

    print(clf_dataset.val_df.head())

    # ml_dataset = MultiLabelDataset()
    # obj_dataset = ObjDetectionDataset()
    # ml_dataset.split()
    # obj_dataset.split()

    clf_dataset.split(test_size=0.2, validation_size=0.2, method="group", seed=41)

    clf_dataset.summary()
