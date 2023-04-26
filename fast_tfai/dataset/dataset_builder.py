import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from rich.progress import track
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from fast_tfai.utils.console import console
from fast_tfai.utils.utils import get_all_images
from fast_tfai.utils.validate_args import DatasetConf

VALID_FORMATS = ["folder", "csv"]


@dataclass(repr=False)
class Dataset:
    df: pd.DataFrame
    train_df: pd.DataFrame = field(default=None)
    val_df: pd.DataFrame = field(default=None)
    test_df: pd.DataFrame = field(default=None)

    def __post_init__(self):

        # Check if the DataFrame has the right columns
        if not set(["label", "filename"]).issubset(self.df.columns):
            raise ValueError(
                "The input dataframe has not 'filename' or 'label' column."
            )

        # Get class names and number of classes
        self.class_names = sorted(self.df["label"].unique())
        self.num_classes = len(self.class_names)

        # Maps class2int and int2class
        label_int = [str(x) for x in range(len(self.class_names))]
        self.class2int_map = dict(zip(self.class_names, label_int))
        self.int2class_map = {v: k for k, v in self.class2int_map.items()}

        # Add integer version of the 'label' col
        self.df["label_int"] = self.df["label"].replace(self.class2int_map)

        self.extract_ids()

        # Find class weights
        # TODO: check if this must be done here or after the augmentations
        #       (one could decide to augment only part of the classes)
        # self.class_weights = self._find_class_weights()

    def __repr__(self) -> str:
        out_str = f"Dataset(df: {self.df.shape[0]}"
        if (
            self.train_df is not None
            and self.test_df is not None
            and self.val_df is not None
        ):
            out_str += (
                f", train_df: {self.train_df.shape[0]}, "
                f"val_df: {self.val_df.shape[0]}, test_df: {self.test_df.shape[0]}"
            )
        return out_str + ")"

    def _add_label_int(self):
        datasets: List[pd.DataFrame] = [self.train_df, self.val_df, self.test_df]
        for df in datasets:
            if "label_int" not in df.columns:
                df["label_int"] = df["label"].replace(self.class2int_map)

    @staticmethod
    def build_df(
        images_list: List[Path],
    ) -> pd.DataFrame:
        """Extract filenames and labels (based on the path).

        Args:
            images_list (List[Path]): a list containing all the filenames of the images.

        Returns:
            pd.DataFrame: a pandas DataFrame of the filenames and the list of the labels.
                The labels are inferred from the name of the last directory in the path.
        """
        filenames = [str(x) for x in images_list]
        labels = [x.parts[-2] for x in images_list]
        return pd.DataFrame({"filename": filenames, "label": labels})

    @classmethod
    def from_conf(cls, conf: DatasetConf):
        if conf.format not in VALID_FORMATS:
            raise ValueError(f"dataset 'format' must be in {VALID_FORMATS}")

        if conf.format == "folder":
            return cls.from_folder(folder=conf.path, splitted=conf.is_splitted)
        elif conf.format == "csv":
            return cls.from_df(file_path=conf.path, splitted=conf.is_splitted)

    @classmethod
    def from_folder(
        cls, folder: Path, classes: List[str] = None, splitted: bool = False
    ):
        """Build a dataset object from a folder.

        Args:
            folder (Path): the folder containing all the images. The images can be divided
                into subfolders in two ways:
                1) folder
                    |-- class1
                    |-- class2
                    ...
                    |-- classN
                2) folder
                    |-- train
                        |-- class1
                        ...
                        |-- classN
                    |-- val
                        |-- class1
                        ...
                        |-- classN
                    |-- test
                        |-- class1
                        ...
                        |-- classN
            splitted (bool, optional): set it equal to True if you are in 2) (see above).
                Defaults to False.

        Returns:
            _type_: _description_
        """
        folder = Path(folder)
        if splitted:
            train_images_list = get_all_images(folder / "train")
            val_images_list = get_all_images(folder / "val")
            test_images_list = get_all_images(folder / "test")

            train_df = Dataset.build_df(train_images_list)
            val_df = Dataset.build_df(val_images_list)
            test_df = Dataset.build_df(test_images_list)

            df = pd.concat([train_df, val_df, test_df])
            dataset = Dataset(df)

            dataset.train_df = train_df
            dataset.val_df = val_df
            dataset.test_df = test_df
            dataset._add_label_int()

        else:
            images_list = []
            if classes:
                for cl in classes:
                    images_list += get_all_images(folder / cl)
            else:
                subdirs = [x.name for x in folder.iterdir() if x.name != "augmented"]
                for subd in subdirs:
                    images_list += get_all_images(folder / subd)
            dataset = Dataset(Dataset.build_df(images_list))
        return dataset

    @classmethod
    def from_df(cls, file_path: Union[Path, str], splitted: bool = False):
        file_path = Path(file_path)
        if splitted:
            train_df = pd.read_csv(file_path / "train.csv")
            val_df = pd.read_csv(file_path / "val.df")
            test_df = pd.read_csv(file_path / "test.df")
            df = pd.concat([train_df, val_df, test_df])
            dataset = Dataset(df)
            dataset.train_df = train_df
            dataset.val_df = val_df
            dataset.test_df = test_df
            dataset._add_label_int()
        else:
            if (file_path / "dataset.csv").exists():
                df: pd.DataFrame = pd.read_csv(file_path / "dataset.csv")
            else:
                csv_files = list(file_path.rglob("*.csv"))
                df: pd.DataFrame = pd.read_csv(csv_files[0])
                df.rename(columns={"image": "filename"}, inplace=True)
                df["filename"] = [str(file_path / x) for x in df["filename"]]

            dataset = Dataset(df)

        return dataset

    def to_csv(self, output_path: Union[str, Path]):
        """Write the dataset to csv files.

        Args:
            output_path (Union[str, Path]): a string or a Path for the output folder.
        """
        output_path = Path(output_path)
        self.df.to_csv(str(output_path / "df.csv"), index=False)
        if self.train_df is not None:
            self.train_df.to_csv(str(output_path / "train.csv"), index=False)
        if self.val_df is not None:
            self.val_df.to_csv(str(output_path / "val.csv"), index=False)
        if self.test_df is not None:
            self.test_df.to_csv(str(output_path / "test.csv"), index=False)

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

    def extract_ids(self):
        sample = []
        for x in self.df["filename"]:
            if "_frame" in x:
                sample.append(Path(x).name.split("_frame")[0])
            else:
                sample.append("_".join(Path(x).name.split("_")[:-1]))
        self.df["sample"] = sample

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

        X, y = self.df["filename"].values, self.df["label"].values
        groups = self.df["sample"].values

        k = int(1 / test_size)
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        train_index, test_index = next(sgkf.split(X, y, groups))

        self.train_df = self.df.iloc[train_index]
        self.test_df = self.df.iloc[test_index]
        self.train_df.reset_index()
        self.test_df.reset_index()

        k = int(1 / validation_size)
        sgkf2 = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)

        X, y = self.train_df["filename"].values, self.train_df["label"].values
        groups = self.train_df["sample"].values
        train_index, val_index = next(sgkf2.split(X, y, groups))

        self.val_df = self.train_df.iloc[val_index]
        self.train_df = self.train_df.iloc[train_index]

        self.train_df.reset_index(inplace=True)
        self.val_df.reset_index(inplace=True)
        self.test_df.reset_index(inplace=True)

    def _find_class_weights(self):
        # Scaling by total/num_classes helps keep the loss to a similar magnitude.
        # The sum of the weights of all examples stays the same.

        counts = self.df["label_int"].value_counts()
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

    def augment(
        self,
        output_path: Union[str, Path],
        transform: A.Compose,
        num_augmentations: int = 3,
    ) -> None:

        console.rule("[bold red] Augmentations")

        # TODO: rewrite this piece of code. Merge train and validation in a single
        #       algo?
        output_path = Path(output_path) / "augmented"
        if output_path.exists():
            # we remove the "augmented" sub-folder
            shutil.rmtree(output_path, ignore_errors=True)
        output_path.mkdir(exist_ok=True, parents=True)

        images = zip(self.train_df["filename"].values, self.train_df["label"].values)
        images_val = zip(self.val_df["filename"].values, self.val_df["label"].values)

        new_images, new_labels, new_labels_int = [], [], []
        for img_fn, lbl in track(
            images,
            description="Augmenting training images...",
            total=self.train_df.shape[0],
        ):
            for i in range(num_augmentations):
                img = cv2.imread(img_fn, cv2.IMREAD_COLOR)
                transformed = transform(image=img)
                new_img = transformed["image"]
                new_fn = output_path / f"{Path(img_fn).stem}_augmented_{i}.png"
                cv2.imwrite(str(new_fn), new_img)
                new_images.append(str(new_fn))
                new_labels.append(lbl)
                if lbl == "good":
                    new_labels_int.append(str(0))
                else:
                    new_labels_int.append(str(1))

        augm_images_df = pd.DataFrame(
            {
                "filename": new_images,
                "label": new_labels,
                "label_int": new_labels_int,
            }
        )

        self.train_df = pd.concat([self.train_df, augm_images_df])

        new_images, new_labels, new_labels_int = [], [], []
        for img_fn, lbl in track(
            images_val,
            description="Augmenting validation images...",
            total=self.val_df.shape[0],
        ):
            for i in range(num_augmentations):
                img = cv2.imread(img_fn, cv2.IMREAD_COLOR)
                transformed = transform(image=img)
                new_img = transformed["image"]
                new_fn = output_path / f"{Path(img_fn).stem}_augmented_{i}.png"
                cv2.imwrite(str(new_fn), new_img)
                new_images.append(str(new_fn))
                new_labels.append(lbl)
                if lbl == "good":
                    new_labels_int.append(str(0))
                else:
                    new_labels_int.append(str(1))

        augm_images_df = pd.DataFrame(
            {
                "filename": new_images,
                "label": new_labels,
                "label_int": new_labels_int,
            }
        )

        self.val_df = pd.concat([self.val_df, augm_images_df])

        # Recompute class weights
        self.class_weights = self._find_class_weights()


if __name__ == "__main__":

    # Test 1
    path = Path("/path/to/dataset/")
    data = Dataset.from_folder(path)
    # console.print(data)

    output_path = Path("./outputs")
    data.split(validation_size=0.3, test_size=0.3)
    data.to_csv(output_path)
    # console.print(data)

    data.summary()

    console.print(data.train_df.head())
    # Test 2
    # path = Path("/path/to/dataset/")
    # data = Dataset.from_folder(path)
    # console.print(data)

    # output_path = Path("./outputs")
    # data.split()
    # data.to_csv(output_path)

    # console.print(data)
