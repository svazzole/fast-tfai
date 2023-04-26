from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from fast_tfai.utils.console import console
from fast_tfai.utils.validate_args import DatasetConf

VALID_FORMATS = ["csv"]


@dataclass(repr=False)
class MultiLabelDataset:
    df: pd.DataFrame
    train_df: pd.DataFrame = field(default=None)
    val_df: pd.DataFrame = field(default=None)
    test_df: pd.DataFrame = field(default=None)

    def __post_init__(self):

        # Get class names and number of classes
        self.full_class_names = sorted([x for x in self.df.columns if x[0] == "class"])
        self.class_names = sorted([x[1] for x in self.df.columns if x[0] == "class"])
        self.num_classes = len(self.class_names)

        # Maps class2int and int2class
        label_int = [str(x) for x in range(len(self.class_names))]
        self.class2int_map = dict(zip(self.class_names, label_int))
        self.int2class_map = {v: k for k, v in self.class2int_map.items()}

    def __repr__(self) -> str:
        out_str = f"Dataset(df: {self.df.shape[0]}"
        if (
            self.train_df is not None
            and self.val_df is not None
            and self.test_df is not None
        ):
            out_str += (
                f", train_df: {self.train_df.shape[0]}, "
                f"val_df: {self.val_df.shape[0]}, test_df: {self.test_df.shape[0]}"
            )
        return out_str + ")"

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

        return cls.from_df(file_path=conf.path, splitted=conf.is_splitted)

    @classmethod
    def from_df(cls, file_path: Union[Path, str], splitted: bool = False):
        file_path = Path(file_path)
        if splitted:
            train_df = pd.read_csv(file_path / "train.csv")
            val_df = pd.read_csv(file_path / "val.df")
            test_df = pd.read_csv(file_path / "test.df")
            df = pd.concat([train_df, val_df, test_df])
            dataset = MultiLabelDataset(df)
            dataset.train_df = train_df
            dataset.val_df = val_df
            dataset.test_df = test_df
            dataset._add_label_int()
        else:
            df: pd.DataFrame = pd.read_csv(file_path / "dataset.csv", header=[0, 1])
            dataset = MultiLabelDataset(df)

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

        console.print(np.sum(self.df["class"]), "\n")

        if (
            (self.train_df is not None)
            and (self.val_df is not None)
            and (self.test_df is not None)
        ):
            console.print(f"Train size: {self.train_df.shape[0]}")
            console.print("Train distribution:")
            console.print(np.sum(self.train_df["class"]), "\n")
            console.print(f"Validation size: {self.val_df.shape[0]}")
            console.print("Validation distribution:")
            console.print(np.sum(self.val_df["class"]), "\n")
            console.print(f"Test size: {self.test_df.shape[0]}")
            console.print("Test distribution:")
            console.print(np.sum(self.test_df["class"]), "\n")
        console.rule()

    def split(
        self,
        test_size: float = 0.2,
        validation_size: float = 0.1,
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

        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=seed
        )

        msss2 = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=validation_size, random_state=seed
        )

        X, y = self.df["image"].to_numpy(), self.df["class"].to_numpy()

        self.X = X
        self.y = y
        for train_index, test_index in msss.split(X, y):
            X_train, X_test = X[train_index, :], X[test_index, :]
            y_train, y_test = y[train_index, :], y[test_index, :]

        for train_index, test_index in msss2.split(X_train, y_train):
            X_train, X_val = (
                X_train[train_index, :],
                X_train[test_index, :],
            )
            y_train, y_val = (
                y_train[train_index, :],
                y_train[test_index, :],
            )

        self.train_df = pd.concat(
            [pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1
        )
        self.val_df = pd.concat([pd.DataFrame(X_val), pd.DataFrame(y_val)], axis=1)
        self.test_df = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis=1)

        self.train_df.columns = self.df.columns
        self.val_df.columns = self.df.columns
        self.test_df.columns = self.df.columns
