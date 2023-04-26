"""
This module is used to train a classification model.

Must configure various parameters using a yaml file.
"""

# import hashlib
from pathlib import Path
from typing import Union

import albumentations as A
import mlflow
import numpy as np
import onnx
import tensorflow as tf
import tf2onnx
import yaml

from fast_tfai.callbacks.lr_finder import LRFinder
from fast_tfai.dataset.dataset_builder import Dataset
from fast_tfai.dataset.multilabel_dataset import MultiLabelDataset
from fast_tfai.models.model_builder import ModelBuilder
from fast_tfai.utils.console import console, parse_cli
from fast_tfai.utils.validate_args import TrainerConf

# from cv2 import imwrite
# import pandas as pd
# from tqdm import tqdm


def build_optimizer(
    optimizer_type: str = "adam", learning_rate: float = 1e-3
) -> tf.keras.optimizers.Optimizer:
    if optimizer_type == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_type == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer_type == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError

    return optimizer


class TrainerDefaultParams:
    mlflow_url = "http://10.1.2.100:5000"


class Trainer:
    def __init__(
        self,
        model: tf.keras.models.Model,
        task: str,
        name: str,
        save_folder: Path,
        batch_size: int,
        seed: int,
        optimizer: str,
        learning_rate: float,
        learning_rate_finder: bool,
        patience: int,
        epochs: int,
        trainer_conf: TrainerConf,
    ) -> None:
        self.model = model
        self.task = task
        self.name = name
        self.save_folder = save_folder
        self.batch_size = batch_size
        self.seed = seed
        self.optimizer_type = optimizer
        self.learning_rate = learning_rate
        self.learning_rate_finder = learning_rate_finder
        self.patience = patience
        self.epochs = epochs
        self.class_names = []
        self.trainer_conf: TrainerConf = trainer_conf
        self.version = self.trainer_conf.version

    @classmethod
    def from_conf(cls, model: tf.keras.Model, conf: TrainerConf):
        return Trainer(
            model=model,
            task=conf.task,
            name=conf.name,
            save_folder=conf.save_folder,
            trainer_conf=conf,
            seed=conf.seed,
            **conf.train.__dict__,
        )

    def train(self, dataset: Union[Dataset, MultiLabelDataset]):

        self.input_shape = self.model.input_shape[1:]
        self.class_names = dataset.class_names

        train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator()
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator()
        test_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator()

        train_data = train_datagenerator.flow_from_dataframe(
            dataset.train_df,
            x_col=("image", "path"),
            y_col=dataset.full_class_names,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
            seed=self.seed,
            class_mode="raw",
            subset="training",
            shuffle=True,
        )

        valid_data = valid_datagenerator.flow_from_dataframe(
            dataset.val_df,
            x_col=("image", "path"),
            y_col=dataset.full_class_names,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
            seed=self.seed,
            class_mode="raw",
            subset="training",
            shuffle=False,
        )

        test_data = test_datagenerator.flow_from_dataframe(
            dataset.test_df,
            x_col=("image", "path"),
            y_col=dataset.full_class_names,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
            seed=self.seed,
            class_mode="raw",
            subset="training",
            shuffle=False,
        )

        lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.8, patience=10, verbose=1, mode="min"
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            patience=self.patience, restore_best_weights=True
        )

        optimizer = build_optimizer(
            optimizer_type=self.optimizer_type, learning_rate=self.learning_rate
        )
        loss_fn = (
            tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2)
            # tf.keras.losses.BinaryFocalCrossentropy()
        )
        metrics = [
            tf.keras.metrics.CategoricalAccuracy(name="acc"),
            tf.keras.metrics.AUC(name="prc", curve="PR"),  # precision-recall curve
        ]

        self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
        self.model.summary()

        if self.learning_rate_finder:
            console.rule("[bold red] LR Finder")
            lr_finder = LRFinder(self.model)
            lr_finder.find(train_data, start_lr=0.00001, end_lr=0.01, epochs=10)
            best_lr = lr_finder.get_best_lr(sma=5)
            console.print("\n[bold red]Best learning rate:", best_lr, "\n")
            optimizer = build_optimizer(
                optimizer_type=self.optimizer_type, learning_rate=best_lr
            )
            self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

        epochs = self.epochs
        if self.trainer_conf.model.params.get("trainable_backbone", False):
            epochs = (self.epochs // 5) * 2

        console.rule("[bold red] Training")
        self.history = self.model.fit(
            train_data,
            batch_size=self.batch_size,
            epochs=epochs,
            validation_data=valid_data,
            class_weight=dataset.class_weights,
            callbacks=[
                early_stopping,
                lr_reducer,
            ],
        )

        if self.trainer_conf.model.params.get("trainable_backbone", False):
            self.model.trainable = True
            self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

            self.history = self.model.fit(
                train_data,
                batch_size=self.batch_size,
                epochs=(self.epochs // 5) * 3,
                validation_data=valid_data,
                class_weight=dataset.class_weights,
                callbacks=[
                    early_stopping,
                    lr_reducer,
                ],
            )

        console.rule("[bold red] Test Evaluation")
        self.results = self.model.evaluate(test_data, return_dict=True)
        console.print(self.results)

    def _save(self, model: tf.keras.Model, debug=False, opset=13):

        output_folder = Path(self.save_folder) / self.version
        if debug:
            output_folder = output_folder / "debug"

        output_folder.mkdir(parents=True, exist_ok=True)
        if self.history:
            np.save(str(output_folder / f"{self.version}.history"), self.history)

        model.save(output_folder / "keras")

        input_signature = [
            tf.TensorSpec([None, *model.input_shape[1:]], tf.float32, name="inputs")
        ]
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=opset)
        onnx_model.doc_string = self.version
        onnx_output_folder = output_folder / "onnx"
        onnx_output_folder.mkdir(exist_ok=True, parents=True)
        self.onnx_filename = onnx_output_folder / f"{self.version}.onnx"
        onnx.save(onnx_model, str(self.onnx_filename))

        # with open(self.onnx_filename, "rb") as f:
        #     data = f.read()
        #     computed_md5 = hashlib.md5(data).hexdigest()

    def save(self, heatmap=False, opset=13):
        console.rule("[bold red] Saving models and training history")
        self._save(self.model, opset=opset)
        if heatmap:
            model_wh = ModelBuilder.wrap_with_heatmap(self.model)
            self._save(model_wh, debug=True, opset=opset)

    def publish(self):
        console.rule("[bold red] Publishing results to MlFlow")
        mlflow.set_tracking_uri(TrainerDefaultParams.mlflow_url)
        mlflow.set_experiment(self.trainer_conf.name)

        with mlflow.start_run():  # run_name=str(self.trainer_conf.uid)):
            mlflow.set_tag("mlflow.runName", self.version)
            mlflow.log_param("model", self.trainer_conf.model.name)

            config_file = Path(self.save_folder) / self.version / "onnx" / "config.json"
            mlflow.log_artifact(str(config_file))

            for k, value in self.results.items():
                mlflow.log_metric(k, value)

            if self.trainer_conf.publish.get("register_model"):
                mlflow.onnx.log_model(
                    onnx_model=onnx.load(self.onnx_filename),
                    artifact_path=self.version,
                    registered_model_name=self.trainer_conf.name,
                )


class Evaluator:
    def __init__(self, model_folder: Path, version: str):
        self.model_folder: Path = Path(model_folder)
        self.version: str = version
        self.model: tf.keras.Model = None
        self.results = {}
        self._load_model()

    def _load_model(self):
        full_path = self.model_folder / self.version
        self.model = tf.keras.models.load_model(full_path)

    def __call__(self, data):
        self.evaluate(data)

    def evaluate(self, data):
        self.results = self.model.evaluate(data, return_dict=True)

    def publish(self):
        console.print(TrainerDefaultParams.mlflow_url)


def train():

    args = parse_cli()
    cfg = yaml.safe_load(args.conf.read_text())

    trainer_conf = TrainerConf.from_dict(cfg)

    console.rule("[bold red] Configuration")
    console.print(trainer_conf)

    dataset_conf = trainer_conf.dataset
    dataset = MultiLabelDataset.from_conf(dataset_conf)

    if dataset_conf.split:
        dataset.split(
            validation_size=dataset_conf.validation_size,
            test_size=dataset_conf.test_size,
            seed=trainer_conf.seed,
        )

    aug_cfg = trainer_conf.dataset.augmentations
    if aug_cfg:
        transform = build_transform(aug_cfg)
        dataset.augment(
            transform=transform, output_path=dataset_conf.path, num_augmentations=10
        )

    dataset.summary()
    dataset.to_csv(trainer_conf.output_folder)

    model = ModelBuilder.from_conf(trainer_conf, dataset.num_classes)

    trainer = Trainer.from_conf(model=model, conf=trainer_conf)
    trainer.train(dataset)
    trainer.save(heatmap=True)

    if trainer_conf.publish.get("mlflow", False):
        trainer.publish()


# def preprocess(dataset: MultiLabelDataset, source_dir, output_dir: Union[str, Path]):
#     output_dir = Path(output_dir)
#     for part_name in ["train", "valid", "test"]:
#         if part_name == "train":
#             part_data = dataset.train_df
#         elif part_name == "val":
#             part_data = dataset.val_df
#         else:
#             part_data = dataset.test_df

#         part_data.iloc[:, 2:] = part_data.iloc[:, 2:].astype(str)  # labels
#         part_data.sort_values(by=["image", "frame"], inplace=True)
#         part_data["frame_image"] = ""
#         frames_path = list()
#         for sample_path, sample_rows in tqdm(part_data.groupby("image"), part_name):
#             sample_path: str
#             sample_rows: pd.DataFrame
#             image_path = source_dir / sample_path
#             frames_path = list()
#             if ("generated" in sample_path) or ("gan" in sample_path):
#                 token_data = image_load(image_path=image_path, mode="IMREAD_COLOR")[
#                     "image_data"
#                 ]
#                 output_subdir = image_path.relative_to(source_dir).parent
#                 output_path = (
#                     output_dir / output_subdir / image_path.with_suffix(".png").name
#                 )
#                 output_path.parent.mkdir(exist_ok=True, parents=True)
#                 imwrite(str(output_path), np.array(token_data, dtype=np.uint8))
#                 frames_path.append(output_path)
#                 part_data.loc[sample_rows.index, "frame_image"] = frames_path
#             else:
#                 frames = preprocess(image_path=image_path)
#                 for frame_id in sample_rows["frame"]:
#                     frame = frames[int(frame_id)]
#                     output_subdir = image_path.relative_to(source_dir).parent
#                     output_name = f"{image_path.stem}_{int(frame_id)}.png"
#                     output_path = output_dir / output_subdir / output_name
#                     output_path.parent.mkdir(exist_ok=True, parents=True)
#                     imwrite(str(output_path), frame["image_data"])
#                     frames_path.append(output_path)
#                 part_data.loc[sample_rows.index, "frame_image"] = frames_path
#             part_data.to_csv(output_dir / f"{part_name}.csv", index=False, header=None)


def build_transform(aug_cfg: dict):

    aug_list = []
    aug_map = {
        "RandomRotate90": A.RandomRotate90,
        "ColorJitter": A.ColorJitter,
        "RandomBrightnessContrast": A.RandomBrightnessContrast,
        "ImageCompression": A.ImageCompression,
        "ISONoise": A.ISONoise,
        "PixelDropout": A.PixelDropout,
        "HorizontalFlip": A.HorizontalFlip,
        "VerticalFlip": A.VerticalFlip,
        "SafeRotate": A.SafeRotate,
        "Affine": A.Affine,
        "CropAndPad": A.CropAndPad,
    }

    for k, v in aug_cfg.items():
        v = {} if v is None else v
        aug_list.append(aug_map[k](**v))

    transform = A.Compose([A.OneOf(aug_list, p=1)])
    return transform


if __name__ == "__main__":
    train()
