"""
This module is used to train a classification model.

Must configure various parameters using a yaml file.
"""

# import hashlib
from pathlib import Path

import albumentations as A
import mlflow
import numpy as np
import onnx
import tensorflow as tf
import tf2onnx
import yaml

from fast_tfai.callbacks.lr_finder import LRFinder
from fast_tfai.dataset.dataset_builder import Dataset
from fast_tfai.models.model_builder import ModelBuilder
from fast_tfai.utils.console import console, parse_cli
from fast_tfai.utils.validate_args import TrainerConf


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

    def train(self, dataset: Dataset):

        self.input_shape = self.model.input_shape[1:]
        self.class_names = dataset.class_names

        train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            # vertical_flip=True,
            # horizontal_flip=True,
            # width_shift_range=0.01,
            # height_shift_range=0.01,
            # rotation_range=2,
            # zoom_range=0.05,
            # brightness_range=(0.8, 1.2),
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator()
        test_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator()

        train_data = train_datagenerator.flow_from_dataframe(
            dataset.train_df,
            x_col="filename",
            y_col="label_int",
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
            seed=self.seed,
            class_mode="categorical",
            subset="training",
            shuffle=True,
        )

        valid_data = valid_datagenerator.flow_from_dataframe(
            dataset.val_df,
            x_col="filename",
            y_col="label_int",
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
            seed=self.seed,
            class_mode="categorical",
            subset="training",
            shuffle=False,
        )

        test_data = test_datagenerator.flow_from_dataframe(
            dataset.test_df,
            x_col="filename",
            y_col="label_int",
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
            seed=self.seed,
            class_mode="categorical",
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
        if self.learning_rate_finder:
            optimizer = self.find_lr(train_data)

        self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
        self.model.summary()

        epochs = self.epochs
        trainable_backbone = self.trainer_conf.model.params.get(
            "trainable_backbone", False
        )

        if trainable_backbone:
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

        if trainable_backbone:
            self.model.trainable = True
            if self.learning_rate_finder:
                optimizer = self.find_lr(train_data)
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

    def find_lr(self, train_data):
        console.rule("[bold red] LR Finder")
        lr_finder = LRFinder(self.model)
        lr_finder.find(train_data, start_lr=0.00001, end_lr=0.01, epochs=10)
        best_lr = lr_finder.get_best_lr(sma=5)
        console.print("\n[bold red]Best learning rate:", best_lr, "\n")
        optimizer = build_optimizer(
            optimizer_type=self.optimizer_type, learning_rate=best_lr
        )
        return optimizer

    def save(self, heatmap=False, opset=13):
        console.rule("[bold red] Saving models and training history")
        output_folder = Path(self.save_folder) / self.version
        output_folder.mkdir(parents=True, exist_ok=True)
        onnx_output_folder = output_folder / "onnx"
        onnx_output_folder.mkdir(exist_ok=True, parents=True)
        self.onnx_filename = onnx_output_folder / f"{self.version}.onnx"

        with open(onnx_output_folder / "params.yaml", "w") as f:
            yaml.dump(self.trainer_conf.__dict__, f)

        if heatmap:
            model_wh = ModelBuilder.wrap_with_heatmap(self.model)
        else:
            model_wh = self.model

        if self.history:
            np.save(str(output_folder / "history"), self.history)

        model_wh.save(output_folder / "keras")

        input_signature = [
            tf.TensorSpec([None, *model_wh.input_shape[1:]], tf.float32, name="inputs")
        ]
        onnx_model, _ = tf2onnx.convert.from_keras(
            model_wh, input_signature, opset=opset
        )
        onnx_model.doc_string = self.version
        onnx.save(onnx_model, str(self.onnx_filename))

        # with open(self.onnx_filename, "rb") as f:
        #     data = f.read()
        #     computed_md5 = hashlib.md5(data).hexdigest()

    def publish(self):
        console.rule("[bold red] Publishing results to MlFlow")
        mlflow.set_tracking_uri(TrainerDefaultParams.mlflow_url)
        mlflow.set_experiment(self.trainer_conf.name)

        with mlflow.start_run():  # run_name=str(self.trainer_conf.uid)):
            mlflow.set_tag("mlflow.runName", self.version)
            mlflow.log_param("model", self.trainer_conf.model.name)

            config_file = Path(self.save_folder) / self.version / "onnx" / "config.json"
            mlflow.log_artifact(str(config_file))
            params_file = Path(self.save_folder) / self.version / "onnx" / "params.yaml"
            mlflow.log_artifact(str(params_file))

            for k, value in self.results.items():
                mlflow.log_metric(k, value)

            reg_mod_name = None
            if self.trainer_conf.publish.get("register_model"):
                reg_mod_name = self.trainer_conf.name

            mlflow.onnx.log_model(
                onnx_model=onnx.load(self.onnx_filename),
                artifact_path=self.version,
                registered_model_name=reg_mod_name,
            )


def train():

    args = parse_cli()
    cfg = yaml.safe_load(args.conf.read_text())

    trainer_conf = TrainerConf.from_dict(cfg)

    console.rule("[bold red] Configuration")
    console.print(trainer_conf)

    dataset_conf = trainer_conf.dataset
    dataset = Dataset.from_conf(dataset_conf)

    if dataset_conf.split:
        dataset.split(
            validation_size=dataset_conf.validation_size,
            test_size=dataset_conf.test_size,
            method=dataset_conf.method,
            seed=trainer_conf.seed,
        )

    aug_cfg = trainer_conf.dataset.augmentations
    num_aug = aug_cfg.get("num_augmentations", 3)
    if aug_cfg:
        transform = Augmentator(num_augmentations=num_aug).build_transform(aug_cfg)
        dataset.augment(
            transform=transform,
            output_path=dataset_conf.path,
            num_augmentations=num_aug,
        )

    dataset.summary()
    dataset.to_csv(trainer_conf.output_folder)

    model = ModelBuilder.from_conf(trainer_conf)

    trainer = Trainer.from_conf(model=model, conf=trainer_conf)
    trainer.train(dataset)
    trainer.save(heatmap=trainer_conf.model.heatmap)

    if trainer_conf.publish.mlflow and trainer_conf.publish.url:
        trainer.publish()


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


class Augmentator:
    _aug_map = {
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

    def __init__(
        self,
        num_augmentations,
    ) -> None:
        self.num_augmentations = num_augmentations

    def build_transform(self, aug_cfg: dict):
        aug_list = []
        for k, v in aug_cfg.items():
            if k != "num_augmentations":
                v = {} if v is None else v
                aug_list.append(self._aug_map[k](**v))

        transform = A.Compose([A.OneOf(aug_list, p=1)])
        return transform


if __name__ == "__main__":
    train()
