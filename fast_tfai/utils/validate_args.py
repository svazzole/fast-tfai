import random
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid1

import yaml

from fast_tfai.utils.names_generator import get_random_name

NEEDED_KEYS = {"task", "name", "dataset", "model", "train"}
FORMATS = ["folder", "csv"]


@dataclass
class AugmentationsConf:
    ...


@dataclass
class DatasetConf:
    path: Path
    validation_size: float
    test_size: float
    format: str = field(default="folder")
    is_splitted: bool = field(default=False)
    split: bool = field(default=True)
    method: str = field(default="std")
    # TODO: add augmentations?
    augmentations: dict = field(default_factory=lambda: {})

    def __post_init__(self):
        self.path = Path(self.path)
        if self.validation_size < 0 or self.validation_size > 1:
            raise ValueError("dataset 'validation_size' must be in [0,1]")
        if self.test_size < 0 or self.test_size > 1:
            raise ValueError("dataset 'test_size' must be in [0,1]")


@dataclass
class ModelConf:
    name: str
    params: dict
    heatmap: bool = field(default=False)


@dataclass
class TrainConf:
    batch_size: int
    epochs: int = field(default=10)
    optimizer: str = field(default="adam")
    learning_rate: float = field(default=1e-3)
    learning_rate_finder: bool = field(default=False)
    patience: int = field(default=0)


@dataclass
class MLFlowConf:
    url: str = field(default=None)
    mlflow: bool = field(default=False)
    register_model: bool = field(default=False)


@dataclass
class TrainerConf:
    task: str
    name: str
    save_folder: Path
    dataset: DatasetConf
    model: ModelConf
    train: TrainConf

    publish: MLFlowConf

    seed: int = field(default=random.randint(a=0, b=1000000))
    version: str = field(init=False)

    def __post_init__(self):
        self.uid = uuid1()
        mnemonic_uid = get_random_name()
        self.version = mnemonic_uid + "_" + str(self.uid).split("-")[0]
        self.output_folder = Path(self.save_folder) / self.version
        if not self.output_folder.exists():
            self.output_folder.mkdir(exist_ok=True, parents=True)

    @classmethod
    def from_dict(cls, cfg: dict):
        cfg = cfg["trainer"]
        if not NEEDED_KEYS.issubset(cfg.keys()):
            raise ValueError("Some keys in cfg file are missing")

        dataset_conf = DatasetConf(**cfg["dataset"])
        model_conf = ModelConf(**cfg["model"])
        train_conf = TrainConf(**cfg["train"])
        publish_conf = MLFlowConf(**cfg["publish"])

        return TrainerConf(
            task=cfg["task"],
            name=cfg["name"],
            save_folder=cfg["save_folder"],
            seed=cfg["seed"],
            dataset=dataset_conf,
            model=model_conf,
            train=train_conf,
            publish=publish_conf,
        )


if __name__ == "__main__":
    from fast_tfai.utils.console import console, parse_cli

    args = parse_cli()
    conf: dict = yaml.safe_load(args.conf.read_text())
    trainer_conf = TrainerConf.from_dict(conf)

    console.print(trainer_conf)

    # console.print(trainer_conf.dataset.augmentations)
    # console.print(trainer_conf.dataset.path)
    # console.print(trainer_conf.train.__dict__)
