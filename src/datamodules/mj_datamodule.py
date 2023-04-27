# Copyright (c) 2023, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/

from typing import Optional, Tuple
import pathlib

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import transforms
from torchvision.datasets.folder import default_loader


class ListDataset(Dataset):
    def __init__(self, dir_path, list_path, transforms, lexicon=None, batch_max_length=20):
        super().__init__()

        self.dir_path = pathlib.Path(dir_path)
        self.transforms = transforms
        self.lexicon = lexicon
        self.batch_max_length = batch_max_length
        self.loader = default_loader

        with open(list_path, 'r') as f:
            self.image_paths = [li.split() for li in f.read().splitlines()]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        p, label = self.image_paths[index]
        label = int(label)

        image = self.loader(self.dir_path / p)

        if self.lexicon is None:
            return self.transforms(image), label
        else:
            return self.transforms(image), self.lexicon[label][:self.batch_max_length]


class MJDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        label_as_str: bool = False,
        batch_max_length: int = 20,
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose([
            transforms.Resize((32, 200)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        with open(pathlib.Path(self.hparams.data_dir) / "lexicon.txt", 'r') as f:
            self.lexicon = [t.lower() for t in f.read().splitlines()]

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return len(self.lexicon)

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        # MNIST(self.hparams.data_dir, train=True, download=True)
        # MNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        if not self.data_train and not self.data_val and not self.data_test:
            dd = pathlib.Path(self.hparams.data_dir)
            lex = self.lexicon if self.hparams.label_as_str else None
            arg = {
                "dir_path": dd, "transforms": self.transforms,
                "lexicon": lex, "batch_max_length": self.hparams.batch_max_length
            }
            self.data_train = ListDataset(list_path=dd / "annotation_train.txt", **arg)
            self.data_test = ListDataset(list_path=dd / "annotation_test.txt", **arg)
            self.data_val = ListDataset(list_path=dd / "annotation_val.txt", **arg)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=not True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=Subset(self.data_test, range(100)),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
