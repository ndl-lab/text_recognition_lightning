# Copyright (c) 2023, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/

from typing import Any, Dict, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as visiontransforms
from pytorch_lightning.trainer.states import TrainerFn

from .ndl_components.ndl_dataset import get_dataset

from .. import utils

log = utils.get_pylogger(__name__)


class NormalizePAD(object):

    def __init__(self, max_size, mean=0.1307, std=0.308, keep_aspect=True):
        self.max_size = max_size
        self.mean = mean
        self.std = std
        self.keep_aspect = keep_aspect

    def __call__(self, img):
        img = visiontransforms.F.resize(img, self.max_size[1])
        if self.keep_aspect or img.width < self.max_size[2]:
            img = visiontransforms.F.to_tensor(img)[:, :self.max_size[1], :self.max_size[2]]
            w = img.size(2)
            tensor = torch.FloatTensor(*self.max_size).fill_(0)
            tensor[:, :, :w] = img  # right pad
        else:
            img = visiontransforms.F.resize(img, (self.max_size[1], self.max_size[2]))
            tensor = visiontransforms.F.to_tensor(img)

        tensor.sub_(self.mean).div_(self.std)

        return tensor


class RandomAspect(torch.nn.Module):
    def __init__(self, max_variation: int):
        super().__init__()
        self.max_variation = max_variation

    @staticmethod
    def get_params(img: torch.Tensor, max_variation: int):
        w, h = F.get_image_size(img)
        w = torch.randint(max(w - max_variation, w // 2), w + max_variation, size=(1,)).item()
        h = torch.randint(max(h - max_variation, h // 2), h + max_variation, size=(1,)).item()
        return w, h

    def forward(self, img):
        w, h = self.get_params(img, self.max_variation)
        return F.resize(img, (h, w))


class RandomPad(torch.nn.Module):
    def __init__(self, max_padding: int, fill=0, padding_mode="constant"):
        super().__init__()
        self.max_padding = max_padding
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img: torch.Tensor, max_padding: int):
        return torch.randint(0, max_padding, size=(4,)).tolist()

    def forward(self, img):
        pad = self.get_params(img, self.max_padding)
        return F.pad(img, pad, fill=self.fill, padding_mode=self.padding_mode)


class DumpTensor(object):
    def __init__(self):
        self.i = 0

    def __call__(self, tensor):
        img = F.to_pil_image(tensor)
        img.save(f"dump/{self.i:03d}-1.png")
        self.i += 1
        return tensor


class NDLDataModule(LightningDataModule):
    """LightningDataModule for NDL dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))
        - predict_dataloader (the predict dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        dataset,
        label_as_str: bool = False,
        batch_max_length: int = 20,
        test_batch_max_length: int = 20,
        max_width: int = 600,
        test_max_width: int = 600,
        height: int = 32,
        swap_bgr2rgb: bool = False,
        batch_size: int = 64,
        additional_elements=None,
        num_workers: int = 0,
        pin_memory: bool = False,
        character_file: str = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        def get_transforms(fill=255, pad=(2, 1), swap_bgr2rgb=swap_bgr2rgb,
                           max_size=(3, height, max_width), mean=0.5, std=0.5,
                           augmentation=False, random_aspect=10, random_pad=10, random_rotate=2):
            ret = [
                # visiontransforms.Grayscale(),
                visiontransforms.Resize(height),
                visiontransforms.Pad(pad, fill=fill),
            ]
            if augmentation:
                ret += [
                    RandomAspect(random_aspect),
                    RandomPad(random_pad, fill=fill),
                    visiontransforms.RandomAffine(degrees=random_rotate, fill=fill),
                ]
            ret += [
                NormalizePAD(max_size, mean=mean, std=std),
                # DumpTensor(),
            ]
            if swap_bgr2rgb:
                ret += [
                    visiontransforms.Lambda(lambda t: t[(2, 1, 0), :, :]),
                ]
            return visiontransforms.Compose(ret)

        # data transformations
        self.transforms = get_transforms(augmentation=True, max_size=(3, height, max_width))
        self.transforms_test = get_transforms(augmentation=False, max_size=(3, height, test_max_width))

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data_pred: Optional[Dataset] = None

        self.image = None
        self.xml = None
        self.pid = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """
        log.info(stage)

        arg = {
            "transforms": self.transforms,
            "batch_max_length": self.hparams.batch_max_length,
            "additional_elements": self.hparams.additional_elements
        }
        arg2 = {
            "transforms": self.transforms_test,
            "batch_max_length": self.hparams.test_batch_max_length,
            "additional_elements": self.hparams.additional_elements
        }

        if self.hparams.character_file is not None:
            with open(self.hparams.character_file, 'r') as f:
                char = f.read().replace('\n', "")
        if stage == TrainerFn.FITTING:
            self.data_train = get_dataset(args=arg, datasets=self.hparams.dataset.train, char=char)
            self.data_val = get_dataset(args=arg2, datasets=self.hparams.dataset.val, char=char)
            self.data_test = get_dataset(args=arg2, datasets=self.hparams.dataset.test, char=char)
        elif stage == TrainerFn.TESTING:
            self.data_test = get_dataset(args=arg2, datasets=self.hparams.dataset.test, char=char)
        elif stage == TrainerFn.PREDICTING:
            self.data_pred = get_dataset(args=arg2, datasets=self.hparams.dataset.pred, char=char)
            for dataset in self.data_pred:
                dataset.set_data(self.image, self.xml, self.pid)

    def set_input_data(self, image, xml, pid):
        self.image = image
        self.xml = xml
        self.pid = pid

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
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
        return [DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False
        ) for dataset in self.data_test]

    def predict_dataloader(self):
        return [DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False
        ) for dataset in self.data_pred]

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self) -> Dict[str, Any]:
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Things to do when loading checkpoint."""
        pass
