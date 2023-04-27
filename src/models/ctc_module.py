# Copyright (c) 2023, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/

from typing import Any, List
import string
import pathlib

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric
from torchmetrics.text import CharErrorRate
from torch import nn

from .components.ctclabelconverter import CTCLabelConverter
from .components.resnet import OrientationClassifier


class CTCLitModule(LightningModule):
    """LightningModule for CTC.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        seq: torch.nn.Module = None,
        pred: str = "CTC",
        lr: float = 1.0,
        weight_decay: float = 0.0,
        batch_max_length: int = 20,
        test_batch_max_length: int = 20,
        character: string = string.digits + string.ascii_lowercase,
        character_file: string = None,
        ambiguous_char: string = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        if character_file is not None:
            assert pathlib.Path(character_file).exists()
            with open(character_file, 'r') as f:
                character = f.read().replace('\n', "")
        self.converter = CTCLabelConverter(character, ambiguous_char)

        self.FeatureExtraction = net
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1
        if pred == 'CTC':
            self.Prediction = nn.Linear(self.FeatureExtraction.output_size, len(self.converter.character))

        # loss function
        self.criterion = torch.nn.CTCLoss(zero_infinity=True)

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.val_ed = CharErrorRate()
        self.test_ed = CharErrorRate()
        self.i = 0

        # for logging best so far validation accuracy
        self.val_ed_best = MinMetric()

    def forward(self, x: torch.Tensor):
        visual_feature1, _ = self.FeatureExtraction(x)
        visual_feature = self.AdaptiveAvgPool(visual_feature1.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        prediction = self.Prediction(contextual_feature.contiguous())

        return prediction

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_ed_best.reset()

    def step(self, batch: Any, batch_max_length):
        x, y, d = batch
        batch_size = x.size(0)

        preds = self.forward(x)

        text, length = self.converter.encode(y, batch_max_length=batch_max_length)

        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        loss = self.criterion(preds.log_softmax(2).permute(1, 0, 2), text, preds_size, length)

        return {
            "loss": loss
        }, preds, y, d

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, data = self.step(batch, self.hparams.batch_max_length)

        batch_size = preds.size(0)
        for k, v in loss.items():
            self.log(f"train/{k}", v, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!

        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, data = self.step(batch, self.hparams.test_batch_max_length)

        batch_size = preds.size(0)
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        _, preds_index = preds.max(2)
        text = self.converter.decode(preds_index.data, preds_size.data)
        # if batch_idx == 0:
        #     for i, v in enumerate(zip(text, targets)):
        #         print(f"val sample{i}: {'o' if v[0] == v[1] else 'x'}: gt/pred-> {v[1]} {v[0]}")
        #         if i >= 9:
        #             break

        # log val metrics
        self.val_ed.update(text, targets)
        for k, v in loss.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

        return loss

    def validation_epoch_end(self, outputs: List[Any]):
        self.val_ed_best.update(self.val_ed.compute())
        self.log("val/editdistance", self.val_ed.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ed_best", self.val_ed_best.compute(), on_epoch=True, prog_bar=True)
        self.val_ed.reset()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        loss, preds, targets, data = self.step(batch, self.hparams.test_batch_max_length)

        batch_size = preds.size(0)
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        _, preds_index = preds.max(2)
        text = self.converter.decode(preds_index.data, preds_size.data)

        # log test metrics
        self.test_ed.update(text, targets)
        self.log("editdistance", self.test_ed, on_epoch=True)

        return {"loss": loss}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        loss, preds, targets, data = self.step(batch, self.hparams.test_batch_max_length)

        batch_size = preds.size(0)
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        _, preds_index = preds.max(2)
        text = self.converter.decode(preds_index.data, preds_size.data)

        return {"text": text, **data}

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx: int = 0):
        if batch_idx == 0:
            self.test_ed.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        # return torch.optim.Adam(
        #     params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        # )
        return torch.optim.Adadelta(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay,
            rho=0.95, eps=1e-8,
        )


class OrientCTCLitModule(CTCLitModule):
    def __init__(
        self,
        net: torch.nn.Module,
        pred: str = "CTC",
        **kwargv,
    ):
        super().__init__(net, **kwargv)

        if pred == 'CTC':
            self.Prediction = nn.Linear(self.FeatureExtraction.output_size + 1, len(self.converter.character))

        self.OrientationClassifier = OrientationClassifier(net.output_size * 4)

        self.criterion_orient = torch.nn.BCELoss()

    def forward(self, x: torch.Tensor):
        visual_feature1, visual_feature2 = self.FeatureExtraction(x)
        visual_feature = self.AdaptiveAvgPool(visual_feature1.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        orient = self.OrientationClassifier(visual_feature2)
        orient_ex = orient.expand(contextual_feature.size()[:2]).unsqueeze(2)
        contextual_feature = torch.concat((contextual_feature, orient_ex), dim=2)

        prediction = self.Prediction(contextual_feature.contiguous())

        return prediction, orient

    def step(self, batch: Any, batch_max_length):
        x, y, d = batch
        batch_size = x.size(0)

        preds, orient = self.forward(x)

        text, length = self.converter.encode(y, batch_max_length=batch_max_length)

        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        loss = self.criterion(preds.log_softmax(2).permute(1, 0, 2), text, preds_size, length)
        loss_orient = self.criterion_orient(torch.sigmoid(orient).reshape(-1), d['orient'].to(torch.float))

        return {
            "loss": loss + loss_orient,
            "loss_ctc": loss,
            "loss_orient": loss_orient
        }, preds, y, d


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
    _ = hydra.utils.instantiate(cfg)
