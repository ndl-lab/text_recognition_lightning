# Copyright (c) 2023, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/

import torch
from pytorch_lightning.callbacks import Callback


class MeasureTimeCallback(Callback):
    def __init__(self):
        self.ready = torch.cuda.Event(enable_timing=True)
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.t1 = torch.cuda.Event(enable_timing=True)
        self.t2 = torch.cuda.Event(enable_timing=True)

        self.pids = set()
        self.time_batch = list()

    def setup(self, trainer, pl_module, stage):
        self.ready.record()

    def on_predict_start(self, trainer, pl_module):
        self.start.record()

    def on_predict_epoch_end(self, trainer, pl_module, outputs):
        outputs = outputs[0]

        for output in outputs:
            self.pids.update(output['pid'])

    def on_predict_batch_start(self, *a, **b):
        self.t1.record()

    def on_predict_batch_end(self, *a, **b):
        self.t2.record()
        torch.cuda.synchronize()
        self.time_batch.append(self.t1.elapsed_time(self.t2))

    def on_predict_end(self, trainer, pl_module):
        self.end.record()
        torch.cuda.synchronize()
        elapsed_time = self.start.elapsed_time(self.end) / 1000

        print('-----------------------')
        print(f"all elapsed time: {self.ready.elapsed_time(self.end)/1000:0.3f} [sec]")
        print(f"  inference time: {elapsed_time:0.3f} [sec]")
        if self.pids:
            print(f"        per page: {elapsed_time / len(self.pids) / 2:0.3f} [sec]")
        print(f"      num of pid: {len(self.pids)}")
        if self.time_batch:
            print("inference time")
            print(f"   batch average: {sum(self.time_batch) / len(self.time_batch) / 1000:0.3f} [sec]")
        print('-----------------------')
