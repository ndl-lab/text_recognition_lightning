# Copyright (c) 2023, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/

import hydra
import pyrootutils
from omegaconf import DictConfig

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="rf.yaml")
def main(cfg: DictConfig) -> None:

    from src.tasks.rf_task import main_task

    main_task(cfg)


if __name__ == "__main__":
    main()
