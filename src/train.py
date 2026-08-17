""""hydra entrypoint""""
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

torch.set_float32_matmul_precision("medium")
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.retrieval_module import _RetrievalBuffer
from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def _instantiate_datamodules(
    cfg_group: Optional[DictConfig], stage: str
) -> Dict[str, LightningDataModule]:
    """Instantiate a dict of named datamodules from a ``cfg.data.<group>`` node.

    Returns ``{}`` if the group is missing or empty.  Each datamodule's
    ``.setup(stage=...)`` is invoked eagerly so we can construct dataloaders
    directly in ``train`` below.
    """
    if not cfg_group:
        return {}
    dms: Dict[str, LightningDataModule] = {}
    for name, dm_cfg in cfg_group.items():
        log.info(f"Instantiating {stage} datamodule '{name}' <{dm_cfg._target_}>")
        dm = hydra.utils.instantiate(dm_cfg)
        dm.setup(stage=stage)
        dms[name] = dm
    return dms


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Train (and optionally test) the model.

    Data layout (required):
        cfg.data.train               single LightningDataModule config
        cfg.data.val.<name>          dict of named LightningDataModule configs
        cfg.data.test.<name>         dict of named LightningDataModule configs (may be empty)
    """
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    if "data" not in cfg or "train" not in cfg.data:
        raise RuntimeError(
            "cfg.data.train is required.  Experiment configs must declare "
            "train/val/test datamodules via the `@data.train` / `@data.val.<name>` / "
            "`@data.test.<name>` Hydra package syntax."
        )

    log.info(f"Instantiating train datamodule <{cfg.data.train._target_}>")
    train_dm: LightningDataModule = hydra.utils.instantiate(cfg.data.train)
    num_devices = int(cfg.trainer.get("devices", 1))
    num_nodes = int(cfg.trainer.get("num_nodes", 1))
    train_dm.target_world_size = num_devices * num_nodes
    train_dm.setup(stage="fit")

    val_dms = _instantiate_datamodules(cfg.data.get("val"), stage="validate")
    test_dms = _instantiate_datamodules(cfg.data.get("test"), stage="test")

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    expected_world = trainer.num_devices * trainer.num_nodes
    if trainer.world_size != expected_world:
        raise RuntimeError(
            f"LAUNCH GEOMETRY MISMATCH: {trainer.world_size} launched process(es) but the config "
            f"plans devices*nodes = {expected_world}. Fix the launcher (srun needs "
            f"--ntasks-per-node={trainer.num_devices}); refusing to train at the wrong world size."
        )

    object_dict = {
        "cfg": cfg,
        "train_datamodule": train_dm,
        "val_datamodules": val_dms,
        "test_datamodules": test_dms,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        ckpt_dir = cfg.paths.output_dir + "/checkpoints"
        log.info(f"Checkpoints will be saved to: {ckpt_dir}")
        log.info(
            f"Training on '{list({cfg.data.train._target_})}'; "
            f"val on {list(val_dms)}; test on {list(test_dms)}"
        )
        trainer.fit(
            model=model,
            train_dataloaders=train_dm.train_dataloader(),
            val_dataloaders=[dm.val_dataloader() for dm in val_dms.values()] or None,
            ckpt_path=cfg.get("ckpt_path"),
        )

    train_metrics = trainer.callback_metrics

    if cfg.get("test") and test_dms:
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(
            model=model,
            dataloaders=[dm.test_dataloader() for dm in test_dms.values()] or None,
            ckpt_path=ckpt_path,
        )
        log.info(f"Best ckpt path: {ckpt_path}")
    elif cfg.get("test"):
        log.info("Skipping test phase: cfg.data.test is empty.")

    test_metrics = trainer.callback_metrics
    metric_dict = {**train_metrics, **test_metrics}
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    extras(cfg)
    metric_dict, _ = train(cfg)
    return get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))


if __name__ == "__main__":
    main()
