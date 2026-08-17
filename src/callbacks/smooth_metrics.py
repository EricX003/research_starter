"""Callback that logs exponentially smoothed copies of the training metrics, so noisy
step-level curves stay readable."""
from __future__ import annotations

import lightning as L
from lightning.pytorch.callbacks import Callback


class SmoothMetricsCallback(Callback):
    """Logs EMA-smoothed training metrics under a 'smooth/' WandB panel.

    At every training step, for each fresh step-level metric (keys matching
    'train/*_step' in callback_metrics), maintains an exponential moving average
    and logs it as 'smooth/<metric_name>'.

    On every validation start the EMA buffer is wiped, so the smooth trace for
    each training segment begins fresh.  This makes it easy to see within-epoch
    trends without cross-epoch contamination.

    Args:
        decay: EMA decay factor.  Higher = smoother / slower to react.
               0.98 ≈ effective window of ~50 steps.
               0.9  ≈ effective window of ~10 steps.
    """

    def __init__(self, decay: float = 0.98):
        self.decay = decay
        self._ema: dict[str, float] = {}

    # ── Training ──────────────────────────────────────────────────────────────

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        new_vals: dict[str, float] = {}
        for key, val in trainer.callback_metrics.items():
            if key.startswith("train/") and key.endswith("_step"):
                try:
                    new_vals[key] = float(val)
                except (TypeError, ValueError):
                    pass

        if not new_vals:
            return

        for key, v in new_vals.items():
            if key not in self._ema:
                self._ema[key] = v              # cold-start: seed with first observed value
            else:
                self._ema[key] = self.decay * self._ema[key] + (1.0 - self.decay) * v

        # Log under smooth/<name> — strip "train/" prefix and "_step" suffix
        smooth = {
            f"smooth/{key[6:-5]}": ema_val
            for key, ema_val in self._ema.items()
        }
        pl_module.log_dict(
            smooth,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            batch_size=batch["pixel_values"].shape[0],
        )

    # ── Validation ────────────────────────────────────────────────────────────

    def on_validation_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Reset the EMA buffer so each training segment starts clean."""
        self._ema.clear()
