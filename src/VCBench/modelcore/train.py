
import logging
import argparse
import sys
import os
import glob
from typing import List, Optional
import multiprocessing
if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method('spawn')
import hydra
from lightning.pytorch.loggers import WandbLogger
import lightning as L
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import Logger
from VCBench.modelcore.utils import multi_instantiate
from hydra.core.hydra_config import HydraConfig

log = logging.getLogger(__name__)


_PARSER_ARGS = None


def _str2bool(v: str | bool | None) -> bool | None:
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, str):
        if v.lower() in {"true", "1", "yes", "y"}:
            return True
        if v.lower() in {"false", "0", "no", "n"}:
            return False
    return None


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Parse command line flags (e.g. from wandb sweep) and later override Hydra config.
    We deliberately keep these as standard --flags so that wandb can inject them.
    """
    parser = argparse.ArgumentParser(add_help=False)

    # config groups (will be translated into Hydra overrides like data=..., model=...)
    parser.add_argument("--data", dest="data", type=str)
    parser.add_argument("--model", dest="model", type=str)
    parser.add_argument("--logger", dest="logger", type=str)

    # leaf parameters that we want to override directly
    parser.add_argument("--data.task", dest="data_task", type=str)
    parser.add_argument("--data.data_path", dest="data_data_path", type=str)

    parser.add_argument("--train", dest="train", type=str)
    parser.add_argument("--test", dest="test", type=str)
    parser.add_argument("--test_ckpt_type", dest="test_ckpt_type", type=str)

    parser.add_argument(
        "--trainer.log_every_n_steps",
        dest="trainer_log_every_n_steps",
        type=int,
    )
    parser.add_argument("--trainer.max_epochs", dest="trainer_max_epochs", type=int)
    parser.add_argument("--trainer.min_epochs", dest="trainer_min_epochs", type=int)
    # devices is usually a Hydra list, keep it as string so Hydra can parse it
    parser.add_argument("--trainer.devices", dest="trainer_devices", type=str)
    parser.add_argument("--trainer.strategy", dest="trainer_strategy", type=str)
    parser.add_argument("--trainer.accelerator", dest="trainer_accelerator", type=str)

    parser.add_argument("--data.val_batch_size", dest="data_val_batch_size", type=int)
    parser.add_argument(
        "--data.test_batch_size",
        dest="data_test_batch_size",
        type=int,
    )
    parser.add_argument("--data.train_num_workers", dest="data_train_num_workers", type=int)
    parser.add_argument("--data.val_num_workers", dest="data_val_num_workers", type=int)
    parser.add_argument("--data.test_num_workers", dest="data_test_num_workers", type=int)
    parser.add_argument("--data.embedding_key", dest="data_embedding_key", type=str)
    parser.add_argument("--data.cov_keys", dest="data_cov_keys", type=str)
    parser.add_argument("--data.result_avg_keys", dest="data_result_avg_keys", type=str)
    parser.add_argument("--data.sample_mode", dest="data_sample_mode", type=str)
    parser.add_argument("--data.cell_set_len", dest="data_cell_set_len", type=int)
    parser.add_argument(
        "--data.transform.use_covs",
        dest="data_transform_use_covs",
        type=str,
    )
    parser.add_argument(
        "--data.transform.drug_map_path",
        dest="data_transform_drug_map_path",
        type=str,
    )
    parser.add_argument(
        "--data.transform.gene_map_path",
        dest="data_transform_gene_map_path",
        type=str,
    )

    parser.add_argument("--model.use_mask", dest="model_use_mask", type=str)
    parser.add_argument("--model.use_cell_emb", dest="model_use_cell_emb", type=str)
    parser.add_argument("--model.diffusion_steps", dest="model_diffusion_steps", type=int)
    parser.add_argument("--model.n_selected_genes", dest="model_n_selected_genes", type=int)
    parser.add_argument("--data.train_batch_size", dest="data_train_batch_size", type=int)
    parser.add_argument("--model.dropout", dest="model_dropout", type=float)
    parser.add_argument("--model.lr", dest="model_lr", type=float)
    parser.add_argument(
        "--model.lr_scheduler_max_lr",
        dest="model_lr_scheduler_max_lr",
        type=float,
    )
    parser.add_argument(
        "--model.lr_scheduler_factor",
        dest="model_lr_scheduler_factor",
        type=float,
    )
    parser.add_argument(
        "--model.lr_scheduler_mode",
        dest="model_lr_scheduler_mode",
        type=str,
    )
    parser.add_argument(
        "--model.lr_scheduler_patience",
        dest="model_lr_scheduler_patience",
        type=int,
    )
    parser.add_argument("--model.wd", dest="model_wd", type=float)

    # early stopping parameters
    parser.add_argument(
        "--callbacks.early_stopping.monitor",
        dest="callbacks_early_stopping_monitor",
        type=str,
    )
    parser.add_argument(
        "--callbacks.early_stopping.patience",
        dest="callbacks_early_stopping_patience",
        type=int,
    )

    # logger / wandb related
    parser.add_argument(
        "--logger.wandb.project",
        dest="logger_wandb_project",
        type=str,
    )
    parser.add_argument(
        "--logger.wandb.name",
        dest="logger_wandb_name",
        type=str,
    )

    return parser


def _apply_cli_overrides(cfg: DictConfig, args: argparse.Namespace | None) -> DictConfig:
    """Override Hydra cfg using highest-priority CLI arguments."""
    if args is None:
        return cfg

    # Allow adding new keys via CLI (e.g., data.task, model.lr) even when struct is enabled.
    # We only relax struct during the override phase.
    from omegaconf import OmegaConf as _OC
    _OC.set_struct(cfg, False)

    mapping: dict[str, str] = {
        # config groups are handled via Hydra overrides in __main__, not here:
        # "data": "data",
        # "model": "model",
        # "logger": "logger",
        "data_task": "data.task",
        "data_data_path": "data.data_path",
        "train": "train",
        "test": "test",
        "test_ckpt_type": "test_ckpt_type",
        "trainer_log_every_n_steps": "trainer.log_every_n_steps",
        "trainer_max_epochs": "trainer.max_epochs",
        "trainer_min_epochs": "trainer.min_epochs",
        "trainer_strategy": "trainer.strategy",
        "trainer_accelerator": "trainer.accelerator",
        # "trainer_devices" is translated to a Hydra override in __main__
        "data_val_batch_size": "data.val_batch_size",
        "data_test_batch_size": "data.test_batch_size",
        "data_train_num_workers": "data.train_num_workers",
        "data_val_num_workers": "data.val_num_workers",
        "data_test_num_workers": "data.test_num_workers",
        "data_embedding_key": "data.embedding_key",
        "data_cov_keys": "data.cov_keys",
        "data_result_avg_keys": "data.result_avg_keys",
        "data_sample_mode": "data.sample_mode",
        "data_cell_set_len": "data.cell_set_len",
        "data_transform_use_covs": "data.transform.use_covs",
        "data_transform_drug_map_path": "data.transform.drug_map_path",
        "data_transform_gene_map_path": "data.transform.gene_map_path",
        "model_use_mask": "model.use_mask",
        "model_use_cell_emb": "model.use_cell_emb",
        "model_diffusion_steps": "model.diffusion_steps",
        "model_n_selected_genes": "model.n_selected_genes",
        "data_train_batch_size": "data.train_batch_size",
        "model_dropout": "model.dropout",
        "model_lr": "model.lr",
        "model_lr_scheduler_max_lr": "model.lr_scheduler_max_lr",
        "model_lr_scheduler_factor": "model.lr_scheduler_factor",
        "model_lr_scheduler_mode": "model.lr_scheduler_mode",
        "model_lr_scheduler_patience": "model.lr_scheduler_patience",
        "model_wd": "model.wd",
        "callbacks_early_stopping_monitor": "callbacks.early_stopping.monitor",
        "callbacks_early_stopping_patience": "callbacks.early_stopping.patience",
        "logger_wandb_project": "logger.wandb.project",
        "logger_wandb_name": "logger.wandb.name",
    }

    for attr, key in mapping.items():
        if not hasattr(args, attr):
            continue
        value = getattr(args, attr)
        if value is None:
            continue

        # Convert string booleans for known boolean fields
        if key in {
            "train",
            "test",
            "data.transform.use_covs",
            "model.use_mask",
            "model.use_cell_emb",
        }:
            value = _str2bool(value)
            if value is None:
                continue
        
        # Handle list-type parameters (e.g., [split_category])
        # If value is a string that looks like a list, parse it
        if key in {"data.cov_keys", "data.result_avg_keys"}:
            if isinstance(value, str):
                # Remove brackets and split by comma
                value = value.strip()
                if value.startswith('[') and value.endswith(']'):
                    value = value[1:-1].strip()
                # Split by comma and strip whitespace
                if value:
                    value = [item.strip().strip("'\"") for item in value.split(',') if item.strip()]
                else:
                    value = []
        
        OmegaConf.update(cfg, key, value, merge=False)

    # Allow dynamic batch_size adjustment for all models (including state_sm)
    # Removed forced batch_size setting - models should handle batch_size requirements internally if needed

    return cfg


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint file for automatic training resume.
    
    Search order:
    1. First look for last.ckpt (if it exists)
    2. Then look for the newest checkpoint under checkpoints/
    3. Return the path of the newest checkpoint
    
    Args:
        output_dir: Hydra output directory (contains checkpoints subdirectory)
        
    Returns:
        Path to the newest checkpoint, or None if not found
    """
    if not output_dir or not os.path.exists(output_dir):
        return None
    
    checkpoints = []
    
    # 1. Look for last.ckpt first (highest priority)
    last_ckpt = os.path.join(output_dir, "last.ckpt")
    if os.path.exists(last_ckpt):
        log.info(f"Found last.ckpt: {last_ckpt}")
        return last_ckpt
    
    # 2. Find all checkpoints under checkpoints/ (now a single unified directory)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    if os.path.exists(ckpt_dir):
        ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
        checkpoints.extend(ckpts)
    
    # 4. Look for other possible checkpoint locations
    for pattern in ["*.ckpt", "checkpoints/**/*.ckpt"]:
        found = glob.glob(os.path.join(output_dir, pattern), recursive=True)
        checkpoints.extend(found)
    
    if not checkpoints:
        return None
    
    # Sort by modification time and return the latest one
    latest_ckpt = max(checkpoints, key=os.path.getmtime)
    log.info(f"Found latest checkpoint: {latest_ckpt} (modified: {os.path.getmtime(latest_ckpt)})")
    return latest_ckpt


def get_auto_experiment_name(cfg: DictConfig, model_name: str) -> str:
    OmegaConf.set_struct(cfg, False)
    # Generate a run name containing model name, learning rate, weight decay, and batch size
    lr = cfg.get("model", {}).get("lr", "unknown")
    wd = cfg.get("model", {}).get("wd", "unknown")
    batch_size = cfg.get("data", {}).get("train_batch_size", "unknown")

    # Format parameters (avoid scientific notation)
    if isinstance(lr, float):
        lr_str = f"{lr:.0e}" if lr < 0.01 else f"{lr}"
    else:
        lr_str = str(lr)

    if isinstance(wd, float):
        wd_str = f"{wd:.0e}" if wd < 0.01 else f"{wd}"
    else:
        wd_str = str(wd)

    batch_size_str = str(batch_size)

    auto_name = f"{model_name}_lr{lr_str}_wd{wd_str}_bs{batch_size_str}"

    # Modify config (before logger creation)
    OmegaConf.update(cfg, "logger.wandb.name", auto_name, merge=False)
    log.info("Auto-generated wandb run name: %s (model: %s, lr: %s, wd: %s, bs: %s)",
             auto_name, model_name, lr_str, wd_str, batch_size_str)
    return auto_name

def train(runtime_context: dict):

    cfg = runtime_context["cfg"]
    # Set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # Set sample_mode and cell_set_len based on model type
    # Model decides data packaging mode: cell-wise [B,G] vs set-wise [B,S,G]
    model_target = cfg.model.get("_target_", "")
    is_state_transition = "StateTransitionPerturbationModel" in model_target or "state_transition" in model_target.lower()
    
    # Set sample_mode: "set" for state_transition models, "cell" for others
    # Only auto-set if not already specified via CLI/config
    if cfg.data.get("sample_mode") is None:
        sample_mode = "set" if is_state_transition else "cell"
        OmegaConf.update(cfg, "data.sample_mode", sample_mode, merge=False)
    else:
        sample_mode = cfg.data.get("sample_mode")
    
    # Set cell_set_len: 128 (or from model config) for state_transition, None for others
    # Only auto-set if not already specified via CLI/config
    if cfg.data.get("cell_set_len") is None:
        cell_set_len = cfg.model.get("cell_set_len", 128) if is_state_transition else None
        OmegaConf.update(cfg, "data.cell_set_len", cell_set_len, merge=False)
    else:
        cell_set_len = cfg.data.get("cell_set_len")
    
    log.info(f"Model {model_target}: sample_mode={sample_mode}, cell_set_len={cell_set_len}")

    log.info("Instantiating datamodule <%s>", cfg.data._target_)
    datamodule: L.LightningDataModule =hydra.utils.instantiate(
        cfg.data,
        seed=cfg.seed
    ) # Initialize the class specified by cfg.data['_target_'] and return an instance

    log.info("Instantiating model <%s>", cfg.model._target_)
    model = hydra.utils.instantiate(cfg.model, datamodule=datamodule) # Initialize the class specified by cfg.model['_target_'] and return an instance

    log.info("Instantiating callbacks...")
    callbacks: List[L.Callback] = multi_instantiate(cfg.get("callbacks"))

    # Before creating loggers, extract model name and update config
    # This allows WandbLogger to use the correct name on initialization
    
    # Infer model name from cfg (extracted from _target_)
    model_target = cfg.get("model", {}).get("_target_", "")
    model_name = model_target.split('.')[-1].lower()
    
    # If logger.wandb.name is default "gears" or unset, update it before creation
    auto_name = get_auto_experiment_name(cfg, model_name)
    log.info("Instantiating loggers...")
    # Try to instantiate loggers; if wandb initialization fails, continue with other loggers

    loggers: List[Logger] = multi_instantiate(cfg.get("logger"))
    
    # Double check: if logger is already created but name is still gears, set it again
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            logger.experiment.name = auto_name
            log.info("Updated wandb run name to: %s", auto_name)
    
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=loggers
    )

    if cfg.get("train"):
        # Automatically detect and resume checkpoint (if available and ckpt_path not manually specified)
        ckpt_path = cfg.get("ckpt_path")
        if ckpt_path is None:
            # Try to automatically find the latest checkpoint
            output_dir = cfg.get("paths", {}).get("output_dir", None)
            if output_dir:
                auto_ckpt = find_latest_checkpoint(output_dir)
                if auto_ckpt:
                    ckpt_path = auto_ckpt
                    log.info(f"Auto-resuming from checkpoint: {ckpt_path}")
                else:
                    log.info("No checkpoint found. Starting fresh training.")
            else:
                log.info("output_dir not found in config. Starting fresh training.")
        else:
            log.info(f"Using manually specified checkpoint: {ckpt_path}")
        
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    train_metrics = trainer.callback_metrics

    summary_metrics_dict = {}
    if cfg.get("test"):
        log.info("Starting testing!")
        # Track which checkpoint type is actually used for testing so that
        # prediction file names can include this as a suffix.
        selected_ckpt_type = "unknown"
        if cfg.get("train"):
            # if (
            #     trainer.checkpoint_callback is None
            #     or trainer.checkpoint_callback.best_model_path == ""
            # ):
            #     ckpt_path = None
            # else:
            #     ckpt_path = "best"
            # Support multiple checkpoint callbacks (e.g., loss and PCC)
            test_ckpt_type = cfg.get("test_ckpt_type", "loss")  # "loss" or "pcc"
            # Find the appropriate checkpoint callback
            ckpt_path = None
            if callbacks:
                for callback in callbacks:
                    if isinstance(callback, L.pytorch.callbacks.ModelCheckpoint):
                        # Check if this is the checkpoint we want
                        if test_ckpt_type == "loss" and "loss" in str(callback.monitor).lower():
                            if callback.best_model_path and callback.best_model_path != "":
                                ckpt_path = callback.best_model_path
                                selected_ckpt_type = "loss"
                                log.info(f"Using checkpoint from {test_ckpt_type} callback: {ckpt_path}")
                                break
                        elif test_ckpt_type == "pcc" and "pcc" in str(callback.monitor).lower():
                            if callback.best_model_path and callback.best_model_path != "":
                                ckpt_path = callback.best_model_path
                                selected_ckpt_type = "pcc"
                                log.info(f"Using checkpoint from {test_ckpt_type} callback: {ckpt_path}")
                                break
                
                # Fallback to first checkpoint callback if specific one not found
                if ckpt_path is None:
                    for callback in callbacks:
                        if isinstance(callback, L.pytorch.callbacks.ModelCheckpoint):
                            if callback.best_model_path and callback.best_model_path != "":
                                ckpt_path = callback.best_model_path
                                # Fallback checkpoint type when a specific monitor is not found
                                if test_ckpt_type in ["loss", "pcc"]:
                                    selected_ckpt_type = test_ckpt_type
                                else:
                                    selected_ckpt_type = "unknown"
                                log.info(f"Fallback: Using checkpoint: {ckpt_path}")
                                break
        else:
            ckpt_path = cfg.get("ckpt_path")

        # Pass the actually used checkpoint type to the model so that it can
        # append the suffix (e.g., predictions_loss.h5ad, predictions_pcc.h5ad).
        try:
            setattr(model, "current_test_ckpt_type", selected_ckpt_type)
        except Exception:
            # If anything goes wrong here, fall back to default behaviour
            # in which predictions are saved without a ckpt-type suffix.
            pass

        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        summary_metrics_dict = model.summary_metrics.to_dict()[
            model.summary_metrics.columns[0]
        ]

    test_metrics = trainer.callback_metrics
    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics, **summary_metrics_dict}

    return metric_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    global _PARSER_ARGS

    # CLI parser args have highest priority: use them to override Hydra cfg
    cfg = _apply_cli_overrides(cfg, _PARSER_ARGS)
    
    # Set TMPDIR: prefer existing env var, otherwise use a temp directory under the log directory
    # This ensures temp files are created on the same filesystem as checkpoints, avoiding cross-device move issues
    import os
    existing_tmpdir = os.environ.get("TMPDIR")
    if existing_tmpdir:
        # If TMPDIR is already set (e.g., from run_multi_gpu_agents.sh), keep the existing setting
        log.info(f"Using existing TMPDIR: {existing_tmpdir}")
    else:
        # Otherwise set it under the log directory to avoid cross-device checkpoint save issues
        log_dir = cfg.get("paths", {}).get("log_dir", None)
        if log_dir:
            # Ensure log_dir exists
            os.makedirs(log_dir, exist_ok=True)
            # Create a temporary directory under log_dir
            tmp_dir = os.path.join(log_dir, ".tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            # Set TMPDIR environment variable (effective for current process only)
            os.environ["TMPDIR"] = tmp_dir
            log.info(f"Set TMPDIR to {tmp_dir} to avoid cross-device checkpoint save errors")

    runtime_context = {"cfg": cfg, "trial_number": HydraConfig.get().job.get("num")}

    try:
    # Train the model
        global metric_dict
        metric_dict = train(runtime_context)

        # Combined metric
        metrics_use = cfg.get("metrics_to_optimize")
        if metrics_use:
            combined_metric = sum(
                [metric_dict.get(metric) * weight for metric, weight in metrics_use.items()]
            )
            return combined_metric
    except Exception as e:
        # Catch all exceptions, log the error, then re-raise
        # This lets wandb agent record the failure and continue to the next run
        log.error("Training failed with exception: %s", e, exc_info=True)
        # Re-raise the exception so wandb agent knows this run failed
        raise


if __name__ == "__main__":
    # 1) First parse standard CLI flags (e.g. from wandb sweep), keep unknowns
    parser = _build_arg_parser()
    _PARSER_ARGS, remaining = parser.parse_known_args()

    # 2) Translate certain argparse flags into Hydra-style overrides so that
    #    config groups (data=..., model=..., logger=...) and complex types
    #    (like trainer.devices=[0]) are handled by Hydra, while leaf parameters
    #    remain as --key flags for our own override logic.
    hydra_overrides: list[str] = []

    if getattr(_PARSER_ARGS, "data", None) is not None:
        hydra_overrides.append(f"data={_PARSER_ARGS.data}")
        _PARSER_ARGS.data = None  # avoid double-handling

    if getattr(_PARSER_ARGS, "model", None) is not None:
        hydra_overrides.append(f"model={_PARSER_ARGS.model}")
        _PARSER_ARGS.model = None

    if getattr(_PARSER_ARGS, "logger", None) is not None:
        hydra_overrides.append(f"logger={_PARSER_ARGS.logger}")
        _PARSER_ARGS.logger = None

    # Handle trainer.devices specially so that Hydra, not Lightning, parses
    # the list syntax (e.g., [0]) correctly.
    if getattr(_PARSER_ARGS, "trainer_devices", None) is not None:
        hydra_overrides.append(f"trainer.devices={_PARSER_ARGS.trainer_devices}")
        _PARSER_ARGS.trainer_devices = None

    # 3) Let Hydra see its overrides plus the remaining arguments
    sys.argv = [sys.argv[0]] + hydra_overrides + remaining

    # 4) Run Hydra entrypoint as usual; inside main we will override cfg with _PARSER_ARGS
    main()
