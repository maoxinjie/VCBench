PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PYTHONPATH:$PROJECT_ROOT/src"
export TMPDIR=/tmp  # Avoid AF_UNIX path too long

HYDRA_FULL_ERROR=1 python "$PROJECT_ROOT/src/VCBench/modelcore/train.py" \
    data=gears \
    data.data_path=./tasks_data/unseen_perts/norman2019_comb.h5ad \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
    data.test_batch_size=32 \
    data.pert_key=perturbation \
    data.cov_keys=[split_category] \
    model=gears \
    model.lr=1e-4 \
    model.lr_scheduler_mode=onecycle \
    model.lr_scheduler_patience=10 \
    model.lr_scheduler_factor=0.5 \
    model.use_mask=false \
    model.use_cell_emb=false \
    train=true \
    test=true \
    trainer=gears \
    trainer.max_epochs=1 \
    trainer.min_epochs=0 \
    trainer.devices=[1] \
    logger=wandb \
    logger.wandb.project=VCBench
