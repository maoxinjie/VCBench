PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_PATH="${VCBENCH_DATA_PATH:-$PROJECT_ROOT/tasks_data/unseen_perts/norman2019_comb_stack.h5ad}"
MODEL_RELATED_DIR="${VCBENCH_MODEL_RELATED_DIR:-$PROJECT_ROOT/tasks_data/model_related}"
LOGGER="${VCBENCH_LOGGER:-csv}"

export PYTHONPATH="$PYTHONPATH:$PROJECT_ROOT/src"
export TMPDIR=/tmp  # Avoid AF_UNIX path too long

HYDRA_FULL_ERROR=1 python "$PROJECT_ROOT/src/VCBench/modelcore/train.py" \
    data=gears \
    data.data_path="$DATA_PATH" \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
    data.test_batch_size=32 \
    data.train_num_workers=0 \
    data.val_num_workers=0 \
    data.test_num_workers=0 \
    data.pert_key=perturbation \
    data.cov_keys=[split_category] \
    model=gears \
    model.gene2go_path="$MODEL_RELATED_DIR/gene2go.pkl" \
    model.gene_set_path="$MODEL_RELATED_DIR/essential_all_data_pert_genes.pkl" \
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
    trainer.accelerator=auto \
    trainer.devices=1 \
    logger="$LOGGER"
