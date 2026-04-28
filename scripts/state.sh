PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_PATH="${VCBENCH_DATA_PATH:-$PROJECT_ROOT/tasks_data/unseen_perts/norman2019_comb_stack.h5ad}"
GENE_MAP_PATH="${VCBENCH_GENE_MAP_PATH:-$PROJECT_ROOT/tasks_data/model_related/ESM2_pert_features.pt}"
LOGGER="${VCBENCH_LOGGER:-csv}"

export PYTHONPATH="$PYTHONPATH:$PROJECT_ROOT/src"
export TMPDIR=/tmp  # Avoid AF_UNIX path too long
HYDRA_FULL_ERROR=1 python "$PROJECT_ROOT/src/VCBench/modelcore/train.py" trainer.devices=1 \
trainer.min_epochs=0 \
trainer.max_epochs=1 \
trainer.accelerator=auto \
data=mix_pert \
data.embedding_key=null \
data.cov_keys=[split_category] \
data.result_avg_keys=[split_category] \
data.train_batch_size=8 \
data.val_batch_size=8 \
data.test_batch_size=8 \
data.train_num_workers=0 \
data.val_num_workers=0 \
data.test_num_workers=0 \
data.sample_mode='set' \
data.cell_set_len=128 \
data.transform.gene_map_path="$GENE_MAP_PATH" \
model=state_sm \
model.use_cell_emb=true \
model.use_mask=false \
logger="$LOGGER" \
data.data_path="$DATA_PATH"
