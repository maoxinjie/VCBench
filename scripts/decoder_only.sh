PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PYTHONPATH:$PROJECT_ROOT/src"
export TMPDIR=/tmp  # Avoid AF_UNIX path too long
HYDRA_FULL_ERROR=1 train trainer.devices=[0] \
trainer.min_epochs=0 \
trainer.max_epochs=1 \
data=mix_pert \
data.embedding_key=null \
data.cov_keys=[split_category] \
data.result_avg_keys=[split_category] \
data.train_batch_size=300 \
data.sample_mode='cell' \
data.transform.gene_map_path='./ESM2_pert_features.pt' \
model=decoder_only \
logger=wandb \
data.data_path='./tasks/unseen_perts/norman2019_comb.h5ad' 