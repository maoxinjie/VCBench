# VCBench

VCBench is a unified framework for single-cell perturbation effect prediction and fair benchmarking. It supports multi-dataset, multi-model training workflows with reproducible experiment management based on Hydra + Weights & Biases (W&B).

Paper demo site: <https://maoxinjie.github.io/VCBench-demo/>

## Acknowledgement

The overall implementation and project organization are built on top of [PerturBench](https://github.com/altoslabs/perturbench).

## Citation

If you use VCBench in your research, or build upon its framework design, please also cite PerturBench:

```bibtex
@article{wu2025perturbench,
  title={Perturbench: Benchmarking machine learning models for cellular perturbation analysis},
  author={Wu, Yan and Wershof, Esther and Schmon, Sebastian M and Nassar, Marcel and Osi{\'n}ski, B{\l}a{\.z}ej and Eksi, Ridvan and Yan, Zichao and Stark, Rory and Zhang, Kun and Graepel, Thore},
  journal={arXiv preprint arXiv:2408.10609},
  year={2024}
}
```

---

## 1. Environment Setup

Run the following commands in the project root:

```bash
conda env create -f ./vcbench.yml
conda activate vcbench
pip install -e .
```

Configure W&B:

```bash
wandb login
```

---

## 2. Core Project Structure

```text
VCBench/
├── src/VCBench/
│   ├── configs/                 # Hydra configs
│   │   └── model/               # Per-model config files
│   ├── modelcore/
│   │   ├── models/              # Model implementations and registration
│   │   └── train.py             # Training entrypoint (Hydra + CLI overrides)
│   └── data/                    # Data and dataset construction logic
├── sweep/                       # W&B sweep configs
├── NEW_MODEL_INTEGRATION.md     # Detailed guide for integrating new models
└── RUN_TUTORIAL.md              # Reproducible experiment running tutorial
```

---

## 3. Data Preparation and Path Configuration

### 3.1 Download Data

Download the following folders from Google Drive and place them in the project root:

- `unseen_perts`
- `model_related`

Download link: <https://drive.google.com/drive/folders/1GrPW9x5_npnT7ILwDVsFWvfDIcqaSjdk?usp=sharing>

### 3.2 Set Data File Paths (Sweep Configs)

- In `./sweep/norman/*.yaml`, set `parameters.data.data_path` to:

```yaml
./unseen_perts/norman2019_comb_stack.h5ad
```

- In `./sweep/replogle/*.yaml`, set `parameters.data.data_path` to:

```yaml
./unseen_perts/ReplogleWeissman2022_K562_stack_hvg_split.h5ad
```

- In `./sweep/sciplex/*.yaml`, set `parameters.data.data_path` to:

```yaml
./unseen_perts/SrivatsanTrapnell2020_sciplex3_stack_hvg_split.h5ad
```

### 3.3 Set Feature Mapping Paths

- Norman / Replogle: `parameters.data.transform.gene_map_path`

```yaml
./model_related/ESM2_pert_features.pt
```

- Sciplex: `parameters.data.transform.drug_map_path`

```yaml
./model_related/SMILES_pert_features.pt
```

### 3.4 Additional GEARS Configuration

- In `./sweep/norman/no-stack/gears.yaml`, set `parameters.model.data_path`:

```yaml
./gears_norman
```

- In `./sweep/replogle/no-stack/gears.yaml`, set `parameters.model.data_path`:

```yaml
./gears_replogle
```

- In `./src/VCBench/configs/model/gears.yaml`, set:

```yaml
gene2go_path: ./model_related/gene2go.pkl
gene_set_path: ./model_related/essential_all_data_pert_genes.pkl
```

---

## 4. Run Experiments

### 4.1 Use `train.py` (Hydra)

```bash
python src/VCBench/modelcore/train.py model=latent_additive train=true test=true
```

Quick config check (no training):

```bash
python src/VCBench/modelcore/train.py model=latent_additive train=false test=false
```

### 4.2 Use W&B Sweep

```bash
wandb sweep [yaml_path]
wandb agent [entity/project/sweep_id]
```

You can start with a single trial run:

```bash
wandb agent --count 1 [entity/project/sweep_id]
```

---

## 5. Integrating a New Model (Minimal Viable Flow)

Goal: make your model callable by both `train.py` and `wandb sweep`, with end-to-end train/validation/test support.

### Step 1: Add Model Code

Create `src/VCBench/modelcore/models/<your_model>.py`. It is recommended to inherit from `PerturbationModel` and implement:

- `forward(...)`
- `training_step(...)`
- `validation_step(...)`
- `predict(...)`

Recommended base-class utilities to reuse:

- `auto_mse(...)` (loss)
- `_get_mask(batch)` (mask logic)

Reference implementations:

- `src/VCBench/modelcore/models/linear_additive.py`
- `src/VCBench/modelcore/models/latent_additive.py`

### Step 2: Add Model Config

Create `src/VCBench/configs/model/<your_model>.yaml`, for example:

```yaml
_target_: VCBench.modelcore.models.YourModel

use_cell_emb: false
use_mask: true
lr: 1e-4
wd: 1e-6
lr_scheduler_mode: onecycle
```

Notes:

- `_target_` must point to the correct Python class.
- YAML argument names should match your `__init__` signature.

### Step 3: Register the Model

Add the import in `src/VCBench/modelcore/models/__init__.py`:

```python
from .your_model import YourModel
```

### Step 4: Update `train.py` Mapping for New Sweep Arguments

`train.py` currently uses a hybrid mechanism (`argparse + Hydra overrides`). If you add a new argument like `model.foo`:

1. Add it in `_build_arg_parser()`:

```text
--model.foo
```

2. Add mapping in `_apply_cli_overrides()`:

```text
"model_foo": "model.foo"
```

Otherwise, sweep arguments may not be applied correctly.

### Step 5: Minimal Validation Sequence

```bash
# 1) Config check only
python src/VCBench/modelcore/train.py model=<your_model> train=false test=false

# 2) One epoch
python src/VCBench/modelcore/train.py model=<your_model> trainer.max_epochs=1 train=true test=false

# 3) Train + test
python src/VCBench/modelcore/train.py model=<your_model> train=true test=true
```

---

## 6. Common Troubleshooting

- `_target_` class not found: check YAML path, class name, and registration in `models/__init__.py`.
- Missing batch fields during training: align with the existing batch protocol (for example, `pert_cell_counts` and `control_cell_counts`).
- Sweep arguments not taking effect: verify parser and mapping in `train.py` include your new arguments.
- Covariate dimension mismatch: check consistency between `use_covs` and transform settings.

---

## 7. Recommended Practices

- Start with a minimally runnable model, then add complex modules incrementally.
- Reuse base-class mask, scheduler, and logging utilities whenever possible to reduce duplicated code.
- Each time you add a sweep argument, update the `train.py` mapping to avoid no-op runs.

---

## 8. Related Documents

- `NEW_MODEL_INTEGRATION.md`: full guide for integrating new models.
- `RUN_TUTORIAL.md`: data preparation and experiment running guide.