# Code Running Tutorial

## 1. Data Preparation

We provide some of the data needed for reproducing the experiments on Google Drive:
https://drive.google.com/drive/folders/1GrPW9x5_npnT7ILwDVsFWvfDIcqaSjdk?usp=drive_link

We provide the data used in this paper in the Google Drive directory.

## 2. Experiment Tutorial

We will provide a tutorial on running the unseen perturbation experiment.

## 3. Download Data

Download the `unseen_perts` directory and `model_related` directory from
Google Drive and place both under `./tasks_data/` in the project root.

Expected layout:

```text
VCBench/
├── tasks_data/
│   ├── unseen_perts/
│   │   ├── norman2019_comb_stack.h5ad
│   │   ├── ReplogleWeissman2022_K562_stack_hvg_split.h5ad
│   │   └── SrivatsanTrapnell2020_sciplex3_stack_hvg_split.h5ad
│   └── model_related/
│       ├── ESM2_pert_features.pt
│       ├── SMILES_pert_features.pt
│       ├── gene2go.pkl
│       └── essential_all_data_pert_genes.pkl
```

If the data already exists elsewhere, create a symlink:

```bash
ln -s /path/to/tasks_data ./tasks_data
```

## 4. Configure Data Paths

### Norman Dataset

Set `parameters.data.data_path` in all `.yaml` files under `./sweep/norman` to:
```yaml
./tasks_data/unseen_perts/norman2019_comb_stack.h5ad
```

### Replogle Dataset

Set `parameters.data.data_path` in all `.yaml` files under `./sweep/replogle` to:
```yaml
./tasks_data/unseen_perts/ReplogleWeissman2022_K562_stack_hvg_split.h5ad
```

### Sciplex Dataset

Set `parameters.data.data_path` in all `.yaml` files under `./sweep/sciplex` to:
```yaml
./tasks_data/unseen_perts/SrivatsanTrapnell2020_sciplex3_stack_hvg_split.h5ad
```

## 5. Configure Feature Mapping Paths

### Norman and Replogle Datasets

Set `parameters.data.transform.gene_map_path` in all `.yaml` files under `./sweep/norman` and `./sweep/replogle` to:
```yaml
./tasks_data/model_related/ESM2_pert_features.pt
```

### Sciplex Dataset

Set `parameters.data.transform.drug_map_path` in all `.yaml` files under `./sweep/sciplex` to:
```yaml
./tasks_data/model_related/SMILES_pert_features.pt
```

## 6. GEARS Model Configuration

For GEARS, the following settings are required:

### Set Data Paths

- Set `parameters.model.data_path` in `./sweep/norman/no-stack/gears.yaml` to:
  ```yaml
  ./tasks_data/gears_norman
  ```

- Set `parameters.model.data_path` in `./sweep/replogle/no-stack/gears.yaml` to:
  ```yaml
  ./tasks_data/gears_replogle
  ```

These GEARS directories must contain the precomputed `go.csv` and
`co-express.csv` files.

### Set Gene Mapping Paths

Set the following parameters in `./src/VCBench/configs/model/gears.yaml`:

- Set `gene2go_path` to:
  ```yaml
  ${paths.data_dir}/model_related/gene2go.pkl
  ```

- Set `gene_set_path` to:
  ```yaml
  ${paths.data_dir}/model_related/essential_all_data_pert_genes.pkl
  ```

## 7. Environment Configuration

### Install and Activate Environment

First, switch to the project root directory in the terminal:

```bash
conda env create -f ./vcbench.yml
conda activate vcbench
pip install -e .
```

`vcbench.yml` is intended for Linux x86_64 GPU reproduction. For other
platforms, use it as a reference dependency list and adjust packages as needed.

### Configure WandB

```bash
wandb login
```

Enter your API KEY.

## 8. Run Experiments

You can run all `.yaml` files under `./sweep` as follows:

```bash
wandb sweep [yaml_path]
```

The terminal will return: Run Sweep Agent With: xxxxx

Copy and paste `xxxxx` into the terminal and press Enter to start the process.

## 9. Check Progress

Log in to WandB in your browser to check the progress.

For a local single-run smoke test without W&B sweeps, use:

```bash
python src/VCBench/modelcore/train.py \
  data=mix_pert \
  model=latent_additive \
  data.data_path=./tasks_data/unseen_perts/norman2019_comb_stack.h5ad \
  data.transform.gene_map_path=./tasks_data/model_related/ESM2_pert_features.pt \
  train=false \
  test=false
```
