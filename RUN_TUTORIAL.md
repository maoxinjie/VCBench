# Code Running Tutorial

## 1. Data Preparation

We provide some of the data needed for reproducing the experiments on Google Drive:
https://drive.google.com/drive/folders/1GrPW9x5_npnT7ILwDVsFWvfDIcqaSjdk?usp=sharing

We provide the data used in this paper in the Google Drive directory.

## 2. Experiment Tutorial

We will provide a tutorial on running the unseen perturbation experiment.

## 3. Download Data

Download the `unseen_perts` directory and `model_related` directory from Google Drive to the root directory of the project respectively.

## 4. Configure Data Paths

### Norman Dataset

Set `parameters.data.data_path` in all `.yaml` files under `./sweep/norman` to:
```yaml
./unseen_perts/norman2019_comb_stack.h5ad
```

### Replogle Dataset

Set `parameters.data.data_path` in all `.yaml` files under `./sweep/replogle` to:
```yaml
./unseen_perts/ReplogleWeissman2022_K562_stack_hvg_split.h5ad
```

### Sciplex Dataset

Set `parameters.data.data_path` in all `.yaml` files under `./sweep/sciplex` to:
```yaml
./unseen_perts/SrivatsanTrapnell2020_sciplex3_stack_hvg_split.h5ad
```

## 5. Configure Feature Mapping Paths

### Norman and Replogle Datasets

Set `parameters.data.transform.gene_map_path` in all `.yaml` files under `./sweep/norman` and `./sweep/replogle` to:
```yaml
./model_related/ESM2_pert_features.pt
```

### Sciplex Dataset

Set `parameters.data.transform.drug_map_path` in all `.yaml` files under `./sweep/sciplex` to:
```yaml
./model_related/SMILES_pert_features.pt
```

## 6. GEARS Model Configuration

For GEARS, the following settings are required:

### Set Data Paths

- Set `parameters.model.data_path` in `./sweep/norman/no-stack/gears.yaml` to:
  ```yaml
  ./gears_norman
  ```

- Set `parameters.model.data_path` in `./sweep/replogle/no-stack/gears.yaml` to:
  ```yaml
  ./gears_replogle
  ```

### Set Gene Mapping Paths

Set the following parameters in `./src/VCBench/configs/model/gears.yaml`:

- Set `gene2go_path` to:
  ```yaml
  ./model_related/gene2go.pkl
  ```

- Set `gene_set_path` to:
  ```yaml
  ./model_related/essential_all_data_pert_genes.pkl
  ```

## 7. Environment Configuration

### Install and Activate Environment

First, switch to the project root directory in the terminal:

```bash
conda env create -f ./vcbench.yml
conda activate vcbench
```

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