# New Model Integration Tutorial (VCBench)

This document provides a directly implementable new model integration process for the current repository. You can treat it as a checklist: follow the steps in order, and your model should be callable by `train.py` and `wandb sweep` normally.

---

## 1. Integration Goals

After completion, it should satisfy:

- Ability to select model and start training through Hydra:
  - `python src/VCBench/modelcore/train.py model=<your_model> ...`
- Ability to inject hyperparameters in sweep and execute:
  - `wandb sweep <yaml>`
  - `wandb agent <entity/project/sweep_id>`
- No interface errors in training/validation/test evaluation pipeline.

---

## 2. Which Files Need to Be Modified

At least 3 types of files are involved:

1. **Model code file**  
   `src/VCBench/modelcore/models/<your_model>.py`

2. **Model configuration file**  
   `src/VCBench/configs/model/<your_model>.yaml`

3. **Model registration entry**  
   `src/VCBench/modelcore/models/__init__.py`

> If you want to run `wandb sweep`, you also need to check the CLI parameter parsing whitelist in `src/VCBench/modelcore/train.py` (see Section 6).

---

## 3. Model Code Specifications (Recommended Template)

Most models in the repository inherit from `PerturbationModel` and follow a unified training interface. Recommended structure:

1. Inheritance:
   - `class YourModel(PerturbationModel):`
2. `__init__`:
   - Accepts `datamodule`, `lr`, `wd`, `use_mask`, scheduler parameters, etc.
   - Calls `super(..., datamodule=datamodule, ...)`
3. Implement core methods:
   - `forward(...)`
   - `training_step(...)`
   - `validation_step(...)`
   - `predict(...)`
4. Loss:
   - Prefer to reuse the base class's `auto_mse(...)`
   - Prefer to reuse `_get_mask(batch)` for mask logic

Recommended references:

- `src/VCBench/modelcore/models/linear_additive.py`
- `src/VCBench/modelcore/models/latent_additive.py`

---

## 4. Configuration File Writing (Hydra)

Create a new `<your_model>.yaml` under `src/VCBench/configs/model/`, with the core being `_target_`:

```yaml
_target_: VCBench.modelcore.models.YourModel

use_cell_emb: false
use_mask: true

lr: 1e-4
wd: 1e-6

lr_scheduler_mode: onecycle
```

Key points:

- `_target_` must be able to locate your Python class.
- Parameter names must align with the model's `__init__`.
- Parameters linked to the data side (such as `use_covs`) should maintain consistent semantics with existing models.

---

## 5. Model Registration

Add the import in `src/VCBench/modelcore/models/__init__.py`:

```python
from .your_model import YourModel
```

Not registering usually leads to poor readability of `_target_` and a less unified entry point, and it's easy to miss during subsequent maintenance.

---

## 6. Sweep Parameter Injection Notes (Important)

The `train.py` in the current repository uses a mixed mode of "argparse + Hydra override".

This means:

- If the parameters passed by sweep are not in `train.py`'s parser or mapping, they may not be applied to the final configuration as expected.

If you add new sweep hyperparameters to the model (e.g., `model.foo`), it is recommended to:

- Add `parser.add_argument("--model.foo", ...)` in `_build_arg_parser()`
- Add to `mapping` in `_apply_cli_overrides()`:
  - `"model_foo": "model.foo"`

This is the most stable approach, especially when your current `wandb` injects parameters through `${args}`.

---

## 7. Minimum Validation Flow

Check in the following order to locate problems most quickly:

1. **Configuration only (no training)**
   - `python src/VCBench/modelcore/train.py model=<your_model> train=false test=false`
2. **1 epoch training**
   - `python src/VCBench/modelcore/train.py model=<your_model> trainer.max_epochs=1 train=true test=false`
3. **Training + testing**
   - `python src/VCBench/modelcore/train.py model=<your_model> train=true test=true`
4. **Single sweep trial**
   - `wandb sweep <your_sweep.yaml>`
   - `wandb agent --count 1 <entity/project/sweep_id>`

---

## 8. Example: Integrating `latent_additive`

Here we use the existing `latent_additive` in the repository to demonstrate "how a model is integrated into the complete pipeline".

### 8.1 Code Entry Points

- Model implementation: `src/VCBench/modelcore/models/latent_additive.py`
- Model configuration: `src/VCBench/configs/model/latent_additive.yaml`
- Registration entry: `src/VCBench/modelcore/models/__init__.py`

### 8.2 How Configuration Points to Code

`latent_additive.yaml` uses:

```yaml
_target_: VCBench.modelcore.models.LatentAdditive
```

to bind the Hydra selector `model=latent_additive` to the Python class `LatentAdditive`.

Parameters in the same file (such as `n_layers`, `encoder_width`, `latent_dim`, `dropout`, `softplus_output`) are directly passed to the class constructor.

### 8.3 What the Model Class Does

Typical path for `LatentAdditive`:

1. Call `PerturbationModel` parent class initialization to access unified optimizer/scheduler/mask logic;
2. Read dimension information from `datamodule` (gene dimension, perturbation encoding dimension, covariate dimension);
3. Use `MixedPerturbationEncoder` to encode perturbation;
4. Encode control input into latent space, add it to perturbation latent;
5. Decode to gene expression space and output prediction;
6. Use `auto_mse` to calculate loss and log during training/validation.

### 8.4 Linkage with Data Configuration

The model supports automatic linkage with `use_covs`:

- If `datamodule.train_dataset.transform.use_covs=True`, the model side automatically enables covariate concatenation (even if not explicitly enabled in the configuration).

This is why `data.transform.use_covs` is directly related to model performance in sweeps.

### 8.5 How It's Called in train.py

The training entry `src/VCBench/modelcore/train.py` will:

1. Parse CLI / sweep injected parameters;
2. Instantiate `cfg.data` and `cfg.model` through Hydra;
3. Execute `trainer.fit(...)` and optional `trainer.test(...)`.

For `latent_additive`, you just need to ensure:

- `model=latent_additive`
- Relevant hyperparameters align with `__init__`

Then you can directly enter the training process.

### 8.6 Copy This Pattern to Integrate New Models

If you want to add a new `my_model`, the most convenient way is:

1. Copy `latent_additive.py` as a template and modify the network structure;
2. Create a new `my_model.yaml` and replace `_target_`;
3. Register in `models/__init__.py`;
4. If new parameters are passed in sweep, add them to `train.py` parameter whitelist;
5. Perform minimum validation as per Section 7.

---

## 9. Common Issues Quick Check

- **`_target_` cannot find the class**
  - Check if the yaml path and class name are consistent
  - Check if `models/__init__.py` is registered

- **Batch field does not exist during training**
  - Align with existing batch protocol (`pert_cell_counts`, `control_cell_counts`, covariate keys)
  - Prefer to reference field access methods in `linear_additive.py`

- **Sweep parameters not taking effect**
  - Check if `train.py`'s `argparse` + `mapping` includes the new parameters

- **Covariates dimension mismatch**
  - Check if `use_covs` is consistent with transform
  - Print tensor shapes before and after concatenation

---

## 10. Recommended Practices

- Start with a "minimally runnable version" for new models, and add complex mechanisms after ensuring interface stability.
- Reuse base class capabilities as much as possible (mask, scheduler, logging) to reduce duplicate code.
- When adding a sweep parameter, synchronously update the `train.py` mapping to avoid empty runs in online trials.