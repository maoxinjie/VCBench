import collections
import pandas as pd
import torch
import numpy as np


class noop_collate:
    """No operation collate function. Returns the batch as is."""

    def __call__(self, batch: list):
        if len(batch) == 1:
            return batch[0]
        else:
            return batch

class inference_collate:
    """Collate function for inference"""

    def __call__(self, data_list: list):
        #[(example_dict,obs),(example_dict,obs),.............]
        keys = data_list[0][0].keys()
        batch_dict = collections.defaultdict(list)
        batch_obs = []

        for example_dict, obs in data_list:
            for key in keys:
                val = example_dict[key]
                if isinstance(val, np.ndarray):
                    batch_dict[key].append(val)
                elif isinstance(val, torch.Tensor):
                    batch_dict[key].append(val)
                elif isinstance(val, int):
                    batch_dict[key].append(val)  # Keep as int, convert uniformly later
                elif isinstance(val, float):
                    batch_dict[key].append(val)  # Keep as float, convert uniformly later
                elif isinstance(val, str):
                    batch_dict[key].append(np.array(val))
                else:
                    batch_dict[key].append(val)
            batch_obs.append(obs)

        batch_obs = pd.concat(batch_obs)
        for key in keys:
            vals = batch_dict[key]
            first_val = vals[0]
            
            if isinstance(first_val, torch.Tensor):
                batch_dict[key] = torch.stack(vals)
            elif isinstance(first_val, np.ndarray):
                batch_dict[key] = np.stack(vals)
            elif isinstance(first_val, int):
                batch_dict[key] = torch.tensor(vals, dtype=torch.long)
            elif isinstance(first_val, float):
                batch_dict[key] = torch.tensor(vals, dtype=torch.float32)

        return batch_dict, batch_obs


class train_collate:
    """Optimized collate: supports padded tensor format"""
    
    def __call__(self, examples: list):
        #[example_dict1,example_dict2,example_dict3.........]
        keys = examples[0].keys()
        batch_dict = collections.defaultdict(list)

        for example_dict in examples:
            for key in keys:
                val = example_dict[key]
                if isinstance(val, np.ndarray):
                    batch_dict[key].append(val)
                elif isinstance(val, torch.Tensor):
                    batch_dict[key].append(val)
                elif isinstance(val, int):
                    batch_dict[key].append(val)  # Keep as int, convert uniformly later
                elif isinstance(val, float):
                    batch_dict[key].append(val)  # Keep as float, convert uniformly later
                elif isinstance(val, str):
                    batch_dict[key].append(np.array(val))
                else:
                    batch_dict[key].append(val)

        for key in keys:
            vals = batch_dict[key]
            first_val = vals[0]
            
            if isinstance(first_val, torch.Tensor):
                batch_dict[key] = torch.stack(vals)
            elif isinstance(first_val, np.ndarray):
                batch_dict[key] = np.stack(vals)
            elif isinstance(first_val, int):
                # Convert int types (like gene_pert_len) to LongTensor
                batch_dict[key] = torch.tensor(vals, dtype=torch.long)
            elif isinstance(first_val, float):
                batch_dict[key] = torch.tensor(vals, dtype=torch.float32)
            # Keep list types as-is (compatible with old code)

        return batch_dict