import numpy as np
import torch
from .base import TransformBase
import os

class OneHotTransform(TransformBase):
    def __init__(self,obs_df,mode,
                 pert_key,
                 cov_keys,
                 pert_map_path,
                 cov_maps_path,
                 use_embedding_key,
                 pert_comb_delim='+'):

        super().__init__(obs_df,mode)
        self.cov_keys = cov_keys
        self.pert_key = pert_key
        self.pert_comb_delim = pert_comb_delim
        self.use_embedding_key = use_embedding_key

        if os.path.exists(cov_maps_path):
            self.cov_maps=torch.load(cov_maps_path,weights_only=False)
            n_total_covs=0
            for map in self.cov_maps.values():
                n_total_covs+=len(list(map.keys()))
            self.n_total_covs=n_total_covs
        else:
            cov_maps_dir=os.path.dirname(cov_maps_path)
            if not os.path.exists(cov_maps_dir):
                os.makedirs(cov_maps_dir)
            self._generate_cov_maps()
            torch.save(self.cov_maps,cov_maps_path)

        if os.path.exists(pert_map_path):
            self.pert_map=torch.load(pert_map_path,weights_only=False)
            self.n_perts=len(list(self.pert_map.values())[0])
        else:
            pert_map_dir=os.path.dirname(pert_map_path)
            if not os.path.exists(pert_map_dir):
                os.makedirs(pert_map_dir)
            self._generate_pert_map()
            torch.save(self.pert_map,pert_map_path)

        self._check_all_in_map()


    def _check_all_in_map(self):
        perts=[]
        for comb_pert in self.obs_df[self.pert_key].unique():
            perts.extend(comb_pert.split(self.pert_comb_delim))
        perts=set(perts)
        assert len(perts-set(self.pert_map.keys())) ==0
        for cov_key,cov_map in self.cov_maps.items():
            assert len(set(self.obs_df[cov_key].unique())-set(cov_map.keys()))==0

    def _generate_pert_map(self):
        perts=[]
        for pert_comb in self.obs_df[self.pert_key].unique():
            perts.extend(pert_comb.split(self.pert_comb_delim))
        perts=np.array(list(set(perts)))
        pert_map={}
        for pert in perts:
            pert_map[pert]=torch.tensor(perts==pert,dtype=torch.float32)
        self.pert_map=pert_map
        self.n_perts=len(perts)

    def _generate_cov_maps(self):
        cov_uniques = {cov_key: self.obs_df[cov_key].unique() for cov_key in self.cov_keys}
        cov_maps = {}
        n_total_covs=0
        for cov_key, unique_vals in cov_uniques.items():
            _map = {}
            for val in unique_vals:
                _map[val] = torch.tensor(unique_vals == val, dtype=torch.float32)
            cov_maps[cov_key] = _map
            n_total_covs+=len(unique_vals)
        self.n_total_covs=n_total_covs
        self.cov_maps = cov_maps


    def __call__(self,example):

        comb_pert=example[self.pert_key]
        pert_emb=0
        for pert in comb_pert.split(self.pert_comb_delim):
            pert_emb+=self.pert_map[pert]

        if self.use_embedding_key:
            controls=example['control_cell_emb']
        else:
            controls=example['control_cell_counts']

        pert_cell_counts=example['pert_cell_counts']

        cov_embs={}
        for cov_key in self.cov_keys:
            cov_embs[cov_key]=self.cov_maps[cov_key][example[cov_key]]

        out = {
            'controls':controls,
            'pert_cell_counts':pert_cell_counts,
            self.pert_key:pert_emb,
            **cov_embs
        }

        # Pass through expression masks for masked loss calculation
        if 'pert_expression_mask' in example:
            out['pert_expression_mask'] = example['pert_expression_mask']
        if 'control_expression_mask' in example:
            out['control_expression_mask'] = example['control_expression_mask']

        return out



