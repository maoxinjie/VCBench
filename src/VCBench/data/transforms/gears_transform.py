from .base  import TransformBase

class GearsTransform(TransformBase):
    def __init__(self,
                 obs_df,
                 pert_key):
        super().__init__(obs_df)
        self.pert_key = pert_key

    def __call__(self, example):

        out = {
            self.pert_key:example[self.pert_key],
            'control_cell_counts':example['control_cell_counts'],
            'pert_cell_counts':example['pert_cell_counts'],
        }

        # Pass through expression masks for masked loss calculation
        if 'pert_expression_mask' in example:
            out['pert_expression_mask'] = example['pert_expression_mask']
        if 'control_expression_mask' in example:
            out['control_expression_mask'] = example['control_expression_mask']

        return out
