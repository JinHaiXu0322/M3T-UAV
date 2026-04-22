import torch
import torch.nn as nn
from einops.einops import rearrange

from .backbone import build_backbone
from .utils.position_encoding import PositionEncodingSine
from .loftdf_module import LocalFeatureTransformer, FinePreprocess
from .utils.coarse_matching import CoarseMatching
from .utils.fine_matching import FineMatching


class LoFTDF(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config
        self.debug_shapes = bool(config.get('debug_shapes', False))
        self._shape_logged = False

        # Modules
        self.backbone = build_backbone(config)
        feat_align_cfg = config.get('feat_align', {})
        coarse_in_dim = int(feat_align_cfg.get('coarse_in_dim', config['coarse']['d_model']))
        fine_in_dim = int(feat_align_cfg.get('fine_in_dim', config['fine']['d_model']))
        coarse_out_dim = int(config['coarse']['d_model'])
        fine_out_dim = int(config['fine']['d_model'])
        self.coarse_align = nn.Identity() if coarse_in_dim == coarse_out_dim else nn.Conv2d(coarse_in_dim, coarse_out_dim, kernel_size=1)
        self.fine_align = nn.Identity() if fine_in_dim == fine_out_dim else nn.Conv2d(fine_in_dim, fine_out_dim, kernel_size=1)

        self.pos_encoding = PositionEncodingSine(
            config['coarse']['d_model'],
            temp_bug_fix=config['coarse']['temp_bug_fix'])
        self.loftdf_coarse = LocalFeatureTransformer(config['coarse'])
        match_coarse_cfg = {**config['match_coarse']}
        match_coarse_cfg['d_model'] = match_coarse_cfg.get('matchability_dim', config['coarse']['d_model'])
        self.coarse_matching = CoarseMatching(match_coarse_cfg)
        self.fine_preprocess = FinePreprocess(config)
        self.loftdf_fine = LocalFeatureTransformer(config["fine"])
        self.fine_matching = FineMatching(config["fine"])

    def forward(self, data):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 1. Local Feature CNN
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
            if isinstance(feats, dict):
                feat_c = feats['feat_c']
                feat_1_4 = feats['feat_1_4']
                feat_c0, feat_c1 = feat_c.split(data['bs'])
                feat_f0, feat_f1 = feat_1_4.split(data['bs'])
            else:
                feats_c, feats_f = feats
                (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(data['bs']), feats_f.split(data['bs'])
        else:  # handle different input shapes
            ret0, ret1 = self.backbone(data['image0']), self.backbone(data['image1'])
            if isinstance(ret0, dict):
                feat_c0, feat_f0 = ret0['feat_c'], ret0['feat_1_4']
                feat_c1, feat_f1 = ret1['feat_c'], ret1['feat_1_4']
            else:
                (feat_c0, feat_f0), (feat_c1, feat_f1) = ret0, ret1

        # feature alignment for channel consistency across backbone -> coarse/fine modules
        feat_c0, feat_c1 = self.coarse_align(feat_c0), self.coarse_align(feat_c1)
        feat_f0, feat_f1 = self.fine_align(feat_f0), self.fine_align(feat_f1)

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })

        if self.debug_shapes and (not self._shape_logged):
            print("feat_c:", feat_c0.shape)
            print("feat_1_4:", feat_f0.shape)

        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
        feat_c0, feat_c1 = self.loftdf_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 4. fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, data)
        if self.debug_shapes and (not self._shape_logged):
            print("[DEBUG] feat_c0(seq):", tuple(feat_c0.shape), "feat_c1(seq):", tuple(feat_c1.shape))
            print("[DEBUG] fine patches feat_f0_unfold:", tuple(feat_f0_unfold.shape), "feat_f1_unfold:", tuple(feat_f1_unfold.shape))
        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.loftdf_fine(feat_f0_unfold, feat_f1_unfold)

        # 5. match fine-level
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)
        if self.debug_shapes and (not self._shape_logged):
            print("[DEBUG] mkpts0_f:", tuple(data["mkpts0_f"].shape), "mkpts1_f:", tuple(data["mkpts1_f"].shape))
            self._shape_logged = True

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)

