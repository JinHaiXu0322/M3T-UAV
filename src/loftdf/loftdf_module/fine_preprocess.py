import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import repeat


class FinePreprocess(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.cat_c_feat = config['fine_concat_coarse_feat']
        self.W = self.config['fine_window_size']
        self.debug_shapes = bool(config.get('debug_shapes', False))

        d_model_c = self.config['coarse']['d_model']
        d_model_f = self.config['fine']['d_model']
        self.d_model_f = d_model_f
        self.corr_fuse = nn.Linear(3 * d_model_f, d_model_f, bias=True)
        if self.cat_c_feat:
            self.down_proj = nn.Linear(d_model_c, d_model_f, bias=True)
            self.merge_feat = nn.Linear(2*d_model_f, d_model_f, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def _extract_patches(self, feat_map, b_ids, x_ids, y_ids):
        # feat_map: [B, C, H, W], output: [M, WW, C]
        bsz, channels, height, width = feat_map.shape
        _ = bsz
        W = self.W
        ww = W ** 2
        unfold = F.unfold(feat_map, kernel_size=(W, W), stride=1, padding=W // 2)  # [B, C*WW, H*W]
        unfold = unfold.transpose(1, 2).contiguous()  # [B, H*W, C*WW]
        linear_idx = (y_ids * width + x_ids).long()
        picked = unfold[b_ids.long(), linear_idx]  # [M, C*WW]
        return picked.view(-1, channels, ww).transpose(1, 2).contiguous()  # [M, WW, C]

    def forward(self, feat_f0, feat_f1, feat_c0, feat_c1, data):
        W = self.W
        ww = W ** 2

        data.update({'W': W})
        if data['b_ids'].shape[0] == 0:
            feat0 = torch.empty(0, self.W**2, self.d_model_f, device=feat_f0.device)
            feat1 = torch.empty(0, self.W**2, self.d_model_f, device=feat_f0.device)
            return feat0, feat1

        # Step 2: map coarse coordinates (H/8) -> H/4
        b_ids, i_ids, j_ids = data['b_ids'], data['i_ids'], data['j_ids']
        w0c, w1c = data['hw0_c'][1], data['hw1_c'][1]
        x0_c, y0_c = i_ids % w0c, i_ids // w0c
        x1_c, y1_c = j_ids % w1c, j_ids // w1c
        x0_1_4, y0_1_4 = x0_c * 2, y0_c * 2
        x1_1_4, y1_1_4 = x1_c * 2, y1_c * 2
        x0_1_4 = x0_1_4.clamp(0, feat_f0.shape[-1] - 1)
        y0_1_4 = y0_1_4.clamp(0, feat_f0.shape[-2] - 1)
        x1_1_4 = x1_1_4.clamp(0, feat_f1.shape[-1] - 1)
        y1_1_4 = y1_1_4.clamp(0, feat_f1.shape[-2] - 1)
        data.update({
            'coords_c0': torch.stack([x0_c, y0_c], dim=-1),
            'coords_c1': torch.stack([x1_c, y1_c], dim=-1),
            'coords_1_4_0': torch.stack([x0_1_4, y0_1_4], dim=-1),
            'coords_1_4_1': torch.stack([x1_1_4, y1_1_4], dim=-1),
        })

        # Crop patches on H/4 feature maps
        p_1_4_uav = self._extract_patches(feat_f0, b_ids, x0_1_4, y0_1_4)  # [M, WW, C]
        p_1_4_map = self._extract_patches(feat_f1, b_ids, x1_1_4, y1_1_4)  # [M, WW, C]

        # Step 3: patch + feature correlation fusion
        # C_patch = correlation(P_uav, P_map)
        c_patch = (p_1_4_uav * p_1_4_map) / (self.d_model_f ** 0.5)  # [M, WW, C]
        # C_uav = correlation(P_uav, feat_1_4_map context)
        ctx_map = feat_f1.mean(dim=(2, 3))[b_ids].unsqueeze(1).expand(-1, ww, -1)  # [M, WW, C]
        c_uav = (p_1_4_uav * ctx_map) / (self.d_model_f ** 0.5)  # [M, WW, C]
        # C_map = correlation(P_map, feat_1_4_uav context)
        ctx_uav = feat_f0.mean(dim=(2, 3))[b_ids].unsqueeze(1).expand(-1, ww, -1)  # [M, WW, C]
        c_map = (p_1_4_map * ctx_uav) / (self.d_model_f ** 0.5)  # [M, WW, C]

        feat_f0_unfold = self.corr_fuse(torch.cat([c_patch, c_uav, c_map], dim=-1))  # [M, WW, C]
        feat_f1_unfold = self.corr_fuse(torch.cat([c_patch, c_map, c_uav], dim=-1))  # [M, WW, C]
        data.update({'C_fusion': feat_f0_unfold})

        # option: use coarse-level loftr feature as context: concat and linear
        if self.cat_c_feat:
            feat_c_win = self.down_proj(torch.cat([feat_c0[data['b_ids'], data['i_ids']],
                                                   feat_c1[data['b_ids'], data['j_ids']]], 0))  # [2n, c]
            feat_cf_win = self.merge_feat(torch.cat([
                torch.cat([feat_f0_unfold, feat_f1_unfold], 0),  # [2n, ww, cf]
                repeat(feat_c_win, 'n c -> n ww c', ww=W**2),  # [2n, ww, cf]
            ], -1))
            feat_f0_unfold, feat_f1_unfold = torch.chunk(feat_cf_win, 2, dim=0)

        if self.debug_shapes:
            print("coords_c:", data['coords_c0'].shape)
            print("coords_1_4:", data['coords_1_4_0'].shape)
            print("patch shape:", p_1_4_uav.shape)
            print("fusion shape:", feat_f0_unfold.shape)

        return feat_f0_unfold, feat_f1_unfold
