import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..loftdf_module.transformer import LoFTDFEncoderLayer


class DDFMatcher(nn.Module):
    """Dual-Deformation Field matcher for fine-level subpixel refinement."""

    def __init__(self, config):
        super().__init__()
        ddf_cfg = config.get("ddf", {})
        self.d_model = config["d_model"]
        self.nhead = config["nhead"]
        self.attention_type = config["attention"]
        self.num_layers = int(ddf_cfg.get("num_layers", 3))

        # ATTENTION: L=3, each layer contains self-attention + cross-attention
        self.self_layers = nn.ModuleList(
            [LoFTDFEncoderLayer(self.d_model, self.nhead, self.attention_type) for _ in range(self.num_layers)]
        )
        self.cross_layers = nn.ModuleList(
            [LoFTDFEncoderLayer(self.d_model, self.nhead, self.attention_type) for _ in range(self.num_layers)]
        )

        # Predict residual deformation field Δphi^l from correlation map
        hidden = max(16, self.d_model // 4)
        self.deform_head = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, kernel_size=3, padding=1, bias=True),
        )

    @staticmethod
    def _seq_to_map(feat_seq, window_size):
        # feat_seq: [B, WW, C] -> feat_map: [B, C, H, W]
        return feat_seq.transpose(1, 2).reshape(feat_seq.shape[0], feat_seq.shape[2], window_size, window_size)

    @staticmethod
    def _build_base_grid(batch, height, width, device, dtype):
        ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1)  # [H, W, 2]
        return grid.unsqueeze(0).repeat(batch, 1, 1, 1)  # [B, H, W, 2]

    def _warp(self, feat_map, phi):
        # WARP
        # feat_map: [B, C, H, W]
        # phi: [B, H, W, 2], pixel displacement (dx, dy)
        bsz, _, h, w = feat_map.shape
        base_grid = self._build_base_grid(bsz, h, w, feat_map.device, feat_map.dtype)

        norm_phi = torch.zeros_like(phi)
        norm_phi[..., 0] = phi[..., 0] / max((w - 1) * 0.5, 1e-6)
        norm_phi[..., 1] = phi[..., 1] / max((h - 1) * 0.5, 1e-6)
        sample_grid = base_grid + norm_phi

        return F.grid_sample(feat_map, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True)

    def _correlation(self, feat_uav_reg, feat_map):
        # CORRELATION
        # feat_uav_reg: [B, C, H, W], feat_map: [B, C, H, W]
        corr = (feat_uav_reg * feat_map).sum(dim=1, keepdim=True) / math.sqrt(feat_uav_reg.shape[1])
        return corr  # [B, 1, H, W]

    def forward(self, feat_f0, feat_f1, data):
        """
        Args:
            feat_f0 (torch.Tensor): UAV patch features [B, WW, C]
            feat_f1 (torch.Tensor): Map patch features [B, WW, C]
            data (dict)
        Update:
            data['expec_f']: [B, 3]  -> [dx_norm, dy_norm, std]
            data['mkpts0_f']: [B, 2]
            data['mkpts1_f']: [B, 2]
        """
        bsz, ww, channels = feat_f0.shape
        window_size = int(math.sqrt(ww))
        assert window_size * window_size == ww, "Fine window must be square."
        scale = data["hw0_i"][0] / data["hw0_f"][0]
        self.W = window_size
        self.scale = scale

        if bsz == 0:
            assert self.training is False, "M is always >0 when training (from coarse matching sampling)."
            data.update(
                {
                    "expec_f": torch.empty(0, 3, device=feat_f0.device),
                    "mkpts0_f": data["mkpts0_c"],
                    "mkpts1_f": data["mkpts1_c"],
                }
            )
            return

        # ATTENTION (forward l=1->3): collect multi-level features I^1, I^2, I^3
        feat_uav = feat_f0
        feat_map = feat_f1
        feat_levels = []
        for layer_idx in range(self.num_layers):
            feat_uav = self.self_layers[layer_idx](feat_uav, feat_uav)
            feat_map = self.self_layers[layer_idx](feat_map, feat_map)
            feat_uav_new = self.cross_layers[layer_idx](feat_uav, feat_map)
            feat_map_new = self.cross_layers[layer_idx](feat_map, feat_uav)
            feat_uav, feat_map = feat_uav_new, feat_map_new
            feat_levels.append((feat_uav, feat_map))

        # Backward deformation aggregation: phi^(4)=0, then l=3->2->1
        phi_next = torch.zeros(bsz, window_size, window_size, 2, device=feat_f0.device, dtype=feat_f0.dtype)
        for level_idx in reversed(range(self.num_layers)):
            feat_uav_l, feat_map_l = feat_levels[level_idx]
            feat_uav_l_map = self._seq_to_map(feat_uav_l, window_size)  # [B, C, H, W]
            feat_map_l_map = self._seq_to_map(feat_map_l, window_size)  # [B, C, H, W]

            # WARP: I_uav_reg = R(I_uav, phi^(l+1))
            feat_uav_reg = self._warp(feat_uav_l_map, phi_next)

            # CORRELATION: corr = correlation(I_uav_reg, I_map^l)
            corr = self._correlation(feat_uav_reg, feat_map_l_map)

            # Residual deformation: Δphi^l = Conv(corr), shape [B, H, W, 2]
            delta_phi = self.deform_head(corr).permute(0, 2, 3, 1).contiguous()

            # DEFORMATION UPDATE: phi^l = Δphi^l + phi^(l+1)
            phi_cur = delta_phi + phi_next
            phi_next = phi_cur

        # Final deformation field: phi = phi^1
        phi = phi_next  # [B, H, W, 2]
        data.update({"phi_f": phi})

        # Center displacement by averaging all pixel displacements
        # Δx = mean(Δx_i), Δy = mean(Δy_i)
        mean_disp = phi.mean(dim=(1, 2))  # [B, 2], in patch-pixel units
        radius = max(window_size // 2, 1)
        coords_normed = mean_disp / float(radius)  # [B, 2], normalized offsets

        disp_mag = torch.linalg.norm(phi, dim=-1)  # [B, H, W]
        std = disp_mag.std(dim=(1, 2))
        data.update({"expec_f": torch.cat([coords_normed, std.unsqueeze(1)], dim=-1)})

        self.get_fine_match(coords_normed, data)

    @torch.no_grad()
    def get_fine_match(self, coords_normed, data):
        # mkpts0_f and mkpts1_f
        mkpts0_f = data["mkpts0_c"]
        scale1 = self.scale * data["scale1"][data["b_ids"]] if "scale0" in data else self.scale
        mkpts1_f = data["mkpts1_c"] + (coords_normed * (self.W // 2) * scale1)[: len(data["mconf"])]
        data.update({"mkpts0_f": mkpts0_f, "mkpts1_f": mkpts1_f})


class FineMatching(DDFMatcher):
    """Compatibility alias for fine matcher import."""

    pass
