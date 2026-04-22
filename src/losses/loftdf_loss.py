from loguru import logger

import torch
import torch.nn as nn


class LoFTDFLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # config under the global namespace
        self.loss_config = config['loftdf']['loss']
        self.match_type = self.config['loftdf']['match_coarse']['match_type']
        self.sparse_spvs = self.config['loftdf']['match_coarse']['sparse_spvs']
        
        # coarse-level
        self.correct_thr = self.loss_config['fine_correct_thr']
        self.c_pos_w = self.loss_config['pos_weight']
        self.c_neg_w = self.loss_config['neg_weight']
        # fine-level
        self.fine_type = self.loss_config['fine_type']
        self.eps = 1e-8
        self.debug_print = bool(self.loss_config.get('debug_print', False))

    def compute_coarse_loss(self, conf, conf_gt, weight=None):
        """ Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
        Args:
            conf (torch.Tensor): (N, HW0, HW1) / (N, HW0+1, HW1+1)
            conf_gt (torch.Tensor): (N, HW0, HW1)
            weight (torch.Tensor): (N, HW0, HW1)
        """
        pos_mask, neg_mask = conf_gt == 1, conf_gt == 0
        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w
        # corner case: no gt coarse-level match at all
        if not pos_mask.any():  # assign a wrong gt
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_neg_w = 0.

        if self.loss_config['coarse_type'] == 'cross_entropy':
            assert not self.sparse_spvs, 'Sparse Supervision for cross-entropy not implemented!'
            conf = torch.clamp(conf, 1e-6, 1-1e-6)
            loss_pos = - torch.log(conf[pos_mask])
            loss_neg = - torch.log(1 - conf[neg_mask])
            if weight is not None:
                loss_pos = loss_pos * weight[pos_mask]
                loss_neg = loss_neg * weight[neg_mask]
            return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
        elif self.loss_config['coarse_type'] == 'focal':
            conf = torch.clamp(conf, 1e-6, 1-1e-6)
            alpha = self.loss_config['focal_alpha']
            gamma = self.loss_config['focal_gamma']
            
            if self.sparse_spvs:
                pos_conf = conf[:, :-1, :-1][pos_mask] \
                            if self.match_type == 'sinkhorn' \
                            else conf[pos_mask]
                loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
                # calculate losses for negative samples
                if self.match_type == 'sinkhorn':
                    neg0, neg1 = conf_gt.sum(-1) == 0, conf_gt.sum(1) == 0
                    neg_conf = torch.cat([conf[:, :-1, -1][neg0], conf[:, -1, :-1][neg1]], 0)
                    loss_neg = - alpha * torch.pow(1 - neg_conf, gamma) * neg_conf.log()
                else:
                    # These is no dustbin for dual_softmax, so we left unmatchable patches without supervision.
                    # we could also add 'pseudo negtive-samples'
                    pass
                # handle loss weights
                if weight is not None:
                    # Different from dense-spvs, the loss w.r.t. padded regions aren't directly zeroed out,
                    # but only through manually setting corresponding regions in sim_matrix to '-inf'.
                    loss_pos = loss_pos * weight[pos_mask]
                    if self.match_type == 'sinkhorn':
                        neg_w0 = (weight.sum(-1) != 0)[neg0]
                        neg_w1 = (weight.sum(1) != 0)[neg1]
                        neg_mask = torch.cat([neg_w0, neg_w1], 0)
                        loss_neg = loss_neg[neg_mask]
                
                loss =  c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean() \
                            if self.match_type == 'sinkhorn' \
                            else c_pos_w * loss_pos.mean()
                return loss
                # positive and negative elements occupy similar propotions. => more balanced loss weights needed
            else:  # dense supervision (in the case of match_type=='sinkhorn', the dustbin is not supervised.)
                loss_pos = - alpha * torch.pow(1 - conf[pos_mask], gamma) * (conf[pos_mask]).log()
                loss_neg = - alpha * torch.pow(conf[neg_mask], gamma) * (1 - conf[neg_mask]).log()
                if weight is not None:
                    loss_pos = loss_pos * weight[pos_mask]
                    loss_neg = loss_neg * weight[neg_mask]
                return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
                # each negative element occupy a smaller propotion than positive elements. => higher negative loss weight needed
        else:
            raise ValueError('Unknown coarse loss: {type}'.format(type=self.loss_config['coarse_type']))
        
    def compute_coarse_loss_lf1(self, data):
        """
        L_f1 = loss_pos + 0.5 * loss_neg_uav + 0.5 * loss_neg_map
        - loss_pos: -mean(log(P_ij)) over positive matches
        - loss_neg_uav: -mean(log(1-sigma_uav)) over UAV-side negatives
        - loss_neg_map: -mean(log(1-sigma_map)) over map-side negatives
        """
        P = data['conf_matrix'].clamp(min=self.eps)
        M_mask = data['conf_matrix_gt'].bool()  # [B, N, M]
        sigma_uav = data.get('sigma_uav', None)
        sigma_map = data.get('sigma_map', None)
        if sigma_uav is None or sigma_map is None:
            sigma_uav = torch.full((P.shape[0], P.shape[1]), 0.5, device=P.device, dtype=P.dtype)
            sigma_map = torch.full((P.shape[0], P.shape[2]), 0.5, device=P.device, dtype=P.dtype)
        sigma_uav = sigma_uav.clamp(min=self.eps, max=1 - self.eps)  # [B, N]
        sigma_map = sigma_map.clamp(min=self.eps, max=1 - self.eps)  # [B, M]

        # Positive matching loss
        if M_mask.any():
            loss_pos = -torch.log(P[M_mask]).mean()
        else:
            loss_pos = torch.tensor(0.0, device=P.device)

        # Negative masks from gt assignment
        neg_uav_mask = (data['conf_matrix_gt'].sum(dim=2) == 0)  # [B, N]
        neg_map_mask = (data['conf_matrix_gt'].sum(dim=1) == 0)  # [B, M]

        # UAV-side non-matching loss
        if neg_uav_mask.any():
            loss_neg_uav = -torch.log((1.0 - sigma_uav)[neg_uav_mask]).mean()
        else:
            loss_neg_uav = torch.tensor(0.0, device=P.device)

        # Map-side non-matching loss
        if neg_map_mask.any():
            loss_neg_map = -torch.log((1.0 - sigma_map)[neg_map_mask]).mean()
        else:
            loss_neg_map = torch.tensor(0.0, device=P.device)

        coarse_loss = loss_pos + 0.5 * loss_neg_uav + 0.5 * loss_neg_map
        return coarse_loss, loss_pos, loss_neg_uav, loss_neg_map

    def compute_fine_loss_lf2(self, data):
        """
        L_f2 = mean(||mkpts_f - mkpts_gt||_2)
        Uses refined keypoints on map-side (mkpts1_f) and gt projected points.
        """
        if ('mkpts1_f' not in data) or ('spv_w_pt0_i' not in data) or ('m_bids' not in data) or ('m_iids' not in data):
            return None
        if data['mkpts1_f'].numel() == 0:
            return None

        mkpts_f = data['mkpts1_f']  # [K, 2]
        mkpts_gt = data['spv_w_pt0_i'][data['m_bids'], data['m_iids']]  # [K, 2], pixel coords
        if mkpts_f.shape[0] != mkpts_gt.shape[0]:
            K = min(mkpts_f.shape[0], mkpts_gt.shape[0])
            mkpts_f, mkpts_gt = mkpts_f[:K], mkpts_gt[:K]
        if mkpts_gt.numel() == 0:
            return None
        return torch.linalg.norm(mkpts_f - mkpts_gt, dim=1).mean()

    def _compute_fine_loss_l2(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 2] <x, y>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr
        if correct_mask.sum() == 0:
            if self.training:  # this seldomly happen when training, since we pad prediction with gt
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
            else:
                return None
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask]) ** 2).sum(-1)
        return offset_l2.mean()

    def _compute_fine_loss_l2_std(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 3] <x, y, std>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        # correct_mask tells you which pair to compute fine-loss
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr

        # use std as weight that measures uncertainty
        std = expec_f[:, 2]
        inverse_std = 1. / torch.clamp(std, min=1e-10)
        weight = (inverse_std / torch.mean(inverse_std)).detach()  # avoid minizing loss through increase std

        # corner case: no correct coarse match found
        if not correct_mask.any():
            if self.training:  # this seldomly happen during training, since we pad prediction with gt
                               # sometimes there is not coarse-level gt at all.
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
                weight[0] = 0.
            else:
                return None

        # l2 loss with std
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask, :2]) ** 2).sum(-1)
        loss = (offset_l2 * weight[correct_mask]).mean()

        return loss
    
    @torch.no_grad()
    def compute_c_weight(self, data):
        """ compute element-wise weights for computing coarse-level loss. """
        if 'mask0' in data:
            c_weight = (data['mask0'].flatten(-2)[..., None] * data['mask1'].flatten(-2)[:, None]).float()
        else:
            c_weight = None
        return c_weight

    def forward(self, data):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        loss_scalars = {}

        # 1) coarse-level L_f1
        loss_f1, loss_pos, loss_neg_uav, loss_neg_map = self.compute_coarse_loss_lf1(data)
        loss = loss_f1
        loss_scalars.update({
            "loss_pos": loss_pos.clone().detach().cpu(),
            "loss_neg_uav": loss_neg_uav.clone().detach().cpu(),
            "loss_neg_map": loss_neg_map.clone().detach().cpu(),
            "loss_f1": loss_f1.clone().detach().cpu(),
        })

        # 2) fine-level L_f2
        loss_fine = self.compute_fine_loss_lf2(data)
        if loss_fine is not None:
            loss = loss + loss_fine
            loss_scalars.update({"loss_fine": loss_fine.clone().detach().cpu()})
        else:
            loss_scalars.update({"loss_fine": torch.tensor(0.0)})

        if self.debug_print:
            print("loss_pos:", float(loss_pos.detach()))
            print("loss_neg_uav:", float(loss_neg_uav.detach()))
            print("loss_neg_map:", float(loss_neg_map.detach()))
            if loss_fine is not None:
                print("loss_fine:", float(loss_fine.detach()))

        loss_scalars.update({'loss': loss.clone().detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars})

