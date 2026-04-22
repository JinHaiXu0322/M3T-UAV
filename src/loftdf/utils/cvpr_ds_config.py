from yacs.config import CfgNode as CN


def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}


_CN = CN()
_CN.BACKBONE_TYPE = 'RepVGG'
_CN.RESOLUTION = (8, 2)  # options: [(8, 2), (16, 4)]
_CN.FINE_WINDOW_SIZE = 5  # window_size in fine_level, must be odd
_CN.FINE_CONCAT_COARSE_FEAT = True
_CN.DEBUG_SHAPES = False

# 1. LoFTR-backbone (local feature CNN) config
_CN.RESNETFPN = CN()
_CN.RESNETFPN.INITIAL_DIM = 128
_CN.RESNETFPN.BLOCK_DIMS = [128, 196, 256]  # s1, s2, s3
_CN.REPVGG = CN()
_CN.REPVGG.STAGE_DIMS = [64, 128, 256]
_CN.FEAT_ALIGN = CN()
_CN.FEAT_ALIGN.COARSE_IN_DIM = 256
_CN.FEAT_ALIGN.FINE_IN_DIM = 256

# 2. LoFTR-coarse module config
_CN.COARSE = CN()
_CN.COARSE.D_MODEL = 256
_CN.COARSE.D_FFN = 256
_CN.COARSE.NHEAD = 8
_CN.COARSE.LAYER_NAMES = ['self', 'cross'] * 4
_CN.COARSE.ATTENTION = 'linear'  # options: ['linear', 'full']
_CN.COARSE.TEMP_BUG_FIX = False

# 3. Coarse-Matching config
_CN.MATCH_COARSE = CN()
_CN.MATCH_COARSE.THR = 0.2
_CN.MATCH_COARSE.BORDER_RM = 2
_CN.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'  # options: ['dual_softmax, 'sinkhorn']
_CN.MATCH_COARSE.DSMAX_TEMPERATURE = 0.1
_CN.MATCH_COARSE.SKH_ITERS = 3
_CN.MATCH_COARSE.SKH_INIT_BIN_SCORE = 1.0
_CN.MATCH_COARSE.SKH_PREFILTER = True
_CN.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.4  # training tricks: save GPU memory
_CN.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 200  # training tricks: avoid DDP deadlock
_CN.MATCH_COARSE.MATCHABILITY_DIM = 256

# 4. LoFTR-fine module config
_CN.FINE = CN()
_CN.FINE.D_MODEL = 128
_CN.FINE.D_FFN = 128
_CN.FINE.NHEAD = 8
_CN.FINE.LAYER_NAMES = ['self', 'cross'] * 1
_CN.FINE.ATTENTION = 'linear'
_CN.FINE.DDF = CN()
_CN.FINE.DDF.NUM_LAYERS = 3
_CN.LOSS = CN()
_CN.LOSS.DEBUG_PRINT = False

default_cfg = lower_config(_CN)
