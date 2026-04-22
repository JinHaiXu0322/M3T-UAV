from yacs.config import CfgNode as CN
_CN = CN()

##############  LoFTDF Pipeline  ##############
_CN.LOFTDF = CN()
_CN.LOFTDF.BACKBONE_TYPE = 'RepVGG'
_CN.LOFTDF.RESOLUTION = (8, 2)  # options: [(8, 2), (16, 4)]
_CN.LOFTDF.FINE_WINDOW_SIZE = 5  # window_size in fine_level, must be odd
_CN.LOFTDF.FINE_CONCAT_COARSE_FEAT = True
_CN.LOFTDF.DEBUG_SHAPES = False

# 1. LoFTDF-backbone (local feature CNN) config
_CN.LOFTDF.RESNETFPN = CN()
_CN.LOFTDF.RESNETFPN.INITIAL_DIM = 128
_CN.LOFTDF.RESNETFPN.BLOCK_DIMS = [128, 196, 256]  # s1, s2, s3
_CN.LOFTDF.REPVGG = CN()
_CN.LOFTDF.REPVGG.STAGE_DIMS = [64, 128, 256]
_CN.LOFTDF.FEAT_ALIGN = CN()
_CN.LOFTDF.FEAT_ALIGN.COARSE_IN_DIM = 256
_CN.LOFTDF.FEAT_ALIGN.FINE_IN_DIM = 256

# 2. LoFTDF-coarse module config
_CN.LOFTDF.COARSE = CN()
_CN.LOFTDF.COARSE.D_MODEL = 256
_CN.LOFTDF.COARSE.D_FFN = 256
_CN.LOFTDF.COARSE.NHEAD = 8
_CN.LOFTDF.COARSE.LAYER_NAMES = ['self', 'cross'] * 4
_CN.LOFTDF.COARSE.ATTENTION = 'linear'  # options: ['linear', 'full']
_CN.LOFTDF.COARSE.TEMP_BUG_FIX = True

# 3. Coarse-Matching config
_CN.LOFTDF.MATCH_COARSE = CN()
_CN.LOFTDF.MATCH_COARSE.THR = 0.2
_CN.LOFTDF.MATCH_COARSE.BORDER_RM = 2
_CN.LOFTDF.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'  # options: ['dual_softmax, 'sinkhorn']
_CN.LOFTDF.MATCH_COARSE.DSMAX_TEMPERATURE = 0.1
_CN.LOFTDF.MATCH_COARSE.SKH_ITERS = 3
_CN.LOFTDF.MATCH_COARSE.SKH_INIT_BIN_SCORE = 1.0
_CN.LOFTDF.MATCH_COARSE.SKH_PREFILTER = False
_CN.LOFTDF.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.2  # training tricks: save GPU memory
_CN.LOFTDF.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 200  # training tricks: avoid DDP deadlock
_CN.LOFTDF.MATCH_COARSE.SPARSE_SPVS = True
_CN.LOFTDF.MATCH_COARSE.MATCHABILITY_DIM = 256

# 4. LoFTDF-fine module config
_CN.LOFTDF.FINE = CN()
_CN.LOFTDF.FINE.D_MODEL = 128
_CN.LOFTDF.FINE.D_FFN = 128
_CN.LOFTDF.FINE.NHEAD = 8
_CN.LOFTDF.FINE.LAYER_NAMES = ['self', 'cross'] * 1
_CN.LOFTDF.FINE.ATTENTION = 'linear'
_CN.LOFTDF.FINE.DDF = CN()
_CN.LOFTDF.FINE.DDF.NUM_LAYERS = 3

# 5. LoFTDF Losses
# -- # coarse-level
_CN.LOFTDF.LOSS = CN()
_CN.LOFTDF.LOSS.COARSE_TYPE = 'focal'  # ['focal', 'cross_entropy']
_CN.LOFTDF.LOSS.COARSE_WEIGHT = 1.0
# _CN.LOFTDF.LOSS.SPARSE_SPVS = False
# -- - -- # focal loss (coarse)
_CN.LOFTDF.LOSS.FOCAL_ALPHA = 0.25
_CN.LOFTDF.LOSS.FOCAL_GAMMA = 2.0
_CN.LOFTDF.LOSS.POS_WEIGHT = 1.0
_CN.LOFTDF.LOSS.NEG_WEIGHT = 1.0
# _CN.LOFTDF.LOSS.DUAL_SOFTMAX = False  # whether coarse-level use dual-softmax or not.
# use `_CN.LOFTDF.MATCH_COARSE.MATCH_TYPE`

# -- # fine-level
_CN.LOFTDF.LOSS.FINE_TYPE = 'l2_with_std'  # ['l2_with_std', 'l2']
_CN.LOFTDF.LOSS.FINE_WEIGHT = 1.0
_CN.LOFTDF.LOSS.FINE_CORRECT_THR = 1.0  # for filtering valid fine-level gts (some gt matches might fall out of the fine-level window)
_CN.LOFTDF.LOSS.DEBUG_PRINT = False


##############  Dataset  ##############
_CN.DATASET = CN()
_CN.DATASET.TRAIN_STAGE = "pretrain"  # options: ['pretrain', 'finetune']
# 1. data config
# training and validating
_CN.DATASET.TRAINVAL_DATA_SOURCE = None  # options: ['M3T', 'MegaDepth']
_CN.DATASET.TRAIN_DATA_ROOT = None
_CN.DATASET.TRAIN_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.TRAIN_NPZ_ROOT = None
_CN.DATASET.TRAIN_LIST_PATH = None
_CN.DATASET.TRAIN_INTRINSIC_PATH = None
_CN.DATASET.VAL_DATA_ROOT = None
_CN.DATASET.VAL_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.VAL_NPZ_ROOT = None
_CN.DATASET.VAL_LIST_PATH = None    # None if val data from all scenes are bundled into a single npz file
_CN.DATASET.VAL_INTRINSIC_PATH = None
# testing
_CN.DATASET.TEST_DATA_SOURCE = None
_CN.DATASET.TEST_DATA_ROOT = None
_CN.DATASET.TEST_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.TEST_NPZ_ROOT = None
_CN.DATASET.TEST_LIST_PATH = None   # None if test data from all scenes are bundled into a single npz file
_CN.DATASET.TEST_INTRINSIC_PATH = None

# 2. dataset config
# general options
_CN.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.4  # discard data with overlap_score < min_overlap_score
_CN.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
_CN.DATASET.AUGMENTATION_TYPE = None  # options: [None, 'dark', 'mobile']

# MegaDepth options
_CN.DATASET.MGDPT_IMG_RESIZE = 640  # resize the longer side, zero-pad bottom-right to square.
_CN.DATASET.MGDPT_IMG_PAD = True  # pad img to square with size = MGDPT_IMG_RESIZE
_CN.DATASET.MGDPT_DEPTH_PAD = True  # pad depthmap to square with size = 2000
_CN.DATASET.MGDPT_DF = 8

# M3T options
_CN.DATASET.M3T_ROOT = "data/m3t"
_CN.DATASET.M3T_TRAIN_MODALITY = "vis"  # ['vis', 'ir', 'both']
_CN.DATASET.M3T_TEST_MODALITIES = ["vis", "ir"]
_CN.DATASET.M3T_GSD_X = 1.0  # meters / pixel
_CN.DATASET.M3T_GSD_Y = 1.0  # meters / pixel
_CN.DATASET.M3T_GEOAUC_T = 50.0
_CN.DATASET.M3T_AUG = CN()
_CN.DATASET.M3T_AUG.ENABLE = True
_CN.DATASET.M3T_AUG.ROTATION_P = 0.4
_CN.DATASET.M3T_AUG.AFFINE_P = 0.4
_CN.DATASET.M3T_AUG.TRANSLATION_P = 0.4
_CN.DATASET.M3T_AUG.ILLUMINATION_P = 0.5
_CN.DATASET.M3T_AUG.NOISE_P = 0.3
_CN.DATASET.M3T_AUG.FOG_P = 0.2
_CN.DATASET.M3T_AUG.MAX_ROT_DEG = 20.0
_CN.DATASET.M3T_AUG.AFFINE_SCALE_MIN = 0.9
_CN.DATASET.M3T_AUG.AFFINE_SCALE_MAX = 1.1
_CN.DATASET.M3T_AUG.MAX_SHEAR_DEG = 8.0
_CN.DATASET.M3T_AUG.MAX_TRANS_RATIO = 0.08
_CN.DATASET.M3T_AUG.MAP_BRIGHTNESS_JITTER = 0.06
_CN.DATASET.M3T_AUG.MAP_CONTRAST_JITTER = 0.06
_CN.DATASET.M3T_AUG.MAP_NOISE_STD = 2.0

##############  Trainer  ##############
_CN.TRAINER = CN()
_CN.TRAINER.WORLD_SIZE = 1
_CN.TRAINER.CANONICAL_BS = 16
_CN.TRAINER.CANONICAL_LR = 2e-3
_CN.TRAINER.SCALING = None  # this will be calculated automatically
_CN.TRAINER.FIND_LR = False  # use learning rate finder from pytorch-lightning

# optimizer
_CN.TRAINER.OPTIMIZER = "adamw"  # [adam, adamw]
_CN.TRAINER.TRUE_LR = None  # this will be calculated automatically at runtime
_CN.TRAINER.ADAM_DECAY = 0.  # ADAM: for adam
_CN.TRAINER.ADAMW_DECAY = 0.1

# step-based warm-up
_CN.TRAINER.WARMUP_TYPE = 'linear'  # [linear, constant]
_CN.TRAINER.WARMUP_RATIO = 0.
_CN.TRAINER.WARMUP_STEP = 4800

# learning rate scheduler
_CN.TRAINER.SCHEDULER = 'MultiStepLR'  # [MultiStepLR, CosineAnnealing, ExponentialLR]
_CN.TRAINER.SCHEDULER_INTERVAL = 'epoch'    # [epoch, step]
_CN.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12]  # MSLR: MultiStepLR
_CN.TRAINER.MSLR_GAMMA = 0.5
_CN.TRAINER.COSA_TMAX = 30  # COSA: CosineAnnealing
_CN.TRAINER.ELR_GAMMA = 0.999992  # ELR: ExponentialLR, this value for 'step' interval

# plotting related
_CN.TRAINER.ENABLE_PLOTTING = True
_CN.TRAINER.N_VAL_PAIRS_TO_PLOT = 32     # number of val/test paris for plotting
_CN.TRAINER.PLOT_MODE = 'evaluation'  # ['evaluation', 'confidence']
_CN.TRAINER.PLOT_MATCHES_ALPHA = 'dynamic'

# geometric metrics and pose solver
_CN.TRAINER.EPI_ERR_THR = 1e-4
_CN.TRAINER.POSE_GEO_MODEL = 'E'  # ['E', 'F', 'H']
_CN.TRAINER.POSE_ESTIMATION_METHOD = 'RANSAC'  # [RANSAC, DEGENSAC, MAGSAC]
_CN.TRAINER.RANSAC_PIXEL_THR = 0.5
_CN.TRAINER.RANSAC_CONF = 0.99999
_CN.TRAINER.RANSAC_MAX_ITERS = 10000
_CN.TRAINER.USE_MAGSACPP = False

# data sampler for train_dataloader
_CN.TRAINER.DATA_SAMPLER = 'scene_balance'  # options: ['scene_balance', 'random', 'normal']
# 'scene_balance' config
_CN.TRAINER.N_SAMPLES_PER_SUBSET = 200
_CN.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = True  # whether sample each scene with replacement or not
_CN.TRAINER.SB_SUBSET_SHUFFLE = True  # after sampling from scenes, whether shuffle within the epoch or not
_CN.TRAINER.SB_REPEAT = 1  # repeat N times for training the sampled data
# 'random' config
_CN.TRAINER.RDM_REPLACEMENT = True
_CN.TRAINER.RDM_NUM_SAMPLES = None

# gradient clipping
_CN.TRAINER.GRADIENT_CLIPPING = 0.5

# reproducibility
# This seed affects the data sampling. With the same seed, the data sampling is promised
# to be the same. When resume training from a checkpoint, it's better to use a different
# seed, otherwise the sampled data will be exactly the same as before resuming, which will
# cause less unique data items sampled during the entire training.
# Use of different seed values might affect the final training result, since not all data items
# are used during training.
_CN.TRAINER.SEED = 66


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()
