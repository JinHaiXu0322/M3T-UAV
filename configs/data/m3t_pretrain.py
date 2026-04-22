from configs.data.base import cfg


cfg.DATASET.TRAIN_STAGE = "pretrain"
cfg.DATASET.TRAINVAL_DATA_SOURCE = "M3T"
cfg.DATASET.TEST_DATA_SOURCE = "M3T"
cfg.DATASET.M3T_ROOT = "data/m3t"
cfg.DATASET.M3T_TRAIN_MODALITY = "both"
cfg.DATASET.M3T_TEST_MODALITIES = ["vis", "ir"]
cfg.DATASET.M3T_GSD_X = 1.0
cfg.DATASET.M3T_GSD_Y = 1.0
cfg.DATASET.M3T_GEOAUC_T = 50.0

# reuse resize/divisible options from MegaDepth pipeline
cfg.DATASET.MGDPT_IMG_RESIZE = 640
cfg.DATASET.MGDPT_DF = 8
