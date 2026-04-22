from .repvgg_backbone import RepVGG_8_2
from .resnet_fpn import ResNetFPN_8_2, ResNetFPN_16_4


def build_backbone(config):
    if config['backbone_type'] == 'ResNetFPN':
        if config['resolution'] == (8, 2):
            return ResNetFPN_8_2(config['resnetfpn'])
        elif config['resolution'] == (16, 4):
            return ResNetFPN_16_4(config['resnetfpn'])
    elif config['backbone_type'] == 'RepVGG':
        if config['resolution'] == (8, 2):
            return RepVGG_8_2(config.get('repvgg', {}))
    else:
        raise ValueError(f"LOFTDF.BACKBONE_TYPE {config['backbone_type']} not supported.")

