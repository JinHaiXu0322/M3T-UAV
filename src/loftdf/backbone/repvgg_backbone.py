import torch.nn as nn

from .repvgg import create_RepVGG


class RepVGG_8_2(nn.Module):
    """
    RepVGG backbone adapted for LoFTR two-scale output:
    - coarse feature: 1/8
    - fine feature: 1/2
    """

    def __init__(self, _config):
        super().__init__()
        backbone = create_RepVGG(False)
        self.layer0 = backbone.stage0
        self.layer1 = backbone.stage1
        self.layer2 = backbone.stage2
        self.layer3 = backbone.stage3
        # Learnable upsampling branch: H/8 -> H/4 with refinement
        self.up_8_to_4 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.layer0(x)  # 1/2
        for module in self.layer1:
            out = module(out)  # 1/2
        for module in self.layer2:
            out = module(out)  # 1/4
        for module in self.layer3:
            out = module(out)  # 1/8
        feat_c = out  # [B, 256, H/8, W/8]
        feat_1_4 = self.up_8_to_4(feat_c)  # [B, 256, H/4, W/4]
        return {"feat_c": feat_c, "feat_1_4": feat_1_4}
