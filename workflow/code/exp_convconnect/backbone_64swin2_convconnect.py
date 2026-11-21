import os
import sys
import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(CURRENT_DIR)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from swin_transformer import SwinTransformer

class SEAttention(nn.Module):
    """通道注意力 (SE)"""
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.size()
        y = x.mean(dim=(2, 3))        # GAP: [B, C]
        y = self.fc(y)                # [B, C]
        y = y.view(B, C, 1, 1)        # 广播到 H,W
        return x * y
class UNet(nn.Module):
    def __init__(self, in_ch=64, base_ch=64):
        super().__init__()

        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # ===== Encoder =====
        self.enc1 = nn.Conv2d(in_ch, base_ch, 3, padding=1)          # 64 → 64
        self.enc2 = nn.Conv2d(base_ch, base_ch * 2, 3, padding=1)    # 64 → 128
        self.enc3 = nn.Conv2d(base_ch * 2, base_ch * 4, 3, padding=1)  # 128 → 256

        # ===== Bottleneck =====
        self.bottleneck = nn.Conv2d(base_ch * 4, base_ch * 8, 3, padding=1)  # 256 → 512

        # ===== Decoder =====
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = nn.Conv2d(base_ch * 8, base_ch * 4, 3, padding=1)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = nn.Conv2d(base_ch * 2, base_ch, 3, padding=1)

        # 输出仍然是 64 维，用来做 residual
        self.out = nn.Conv2d(base_ch, in_ch, 1)

    def forward(self, x, return_bottleneck=False):
        # x: (B, 64, H, W)

        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(self.pool(e1)))
        e3 = self.relu(self.enc3(self.pool(e2)))

        b = self.relu(self.bottleneck(self.pool(e3)))

        if return_bottleneck:
            return self.pool(b)

        d3 = self.relu(self.dec3(torch.cat([e3, self.up3(b)], dim=1)))
        d2 = self.relu(self.dec2(torch.cat([e2, self.up2(d3)], dim=1)))
        d1 = self.relu(self.dec1(torch.cat([e1, self.up1(d2)], dim=1)))

        return self.out(d1)

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class Adapter(nn.Module):
    def __init__(self, in_ch=10, reduction=2):
        super().__init__()
        mid = max(1, in_ch // reduction)
        self.down = nn.Conv2d(in_ch, mid, 1)
        self.up = nn.Conv2d(mid, in_ch, 1)
        self.act = nn.GELU()
        self.bn = nn.BatchNorm2d(in_ch)

    def forward(self, x):
        return x + self.bn(self.up(self.act(self.down(x))))

class STC_DA(nn.Module):
    def __init__(self, in_channels=64, out_classes=4, use_pretrained=False):
        super().__init__()
        self.C = in_channels
        self.out_classes = out_classes
        self.Swin = SwinTransformer(
            img_size=128,
            patch_size=4,
            in_chans=in_channels,
            num_classes=2,
            embed_dim=96,
            depths=[2,2,6,2],
            num_heads=[3,6,12,24],
            window_size=8
        )#UNet(in_ch=in_channels, base_ch=64)
        # 如果需要加载预训练权重
        if use_pretrained:
            self._load_swin_pretrained()
        self.fc1_DA = torch.nn.Linear(12288, 512)
        self.fc2_DA = torch.nn.Linear(768, 128)
        self.fc3_DA = torch.nn.Linear(128, 2)
        # attention per channel
        self.att_fc = nn.Linear(16 * 2 * 2, 1)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, out_classes, 1)  # 最终 logits
        )
        self.norm = nn.LayerNorm(768)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # # segmentation head
        # self.seg_head = nn.Conv2d(self.C, 2, 1)

    def _load_swin_pretrained(self):
        """加载官方 Tiny Swin Transformer 预训练权重，只加载 backbone 部分"""
        print("加载 Swin Transformer 预训练权重...")

        import timm
        from safetensors.torch import load_file

        # 创建官方 Tiny Swin 模型
        model = timm.create_model(
            'swin_tiny_patch4_window7_224.ms_in1k',
            pretrained=False
        )

        # 加载 safetensors 文件
        pre_dict = load_file("@ASSETS@/code/pre_train/swin_tiny_patch4_window7_224.ms_in1k.safetensors")
        model.load_state_dict(pre_dict, strict=False)

        pre_dict = model.state_dict()
        model_dict = self.Swin.state_dict()

        # 跳过 PatchEmbed
        pretrained_dict = {k: v for k, v in pre_dict.items() if "patch_embed" not in k}

        # 只保留形状匹配的权重
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict and model_dict[k].shape == v.shape}

        # 更新模型权重
        model_dict.update(pretrained_dict)
        self.Swin.load_state_dict(model_dict)

        print(f"successful加载 {len(pretrained_dict)} 个权重参数，PatchEmbed 未加载（随机初始化）。")

    def forward(self, x, alpha=None, DA=False):

        if DA:
            return self.forward_DA(x, alpha)

        delta = self.Swin(x)   # (B, 64, H, W)
        Z = self.fuse_conv(torch.cat([x, delta], dim=1))

        seg_out = self.decoder(Z)
        return seg_out


    def forward_DA(self, x, alpha):

        bottleneck = self.Swin(x, return_bottleneck=True)
        bottleneck= self.norm(bottleneck)
        bottleneck = ReverseLayerF.apply(bottleneck, alpha)
        #print(bottleneck.shape)
        x = self.avgpool(bottleneck.transpose(1, 2))
        x = x.flatten(1)
        feat = F.relu(self.fc2_DA(x))
        domain_out = self.fc3_DA(feat)
        return domain_out
