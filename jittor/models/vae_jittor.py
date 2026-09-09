import os

import jittor as jt
import jittor.nn as nn


def silu(x):
    return x * jt.sigmoid(x)


class Upsample2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def execute(self, x):
        b, c, h, w = x.shape
        x = nn.interpolate(x, size=(h * 2, w * 2), mode="nearest")
        x = self.conv(x)
        return x


class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, groups=32, eps=1e-6):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def execute(self, x):
        h = self.conv1(silu(self.norm1(x)))
        h = self.conv2(silu(self.norm2(h)))
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


class AttentionBlock(nn.Module):
    def __init__(self, channels, groups=32, eps=1e-6):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=channels, eps=eps, affine=True)
        self.to_q = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.to_k = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.to_v = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.to_out = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    def execute(self, x):
        b, c, h, w = x.shape
        norm_x = self.group_norm(x)
        q = self.to_q(norm_x).reshape((b, c, h * w)).transpose((0, 2, 1))  # [B, HW, C]
        k = self.to_k(norm_x).reshape((b, c, h * w))  # [B, C, HW]
        v = self.to_v(norm_x).reshape((b, c, h * w)).transpose((0, 2, 1))  # [B, HW, C]

        scale = c ** -0.5
        attn = jt.matmul(q, k) * scale  # [B, HW, HW]
        attn = nn.softmax(attn, dim=-1)
        out = jt.matmul(attn, v).transpose((0, 2, 1)).reshape((b, c, h, w))
        out = self.to_out(out)
        return x + out


class UNetMidBlock2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(channels, channels),
                ResnetBlock2D(channels, channels),
            ]
        )
        self.attentions = nn.ModuleList([AttentionBlock(channels)])

    def execute(self, x):
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x


class UpDecoderBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=3, add_upsample=True):
        super().__init__()
        resnets = []
        ch = in_channels
        for _ in range(num_layers):
            resnets.append(ResnetBlock2D(ch, out_channels))
            ch = out_channels
        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = nn.ModuleList([Upsample2D(out_channels)]) if add_upsample else None

    def execute(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsamplers is not None:
            for up in self.upsamplers:
                x = up(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, block_out_channels=(128, 256, 512, 512), layers_per_block=2):
        super().__init__()
        reversed_channels = list(reversed(block_out_channels))  # [512, 512, 256, 128]
        self.conv_in = nn.Conv2d(in_channels, reversed_channels[0], kernel_size=3, stride=1, padding=1)
        self.mid_block = UNetMidBlock2D(reversed_channels[0])

        self.up_blocks = nn.ModuleList()
        prev_output_channel = reversed_channels[0]
        for i, out_ch in enumerate(reversed_channels):
            is_final = i == (len(reversed_channels) - 1)
            self.up_blocks.append(
                UpDecoderBlock2D(
                    in_channels=prev_output_channel,
                    out_channels=out_ch,
                    num_layers=layers_per_block + 1,
                    add_upsample=not is_final,
                )
            )
            prev_output_channel = out_ch

        self.conv_norm_out = nn.GroupNorm(num_groups=32, num_channels=block_out_channels[0], eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, stride=1, padding=1)

    def execute(self, z):
        x = self.conv_in(z)
        x = self.mid_block(x)
        for up in self.up_blocks:
            x = up(x)
        x = self.conv_out(silu(self.conv_norm_out(x)))
        return x


class AutoencoderKLJittor(nn.Module):
    """
    SD-VAE compatible decoder path used by inference:
    z -> post_quant_conv -> decoder -> image in [-1, 1]
    """

    def __init__(self):
        super().__init__()
        self.post_quant_conv = nn.Conv2d(4, 4, kernel_size=1, stride=1, padding=0)
        self.decoder = Decoder()

    @classmethod
    def from_pretrained(cls, ckpt_path):
        if (ckpt_path is None) or (not os.path.exists(ckpt_path)):
            raise RuntimeError(
                "A converted Jittor VAE checkpoint is required. "
                "Use tools/convert_sd_vae_pt_to_jt.py first and pass --vae-ckpt."
            )
        model = cls()
        raw = jt.load(ckpt_path)
        if isinstance(raw, dict) and "state_dict" in raw:
            raw = raw["state_dict"]
        model.load_state_dict(raw)
        return model

    def load_state_dict(self, state_dict):
        if not isinstance(state_dict, dict):
            raise RuntimeError("VAE checkpoint must be a dict-like state_dict.")
        own = self.state_dict()
        loaded, skipped = 0, 0
        for k, v in state_dict.items():
            if k not in own:
                skipped += 1
                continue
            src = v if isinstance(v, jt.Var) else jt.array(v)
            if tuple(src.shape) != tuple(own[k].shape):
                skipped += 1
                continue
            own[k].assign(src)
            loaded += 1
        print(f"[vae_jittor] load_state_dict loaded={loaded}, skipped={skipped}")

    def decode(self, z):
        z = self.post_quant_conv(z)
        return self.decoder(z)
