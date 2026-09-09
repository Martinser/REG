import math

import jittor as jt
import jittor.nn as nn
import numpy as np


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ApproxGELU(nn.Module):
    def execute(self, x):
        # tanh-approx GELU to match PyTorch's approximate="tanh"
        c = np.sqrt(2.0 / np.pi)
        return 0.5 * x * (1.0 + jt.tanh(c * (x + 0.044715 * (x ** 3))))


class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=2, in_chans=4, embed_dim=768, bias=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def execute(self, x):
        x = self.proj(x)
        b, c, h, w = x.shape
        x = x.reshape((b, c, h * w)).transpose((0, 2, 1))
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=False):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
            self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.proj = nn.Linear(dim, dim)
        self.fused_attn = False

    def execute(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape((b, n, 3, self.num_heads, self.head_dim)).permute((2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        attn = jt.matmul(q * self.scale, k.transpose((0, 1, 3, 2)))
        attn = nn.softmax(attn, dim=-1)
        x = jt.matmul(attn, v).transpose((0, 2, 1, 3)).reshape((b, n, c))
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = ApproxGELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def execute(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = jt.exp(-math.log(max_period) * jt.arange(0, half, dtype="float32") / half)
        args = t[:, None].float32() * freqs[None]
        emb = jt.concat([jt.cos(args), jt.sin(args)], dim=-1)
        if dim % 2:
            emb = jt.concat([emb, jt.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def execute(self, t):
        t_freq = self.positional_embedding(t, dim=self.frequency_embedding_size).astype(t.dtype)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = 1 if dropout_prob > 0 else 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = jt.rand((labels.shape[0],)) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        return jt.where(drop_ids, jt.ones_like(labels) * self.num_classes, labels)

    def execute(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class SiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=block_kwargs.get("qk_norm", False),
        )
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(hidden_size, hidden_features=mlp_hidden_dim, out_features=hidden_size)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def execute(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, cls_token_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.linear_cls = nn.Linear(hidden_size, cls_token_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def execute(self, x, c, cls=None):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        if cls is None:
            return self.linear(x), None
        cls_token = self.linear_cls(x[:, 0]).unsqueeze(1)
        x = self.linear(x[:, 1:])
        return x, cls_token.squeeze(1)


class SiT(nn.Module):
    def __init__(
        self,
        path_type="edm",
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        decoder_hidden_size=768,
        encoder_depth=8,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        use_cfg=False,
        z_dims=None,
        projector_dim=2048,
        cls_token_dim=768,
        **block_kwargs,
    ):
        super().__init__()
        if z_dims is None:
            z_dims = [768]
        self.path_type = path_type
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_cfg = use_cfg
        self.num_classes = num_classes
        self.z_dims = z_dims
        self.encoder_depth = encoder_depth

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(jt.zeros((1, num_patches + 1, hidden_size)))
        self.pos_embed.stop_grad()

        self.blocks = nn.ModuleList(
            [SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **block_kwargs) for _ in range(depth)]
        )
        self.projectors = nn.ModuleList([build_mlp(hidden_size, projector_dim, z_dim) for z_dim in z_dims])

        cls_token_dim = self.z_dims[0]
        self.final_layer = FinalLayer(decoder_hidden_size, patch_size, self.out_channels, cls_token_dim)
        self.cls_projectors2 = nn.Linear(in_features=cls_token_dim, out_features=hidden_size, bias=True)
        self.wg_norm = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=True)

        self.initialize_pos_embed()

    def initialize_pos_embed(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.x_embedder.num_patches ** 0.5),
            cls_token=True,
            extra_tokens=1,
        )
        self.pos_embed.assign(jt.array(pos_embed).float32().unsqueeze(0))

    def unpatchify(self, x, patch_size=None):
        c = self.out_channels
        p = self.x_embedder.patch_size[0] if patch_size is None else patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape((x.shape[0], h, w, p, p, c))
        x = x.permute((0, 5, 1, 3, 2, 4))
        return x.reshape((x.shape[0], c, h * p, w * p))

    def execute(self, x, t, y, return_logvar=False, cls_token=None):
        x = self.x_embedder(x)
        if cls_token is None:
            raise RuntimeError("cls_token is required in REG inference.")

        cls_token = self.cls_projectors2(cls_token)
        cls_token = self.wg_norm(cls_token)
        cls_token = cls_token.unsqueeze(1)
        x = jt.concat((cls_token, x), dim=1)
        x = x + self.pos_embed
        n, token_len, d = x.shape

        t_embed = self.t_embedder(t)
        y_embed = self.y_embedder(y, self.is_training())
        c = t_embed + y_embed

        zs = []
        need_projector = self.is_training()
        for i, block in enumerate(self.blocks):
            x = block(x, c)
            if need_projector and (i + 1) == self.encoder_depth:
                zs = [projector(x.reshape((-1, d))).reshape((n, token_len, -1)) for projector in self.projectors]

        x, cls_token = self.final_layer(x, c, cls=cls_token)
        x = self.unpatchify(x)
        return x, zs, cls_token

    def load_state_dict(self, state_dict):
        if not isinstance(state_dict, dict):
            return
        own = self.state_dict()
        loaded, skipped = 0, 0
        for k, v in state_dict.items():
            if k not in own:
                skipped += 1
                continue
            target = own[k]
            src = v if isinstance(v, jt.Var) else jt.array(v)
            if tuple(target.shape) != tuple(src.shape):
                skipped += 1
                continue
            target.assign(src)
            loaded += 1
        print(f"[sit_jittor] load_state_dict loaded={loaded}, skipped={skipped}")


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


def SiT_XL_2(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def SiT_XL_4(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


def SiT_XL_8(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


def SiT_L_2(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def SiT_L_4(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def SiT_L_8(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def SiT_B_2(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def SiT_B_4(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def SiT_B_8(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def SiT_S_2(**kwargs):
    return SiT(depth=12, hidden_size=384, decoder_hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def SiT_S_4(**kwargs):
    return SiT(depth=12, hidden_size=384, decoder_hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def SiT_S_8(**kwargs):
    return SiT(depth=12, hidden_size=384, decoder_hidden_size=384, patch_size=8, num_heads=6, **kwargs)


SiT_models = {
    "SiT-XL/2": SiT_XL_2,
    "SiT-XL/4": SiT_XL_4,
    "SiT-XL/8": SiT_XL_8,
    "SiT-L/2": SiT_L_2,
    "SiT-L/4": SiT_L_4,
    "SiT-L/8": SiT_L_8,
    "SiT-B/2": SiT_B_2,
    "SiT-B/4": SiT_B_4,
    "SiT-B/8": SiT_B_8,
    "SiT-S/2": SiT_S_2,
    "SiT-S/4": SiT_S_4,
    "SiT-S/8": SiT_S_8,
}
