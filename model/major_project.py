import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------
# Helper conv block
# -------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel, 1, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

# -------------------
# Multi-scale backbone (simple)
# -------------------
class SimpleBackbone(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        # Example 3 levels: low, mid, high
        self.conv1 = ConvBlock(in_ch, base)          # 1/1
        self.pool1 = nn.MaxPool2d(2)                 # down 2
        self.conv2 = ConvBlock(base, base*2)         # 1/2
        self.pool2 = nn.MaxPool2d(2)                 # down 4
        self.conv3 = ConvBlock(base*2, base*4)       # 1/4
        # optionally more layers

    def forward(self, x):
        f1 = self.conv1(x)   # shape (B, C1, H, W)
        p1 = self.pool1(f1)
        f2 = self.conv2(p1)  # (B, C2, H/2, W/2)
        p2 = self.pool2(f2)
        f3 = self.conv3(p2)  # (B, C3, H/4, W/4)
        return f1, f2, f3

# -------------------
# Fusion Transformer block (cross attention)
# -------------------
class FusionTransformerBlock(nn.Module):
    """
    Cross-attention fusion between two feature maps.
    We flatten HxW -> sequence and perform cross attention.
    We implement symmetric cross-attention: pre->post and post->pre,
    then fuse with residual conv.
    """

    def __init__(self, in_channels, embed_dim=None, nhead=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.in_ch = in_channels
        self.embed_dim = embed_dim or in_channels  # simple projection
        self.qkv_proj = nn.Conv2d(in_channels, self.embed_dim, kernel_size=1)
        self.kv_proj = nn.Conv2d(in_channels, self.embed_dim, kernel_size=1)
        self.attn = nn.MultiheadAttention(self.embed_dim, num_heads=nhead, dropout=dropout, batch_first=True)
        # feed-forward for tokens
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, self.embed_dim),
        )
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        # project token stream back to conv shape
        self.token_to_conv = nn.Conv2d(self.embed_dim, in_channels, kernel_size=1)
        self.res_conv = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)  # fuse residuals

    def forward(self, feat_a, feat_b):
        """
        feat_a, feat_b : (B, C, H, W)
        returns fused_a, fused_b : (B, C, H, W)
        """
        B, C, H, W = feat_a.shape
        # project to embeddings
        a = self.qkv_proj(feat_a)  # (B, E, H, W)
        b = self.kv_proj(feat_b)   # (B, E, H, W)
        E = a.shape[1]
        # ↓ reduce spatial size BEFORE attention
        a = torch.nn.functional.adaptive_avg_pool2d(a, (16,16))
        b = torch.nn.functional.adaptive_avg_pool2d(b, (16,16))

        B, E, H, W = a.shape

        # flatten spatial -> sequences
        a_tokens = a.view(B, E, H*W).permute(0,2,1)  # (B, N, E)
        b_tokens = b.view(B, E, H*W).permute(0,2,1)  # (B, N, E)

        # Cross-attention: a queries, b keys/values
        # MultiheadAttention expects (B, N, E) with batch_first=True
        # produce attended tokens
        a2b_attended, _ = self.attn(query=a_tokens, key=b_tokens, value=b_tokens)  # (B, N, E)
        b2a_attended, _ = self.attn(query=b_tokens, key=a_tokens, value=a_tokens)

        # residual + FFN + norm (token-wise)
        a_res = self.norm1(a_tokens + a2b_attended)
        a_ff = self.ffn(a_res)
        a_final_tokens = self.norm2(a_res + a_ff)  # (B, N, E)

        b_res = self.norm1(b_tokens + b2a_attended)
        b_ff = self.ffn(b_res)
        b_final_tokens = self.norm2(b_res + b_ff)

        # reshape back to conv
        a_conv = a_final_tokens.permute(0,2,1).view(B, E, H, W)
        b_conv = b_final_tokens.permute(0,2,1).view(B, E, H, W)

        # project E->C
        a_back = self.token_to_conv(a_conv)  # (B, C, H, W)
        b_back = self.token_to_conv(b_conv)
        
        # 🔹 upsample back to original feature size
        a_back = torch.nn.functional.interpolate(
            a_back,
            size=feat_a.shape[2:],   # restore original H,W
            mode="bilinear",
            align_corners=False
        )

        b_back = torch.nn.functional.interpolate(
            b_back,
            size=feat_b.shape[2:],
            mode="bilinear",
            align_corners=False
)

        # fuse with original via concat+1x1 conv (learnable)
        a_fused = self.res_conv(torch.cat([feat_a, a_back], dim=1))
        b_fused = self.res_conv(torch.cat([feat_b, b_back], dim=1))

        return a_fused, b_fused

# -------------------
# Full Siamese with Multi-level Fusion
# -------------------
class SiameseMultiFusion(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.backbone = SimpleBackbone(in_ch, base=base)
        # create fusion blocks for each scale (channels must match backbone outputs)
        c1 = base
        c2 = base*2
        c3 = base*4
        self.fuse1 = FusionTransformerBlock(c1, embed_dim=c1, nhead=2)
        self.fuse2 = FusionTransformerBlock(c2, embed_dim=c2, nhead=4)
        self.fuse3 = FusionTransformerBlock(c3, embed_dim=c3, nhead=8)
        # after fusion optionally aggregate multi-scale features
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Linear(c1 + c2 + c3, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(base*4, base*2, 3, padding=1),
            nn.BatchNorm2d(base*2),
            nn.ReLU(inplace=True),

            nn.Conv2d(base*2, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),

            nn.Conv2d(base, 5, 1)   # 5 damage classes
        )

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward_one(self, x):
        f1, f2, f3 = self.backbone(x)
        return f1, f2, f3

    def forward(self, pre, post):
        # extract features (shared backbone)
        pre_f1, pre_f2, pre_f3 = self.forward_one(pre)
        post_f1, post_f2, post_f3 = self.forward_one(post)

        # multi-level fusion (symmetric)
        pre_f1, post_f1 = self.fuse1(pre_f1, post_f1)
        pre_f2, post_f2 = self.fuse2(pre_f2, post_f2)
        pre_f3, post_f3 = self.fuse3(pre_f3, post_f3)
        
        # Change Detection
        change_feat = torch.abs(pre_f3 - post_f3)
        
        # DAMAGE DECODER
        damage_logits = self.decoder(change_feat)
        damage_map = self.upsample(damage_logits)

        # aggregate per-side
        def aggregate(f1, f2, f3):
            # pool each to 1x1 then concat channels
            p1 = self.pool(f1).view(f1.size(0), -1)
            p2 = self.pool(f2).view(f2.size(0), -1)
            p3 = self.pool(f3).view(f3.size(0), -1)
            cat = torch.cat([p1, p2, p3], dim=1)
            emb = self.fc(cat)  # final embedding
            return emb

        emb_pre = aggregate(pre_f1, pre_f2, pre_f3)
        emb_post = aggregate(post_f1, post_f2, post_f3)

        return emb_pre, emb_post, damage_map
    

