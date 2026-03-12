import torch
import torch.nn as nn


# ──────────────────────────────────────────────
#  GENERATOR
#  Takes: noise vector z (B, Z_DIM) + class label (B,)
#  Outputs: RGB image (B, 3, 64, 64) in range [-1, 1]
# ──────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self, z_dim=128, num_classes=80, embed_dim=64, feature_map=64):
        super().__init__()
        self.z_dim = z_dim

        # Class embedding: maps integer label → dense vector
        self.label_emb = nn.Embedding(num_classes, embed_dim)

        # Input to first conv = z_dim + embed_dim
        in_dim = z_dim + embed_dim

        # Each ConvTranspose2d doubles the spatial size
        # 1x1 → 4x4 → 8x8 → 16x16 → 32x32 → 64x64
        self.net = nn.Sequential(
            # Block 1: 1x1 → 4x4
            nn.ConvTranspose2d(in_dim, feature_map * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_map * 8),
            nn.ReLU(True),

            # Block 2: 4x4 → 8x8
            nn.ConvTranspose2d(feature_map * 8, feature_map * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map * 4),
            nn.ReLU(True),

            # Block 3: 8x8 → 16x16
            nn.ConvTranspose2d(feature_map * 4, feature_map * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map * 2),
            nn.ReLU(True),

            # Block 4: 16x16 → 32x32
            nn.ConvTranspose2d(feature_map * 2, feature_map, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map),
            nn.ReLU(True),

            # Block 5: 32x32 → 64x64 (output)
            nn.ConvTranspose2d(feature_map, 3, 4, 2, 1, bias=False),
            nn.Tanh()  # Output in [-1, 1]
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, z, labels):
        # Embed label and concatenate with noise
        emb = self.label_emb(labels)           # (B, embed_dim)
        x = torch.cat([z, emb], dim=1)         # (B, z_dim + embed_dim)
        x = x.unsqueeze(2).unsqueeze(3)        # (B, C, 1, 1)
        return self.net(x)


# ──────────────────────────────────────────────
#  DISCRIMINATOR
#  Takes: RGB image (B, 3, 64, 64) + class label (B,)
#  Outputs: scalar realness score (B, 1)
# ──────────────────────────────────────────────
class Discriminator(nn.Module):
    def __init__(self, num_classes=80, embed_dim=64, feature_map=64):
        super().__init__()

        # Project label embedding to a spatial map matching input size
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        self.label_proj = nn.Linear(embed_dim, 64 * 64)  # Will reshape to (1, 64, 64)

        # Input channels = 3 (RGB) + 1 (label map) = 4
        self.net = nn.Sequential(
            # 64x64 → 32x32
            nn.Conv2d(4, feature_map, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 → 16x16
            nn.Conv2d(feature_map, feature_map * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 → 8x8
            nn.Conv2d(feature_map * 2, feature_map * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 → 4x4
            nn.Conv2d(feature_map * 4, feature_map * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 4x4 → 1x1
            nn.Conv2d(feature_map * 8, 1, 4, 1, 0, bias=False),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, img, labels):
        # Build spatial label map: (B, 1, 64, 64)
        emb = self.label_emb(labels)                        # (B, embed_dim)
        label_map = self.label_proj(emb)                    # (B, 64*64)
        label_map = label_map.view(-1, 1, 64, 64)           # (B, 1, 64, 64)

        # Concatenate along channel dim → (B, 4, 64, 64)
        x = torch.cat([img, label_map], dim=1)
        return self.net(x).view(-1, 1)                      # (B, 1)