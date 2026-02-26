class MappingNetwork(nn.Module):
    def __init__(self, z_dim=512, w_dim=512):
        super().__init__()
        layers = []
        for _ in range(8):
            layers.append(nn.Linear(w_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2)) # Standard StyleGAN leakiness
        self.mapping = nn.Sequential(*layers)
    
    def forward(self, z):
        # Normalize the input latent vector
        z = z / (z.norm(dim=1, keepdim=True) + 1e-8)
        return self.mapping(z)

class StyleConv(nn.Module):
    def __init__(self, in_ch, out_ch, w_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, 3, 3))
        self.style = nn.Linear(w_dim, in_ch)
        self.noise_strength = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_ch))
    
    def forward(self, x, w, noise):
        b, c, h, w_ = x.shape
        # Weight demodulation/modulation logic
        style = self.style(w).view(b, 1, c, 1, 1)
        weight = self.weight.unsqueeze(0) * style
        weight = weight.view(-1, c, 3, 3)

        x = x.view(1, -1, h, w_)
        x = F.conv2d(x, weight, padding=1, groups=b)
        x = x.view(b, -1, h, w_)

        x = x + self.noise_strength * noise
        return x + self.bias.view(1, -1, 1, 1)

class Generator(nn.Module):
    def __init__(self, z_dim=512, w_dim=512):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        
        super().__init__()
        self.mapping = MappingNetwork(z_dim, w_dim)
        self.const = nn.Parameter(torch.randn(1, 512, 4, 4))
        self.layers = nn.ModuleList([
            StyleConv(512, 512, w_dim),
            StyleConv(512, 256, w_dim),
            StyleConv(256, 128, w_dim),
            StyleConv(128, 64, w_dim),
        ])
        self.to_rgb = nn.Conv2d(64, 3, 1)

    def forward(self, z):
        w = self.mapping(z)
        x = self.const.repeat(z.size(0), 1, 1, 1)

        for layer in self.layers:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
            x = layer(x, w, noise)
            x = F.leaky_relu(x, 0.2)

        return torch.tanh(self.to_rgb(x))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, 2, 1),
                nn.LeakyReLU(0.2)
            )
        self.net = nn.Sequential(
            block(3, 64),
            block(64, 128),
            block(128, 256),
            block(256, 512),
            nn.Flatten(),
            nn.Linear(512*4*4, 1)
        )

    def forward(self, x):
        return self.net(x)