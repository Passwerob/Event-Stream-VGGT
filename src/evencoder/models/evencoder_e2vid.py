import torch
import torch.nn as nn
import torch.nn.functional as F


def skip_concat(x1, x2):
    return torch.cat([x1, x2], dim=1)


def skip_sum(x1, x2):
    return x1 + x2


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2, norm="bn"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        if norm == "bn":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == "gn":
            self.norm = nn.GroupNorm(32, out_channels)
        else:
            self.norm = None
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        return self.act(x)


class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, padding=2, norm="bn"):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvLayer(in_channels, out_channels, kernel_size=kernel_size, padding=padding, norm=norm)

    def forward(self, x):
        x = self.upsample(x)
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, norm="bn"):
        super().__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, padding=1, norm=norm)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        if norm == "bn":
            self.norm = nn.BatchNorm2d(channels)
        elif norm == "gn":
            self.norm = nn.GroupNorm(32, channels)
        else:
            self.norm = None
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.norm is not None:
            out = self.norm(out)
        return self.act(out + x)


class E2VIDEncoder(nn.Module):
    def __init__(
        self,
        num_input_channels,
        num_encoders=4,
        base_num_channels=32,
        num_residual_blocks=2,
        norm="bn",
    ):
        super().__init__()
        self.num_encoders = num_encoders
        self.base_num_channels = base_num_channels
        self.num_residual_blocks = num_residual_blocks

        self.head = ConvLayer(num_input_channels, base_num_channels, kernel_size=5, padding=2, norm=norm)

        encoder_input_sizes = [base_num_channels * (2**i) for i in range(num_encoders)]
        encoder_output_sizes = [base_num_channels * (2 ** (i + 1)) for i in range(num_encoders)]

        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(encoder_input_sizes, encoder_output_sizes):
            self.encoders.append(
                ConvLayer(input_size, output_size, kernel_size=5, stride=2, padding=2, norm=norm)
            )

        self.resblocks = nn.ModuleList(
            [ResidualBlock(encoder_output_sizes[-1], norm=norm) for _ in range(num_residual_blocks)]
        )

        self.encoder_output_sizes = encoder_output_sizes

    def forward(self, x):
        skips = []
        x = self.head(x)
        head = x
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
        for resblock in self.resblocks:
            x = resblock(x)
        return x, head, skips


class E2VIDDecoder(nn.Module):
    def __init__(
        self,
        encoder_output_sizes,
        base_num_channels=32,
        skip_type="sum",
        norm="bn",
        out_channels=1,
    ):
        super().__init__()
        self.skip_type = skip_type
        self.apply_skip = skip_sum if skip_type == "sum" else skip_concat
        self.decoders = nn.ModuleList()
        decoder_input_sizes = list(reversed(encoder_output_sizes))
        for input_size in decoder_input_sizes:
            decoder_in = input_size if skip_type == "sum" else 2 * input_size
            self.decoders.append(
                UpsampleConvLayer(decoder_in, input_size // 2, kernel_size=5, padding=2, norm=norm)
            )

        pred_in = base_num_channels if skip_type == "sum" else 2 * base_num_channels
        self.pred = nn.Conv2d(pred_in, out_channels, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x, head, skips):
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(self.apply_skip(x, skip))
        x = self.apply_skip(x, head)
        return self.activation(self.pred(x))


class EvEncoderE2VID(nn.Module):
    def __init__(
        self,
        in_channels=8,
        base_channels=32,
        out_channels=64,
        project_out_channels=None,
        num_encoders=4,
        num_residual_blocks=2,
        skip_type="sum",
        norm="bn",
        enable_decoder=False,
        recon_out_channels=1,
    ):
        super().__init__()
        self.encoder = E2VIDEncoder(
            num_input_channels=in_channels,
            num_encoders=num_encoders,
            base_num_channels=base_channels,
            num_residual_blocks=num_residual_blocks,
            norm=norm,
        )
        encoder_out_channels = self.encoder.encoder_output_sizes[-1]
        self.out_proj = nn.Conv2d(encoder_out_channels, out_channels, kernel_size=3, padding=1)

        if project_out_channels is not None:
            self.projector = nn.Conv2d(out_channels, project_out_channels, kernel_size=1)
        else:
            self.projector = None

        self.enable_decoder = enable_decoder
        if self.enable_decoder:
            self.decoder = E2VIDDecoder(
                encoder_output_sizes=self.encoder.encoder_output_sizes,
                base_num_channels=base_channels,
                skip_type=skip_type,
                norm=norm,
                out_channels=recon_out_channels,
            )
        else:
            self.decoder = None

    def forward(self, voxel_grid, use_projector=False, return_recon=False):
        B, T, C, H, W = voxel_grid.shape
        z_list = []
        recon_list = []

        for t in range(T):
            x = voxel_grid[:, t]
            latent, head, skips = self.encoder(x)
            z_t = self.out_proj(latent)
            if use_projector and self.projector is not None:
                z_t = self.projector(z_t)
            if return_recon and self.decoder is not None:
                recon_list.append(self.decoder(latent, head, skips))
            z_list.append(z_t)

        z = torch.stack(z_list, dim=1)
        recon = torch.stack(recon_list, dim=1) if recon_list else None
        return z, recon
