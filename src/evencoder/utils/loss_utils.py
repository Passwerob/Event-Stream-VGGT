import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights

class FeatureProjector(nn.Module):
    """
    Aligns the Student's output dimensions to the Teacher's dimensions.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 1x1 Conv to map channels (e. g., 16 -> 1024)
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.proj(x)


class DistillationLoss(nn.Module):
    """
    Computes MSE Loss and optionally KL Divergence. 
    """
    def __init__(self, kl_weight=0.00001, use_kl=False):
        super().__init__()
        self.kl_weight = kl_weight
        self.use_kl = use_kl
        self.mse = nn.MSELoss()

    def forward(self, student_feat, teacher_feat, mean=None, logvar=None):
        """
        Args:
            student_feat: Projected feature from EvEncoder [B*T, N_patches, Embed_dim]
            teacher_feat: Feature from DINOv2 [B*T, N_patches, Embed_dim]
            mean, logvar: Optional latent distribution parameters from Student
        """
        # 1. Ensure same shape
        if student_feat. shape != teacher_feat.shape:
            print(f"[Warning] Shape mismatch: {student_feat.shape} vs {teacher_feat.shape}")
            # This shouldn't happen if alignment is done correctly
        
        # 2. MSE Loss (Reconstruction/Distillation)
        rec_loss = self.mse(student_feat, teacher_feat)

        # 3. KL Divergence Loss (optional)
        if self.use_kl and mean is not None and logvar is not None:
            # Regularize towards N(0, I)
            # Formula: -0.5 * sum(1 + log(var) - mean^2 - var)
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            # Normalize by batch size and dimension
            kl_loss = kl_loss / (mean.numel())
            
            total_loss = rec_loss + self.kl_weight * kl_loss
            return total_loss, rec_loss, kl_loss
        else: 
            # No KL loss
            return rec_loss


class PerceptualLoss(nn.Module):
    """
    Computes perceptual loss using pretrained VGG16 features.
    """
    def __init__(self, layer_ids=(3, 8, 15, 22)):
        super().__init__()
        weights = VGG16_Weights.DEFAULT
        vgg = vgg16(weights=weights).features
        self.slices = nn.ModuleList()
        prev_idx = 0
        for idx in layer_ids:
            self.slices.append(nn.Sequential(*list(vgg.children())[prev_idx:idx]))
            prev_idx = idx
        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer(
            "_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def _prepare(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = (x - self._mean) / self._std
        return x

    def forward(self, pred, target):
        pred = self._prepare(pred)
        target = self._prepare(target)

        loss = 0.0
        for slice_block in self.slices:
            pred = slice_block(pred)
            target = slice_block(target)
            loss = loss + F.l1_loss(pred, target)
        return loss
