import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============================
#     Improved CSMLP (MLP)
# ============================

class CSMLP(nn.Module):
    def __init__(self, input_dim=2048, hidden_dims=[800, 300, 150],
                 output_dim=128, dropout=0.3):
        super().__init__()

        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

        # Weight init
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ============================
#       Focal Loss
# ============================

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


# ============================
#   ResNet18 Image Encoder
# ============================

class ResNetImageEncoder(nn.Module):
    def __init__(self, embed_dim=256, freeze_backbone=True):
        super().__init__()

        # Load pretrained ResNet18
        backbone = models.resnet18(weights="DEFAULT")

        # Remove classifier
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        backbone_out_dim = 512

        # Freeze if requested
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        # Projection 512 → embed_dim
        self.fc = nn.Linear(backbone_out_dim, embed_dim)

    def unfreeze(self):
        """Unfreeze backbone for fine-tuning"""
        for param in self.features.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ============================
#     MULTIMODAL MODEL
# ============================

class MultimodalModel(nn.Module):
    def __init__(
        self,
        tabular_dim,
        num_classes=8,
        tabular_hidden_dims=[800, 300, 150],
        tabular_output_dim=128,
        tabular_dropout=0.3,
        image_embed_dim=256,
        freeze_backbone=True
    ):
        super().__init__()

        self.image_branch = ResNetImageEncoder(
            embed_dim=image_embed_dim,
            freeze_backbone=freeze_backbone
        )

        self.tabular_branch = CSMLP(
            input_dim=tabular_dim,
            hidden_dims=tabular_hidden_dims,
            output_dim=tabular_output_dim,
            dropout=tabular_dropout
        )

        self.classifier = nn.Linear(image_embed_dim + tabular_output_dim, num_classes)

    def unfreeze_backbone(self):
        self.image_branch.unfreeze()

    def forward(self, imgs, tabs):
        img_feats = self.image_branch(imgs)
        tab_feats = self.tabular_branch(tabs)
        combined = torch.cat([img_feats, tab_feats], dim=1)
        return self.classifier(combined)
