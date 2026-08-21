"""ResNet34 1D para classificação multi-rótulo de arritmias.

Arquitetura seguindo Zhang et al. (2021) / He et al. (2016), adaptada para
sinais 1D de ECG. A rede tem 34 camadas convolucionais organizadas em quatro
estágios residuais empilhados com [3, 4, 6, 3] blocos básicos. Cada bloco
básico contém duas convoluções 1D, duas batch norms, dropout e uma conexão
de atalho (skip connection).

A saída são 9 logits (um por classe diagnóstica). A ativação sigmoide,
mencionada na proposta, é aplicada em inferência (``predict_proba``); durante
o treino usa-se ``BCEWithLogitsLoss`` sobre os logits por estabilidade
numérica.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.config import NUM_CLASSES, NUM_LEADS


def conv3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv1d:
    """Convolução 1D 3x com padding 'same' (kernel=3)."""
    return nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock1d(nn.Module):
    """Bloco residual básico 1D: 2 conv + 2 BN + dropout + skip connection."""

    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 downsample: nn.Module | None = None, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv1 = conv3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = conv3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet34_1D(nn.Module):
    """ResNet34 1D com saída multi-rótulo (9 logits).

    Parameters
    ----------
    num_classes:
        Número de classes diagnósticas (9 no CPSC2018).
    in_channels:
        Número de derivações do ECG (12).
    dropout:
        Probabilidade de dropout dentro de cada bloco residual.
    """

    def __init__(self, num_classes: int = NUM_CLASSES,
                 in_channels: int = NUM_LEADS, dropout: float = 0.2) -> None:
        super().__init__()
        self.in_ch = 64
        self.dropout = dropout

        # Stem: convolução ampla + pooling para reduzir a sequência.
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=15, stride=2,
                               padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 4 estágios residuais (ResNet34: [3, 4, 6, 3] blocos)
        self.layer1 = self._make_layer(64, blocks=3, stride=1)
        self.layer2 = self._make_layer(128, blocks=4, stride=2)
        self.layer3 = self._make_layer(256, blocks=6, stride=2)
        self.layer4 = self._make_layer(512, blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * BasicBlock1d.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, out_ch: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_ch != out_ch * BasicBlock1d.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_ch, out_ch * BasicBlock1d.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch * BasicBlock1d.expansion),
            )
        layers = [BasicBlock1d(self.in_ch, out_ch, stride, downsample, self.dropout)]
        self.in_ch = out_ch * BasicBlock1d.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock1d(self.in_ch, out_ch, dropout=self.dropout))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 12, 15000) -> logits (N, num_classes)."""
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Probabilidades por classe via sigmoide (inferência)."""
        return torch.sigmoid(self.forward(x))


def build_model(**kwargs) -> ResNet34_1D:
    """Fábrica conveniente para instanciar a ResNet34 1D."""
    return ResNet34_1D(**kwargs)
