"""Seleção robusta de dispositivo (GPU com fallback automático para CPU).

Em algumas GPUs do Kaggle (ex.: Tesla P100, capability sm_60) a versão do
PyTorch instalada não possui kernels compatíveis: ``torch.cuda.is_available()``
retorna ``True``, mas a primeira operação na GPU lança
``CUDA error: no kernel image is available for execution on the device``.

Para não deixar o pipeline quebrar nesses casos, testamos a GPU com uma
operação mínima e caímos para CPU se ela falhar.
"""
from __future__ import annotations

import torch


def get_device(verbose: bool = True) -> torch.device:
    """Devolve ``cuda`` se a GPU for utilizável de fato; senão ``cpu``."""
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
            # força o lançamento de um kernel simples para validar compatibilidade
            _ = (torch.zeros(8, device="cuda") + 1).sum().item()
            if verbose:
                print(f"[device] usando GPU: {name}")
            return torch.device("cuda")
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"[device] GPU indisponível/incompatível "
                      f"({type(exc).__name__}: {exc}); usando CPU")
            return torch.device("cpu")
    if verbose:
        print("[device] CUDA não disponível; usando CPU")
    return torch.device("cpu")
