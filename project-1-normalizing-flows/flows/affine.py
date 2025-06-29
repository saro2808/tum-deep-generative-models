from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .nf_utils import Flow


class Affine(Flow):
    """Affine transformation y = e^a * x + b.

    Args:
        dim (int): dimension of input/output data. int
    """

    def __init__(self, dim: int = 2):
        """Create and init an affine transformation."""
        super().__init__()
        self.dim = dim
        self.log_scale = nn.Parameter(torch.ones(self.dim) * 0.5) # a
        self.shift = nn.Parameter(torch.zeros(self.dim))  # b

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the forward transformation given an input x.

        Args:
            x: input sample. shape [batch_size, dim]

        Returns:
            y: sample after forward transformation. shape [batch_size, dim]
            log_det_jac: log determinant of the jacobian of the forward transformation, shape [batch_size]
        """
        B, D = x.shape

        ##########################################################
        y = torch.exp(self.log_scale) * x + self.shift
        log_det_jac_val = torch.sum(self.log_scale)
        log_det_jac = log_det_jac_val.expand(B)
        ##########################################################

        assert y.shape == (B, D)
        assert log_det_jac.shape == (B,)

        return y, log_det_jac

    def inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the inverse transformation given an input y.

        Args:
            y: input sample. shape [batch_size, dim]

        Returns:
            x: sample after inverse transformation. shape [batch_size, dim]
            inv_log_det_jac: log determinant of the jacobian of the inverse transformation, shape [batch_size]
        """
        B, D = y.shape

        ##########################################################
        x = (y - self.shift) / torch.exp(self.log_scale)
        inv_log_det_jac_val = -torch.sum(self.log_scale)
        inv_log_det_jac = inv_log_det_jac_val.expand(B)
        ##########################################################

        assert x.shape == (B, D)
        assert inv_log_det_jac.shape == (B,)

        return x, inv_log_det_jac
