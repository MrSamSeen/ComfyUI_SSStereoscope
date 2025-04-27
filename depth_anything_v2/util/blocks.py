import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import comfy.ops, but provide a fallback if it's not available
try:
    import comfy.ops
    ops = comfy.ops.manual_cast
except ImportError:
    # Define a simple fallback for ops.Conv2d and ops.Linear
    class OpsFallback:
        @staticmethod
        def Conv2d(*args, **kwargs):
            return nn.Conv2d(*args, **kwargs)

        @staticmethod
        def Linear(*args, **kwargs):
            return nn.Linear(*args, **kwargs)

    ops = OpsFallback()

def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )

    return scratch

class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand:
            out_features = features // 2

        self.out_conv = nn.Sequential(
            ops.Conv2d(
                features,
                out_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
                groups=1,
            ),
            nn.ReLU(inplace=True),
            ops.Conv2d(
                out_features,
                out_features,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                groups=1,
            ),
            nn.ReLU(inplace=True),
        )
        self.size = size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = xs[1]
            output = output + res

        if size is None and self.size is None:
            size = output.shape[-2:]
        elif size is None and self.size is not None:
            size = self.size

        output = nn.functional.interpolate(
            output, size=size, mode="bilinear", align_corners=self.align_corners
        )
        output = self.out_conv(output)
        return output
