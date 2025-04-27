import torch
import torch.nn as nn
import torch.nn.functional as F
from .dinov2 import DINOv2
from .util.blocks import FeatureFusionBlock, _make_scratch

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

def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )

class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.conv_block = nn.Sequential(
            ops.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv_block(x)

class DPTHead(nn.Module):
    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False,
        is_metric=False
    ):
        super(DPTHead, self).__init__()
        self.use_clstoken = use_clstoken
        self.is_metric=is_metric

        self.projects = nn.ModuleList([
            ops.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            ops.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        ops.Linear(2 * in_channels, in_channels),
                        nn.GELU()))

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32

        self.scratch.output_conv1 = ops.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)

        if self.is_metric:
            self.scratch.output_conv2 = nn.Sequential(
                ops.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                ops.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
            )
        else:
            self.scratch.output_conv2 = nn.Sequential(
                ops.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                ops.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
                nn.ReLU(True),
                nn.Identity(),
            )

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)

        return out

class DepthAnythingV2(nn.Module):
    def __init__(
        self,
        encoder='vitl',
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        use_clstoken=False,
        is_metric=False,
        max_depth=20.0
    ):
        super(DepthAnythingV2, self).__init__()
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
            'vitg': [9, 19, 29, 39]
        }
        self.is_metric = is_metric
        self.max_depth = max_depth
        self.encoder = encoder

        self.pretrained = DINOv2(model_name=encoder)
        self.depth_head = DPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken, is_metric=is_metric)

    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)

        if self.is_metric:
            depth = self.depth_head(features, patch_h, patch_w) * self.max_depth
        else:
            depth = self.depth_head(features, patch_h, patch_w)
            depth = F.relu(depth)

        return depth.squeeze(1)

    def infer_image(self, img):
        """
        Process a raw image (BGR format from cv2) and return a depth map.

        Args:
            img: A numpy array in BGR format (from cv2.imread)

        Returns:
            depth: A numpy array containing the depth map
        """
        # Convert BGR to RGB
        img = img[:, :, ::-1]

        # Prepare image for model
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0

        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std

        # Ensure dimensions are divisible by 14
        h, w = img.shape[1], img.shape[2]
        h_pad, w_pad = 0, 0
        if h % 14 != 0:
            h_pad = 14 - (h % 14)
        if w % 14 != 0:
            w_pad = 14 - (w % 14)

        if h_pad > 0 or w_pad > 0:
            img = F.pad(img, (0, w_pad, 0, h_pad))

        # Add batch dimension
        img = img.unsqueeze(0)

        # Forward pass
        with torch.no_grad():
            depth = self.forward(img)

        # Remove padding if added
        if h_pad > 0 or w_pad > 0:
            depth = depth[:, :h, :w]

        # Convert to numpy
        depth = depth.squeeze().cpu().numpy()

        return depth
