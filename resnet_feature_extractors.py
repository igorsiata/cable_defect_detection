from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50


class _ResNetFeatureExtractorBase(nn.Module):
    """Patch-level feature extractor based on a ResNet backbone.

    The module captures feature maps from `layer2` and `layer3`, aligns their
    spatial resolution, concatenates channels and returns features as
    `(num_patches, channels)`.
    """

    def __init__(
        self,
        backbone_builder: Callable[..., nn.Module],
        default_weights: Optional[object],
        pretrained: bool = True,
        freeze_backbone: bool = True,
        avg_pool_kernel: int = 3,
        avg_pool_stride: int = 1,
        avg_pool_padding: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.avg_pool_kernel = avg_pool_kernel
        self.avg_pool_stride = avg_pool_stride
        if avg_pool_padding is None:
            # Keep spatial size for odd kernels when stride=1 (e.g., 28x28 stays 28x28).
            avg_pool_padding = avg_pool_kernel // 2 if avg_pool_stride == 1 else 0
        self.avg_pool_padding = avg_pool_padding

        weights = default_weights if pretrained else None
        self.backbone = backbone_builder(weights=weights)

        if freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        def _extract() -> tuple[Tensor, Tensor]:
            out = self.backbone.conv1(x)
            out = self.backbone.bn1(out)
            out = self.backbone.relu(out)
            out = self.backbone.maxpool(out)
            out = self.backbone.layer1(out)
            fmap2 = self.backbone.layer2(out)
            fmap3 = self.backbone.layer3(fmap2)
            return fmap2, fmap3

        if self.freeze_backbone:
            with torch.no_grad():
                fmap2, fmap3 = _extract()
        else:
            fmap2, fmap3 = _extract()

        pooled = [
            F.avg_pool2d(
                fmap,
                self.avg_pool_kernel,
                stride=self.avg_pool_stride,
                padding=self.avg_pool_padding,
            )
            for fmap in (fmap2, fmap3)
        ]
        fmap_size = pooled[0].shape[-2:]
        resized = [F.adaptive_avg_pool2d(fmap, fmap_size) for fmap in pooled]

        patch = torch.cat(resized, dim=1)
        patch = patch.reshape(patch.shape[1], -1).T
        return patch

    def get_config(self) -> Dict[str, Union[bool, int]]:
        return {
            "pretrained": self.pretrained,
            "freeze_backbone": self.freeze_backbone,
            "avg_pool_kernel": self.avg_pool_kernel,
            "avg_pool_stride": self.avg_pool_stride,
            "avg_pool_padding": self.avg_pool_padding,
        }

    def save(self, path: Union[str, Path]) -> None:
        payload = {
            "state_dict": self.state_dict(),
            "config": self.get_config(),
        }
        torch.save(payload, str(path))

    def export_onnx(
        self,
        path: Union[str, Path],
        input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
        opset_version: int = 17,
        dynamic_batch: bool = True,
        dynamic_hw: bool = False,
    ) -> None:
        """Export model to ONNX format."""
        self.eval()
        device = next(self.parameters()).device
        dummy_input = torch.randn(*input_shape, device=device)

        dynamic_axes = None
        if dynamic_batch or dynamic_hw:
            dynamic_axes = {"input": {}, "features": {}}
            if dynamic_batch:
                dynamic_axes["input"][0] = "batch"
                dynamic_axes["features"][0] = "num_patches"
            if dynamic_hw:
                dynamic_axes["input"][2] = "height"
                dynamic_axes["input"][3] = "width"

        torch.onnx.export(
            self,
            dummy_input,
            str(path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["features"],
            dynamic_axes=dynamic_axes,
        )

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> "_ResNetFeatureExtractorBase":
        payload = torch.load(str(path), map_location=map_location)

        config = payload.get("config", {})
        config["pretrained"] = False

        model = cls(**config)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model


class ResNet50FeatureExtractor(_ResNetFeatureExtractorBase):
    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        avg_pool_kernel: int = 3,
        avg_pool_stride: int = 1,
        avg_pool_padding: Optional[int] = None,
    ) -> None:
        super().__init__(
            backbone_builder=resnet50,
            default_weights=ResNet50_Weights.DEFAULT,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            avg_pool_kernel=avg_pool_kernel,
            avg_pool_stride=avg_pool_stride,
            avg_pool_padding=avg_pool_padding,
        )


class ResNet18FeatureExtractor(_ResNetFeatureExtractorBase):
    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        avg_pool_kernel: int = 3,
        avg_pool_stride: int = 1,
        avg_pool_padding: Optional[int] = None,
    ) -> None:
        super().__init__(
            backbone_builder=resnet18,
            default_weights=ResNet18_Weights.DEFAULT,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            avg_pool_kernel=avg_pool_kernel,
            avg_pool_stride=avg_pool_stride,
            avg_pool_padding=avg_pool_padding,
        )

if __name__ == "__main__":
    model = ResNet18FeatureExtractor(pretrained=True, freeze_backbone=True)
    # model.load_state_dict(torch.load("resnet50_extractor_weights.pth", map_location="cpu"))
    model.export_onnx(
        path="resnet18_extractor.onnx",
        input_shape=(1, 3, 224, 224),
        opset_version=18,
        dynamic_batch=True,
        dynamic_hw=False,
        # use_external_data_format=False,
    )