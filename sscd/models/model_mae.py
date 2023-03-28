# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import enum
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import resnet18, resnet50, resnext101_32x8d
from classy_vision.models import build_model

from transformers import ViTConfig, ViTModel, ViTMAEModel

class MAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ViTMAEModel.from_pretrained("facebook/vit-mae-base")

        self.fc = nn.Linear(768, 512)
        self.fc_norm = nn.LayerNorm(512)
        nn.init.xavier_uniform_(self.fc.weight)
        
    def forward(self, x):
        x = self.backbone(x)
        x = x['last_hidden_state'][:, 0, :]
        x = self.fc(x)
        x = self.fc_norm(x)
        return x

# class Implementation(enum.Enum):
#     CLASSY_VISION = enum.auto()
#     TORCHVISION = enum.auto()


# class Backbone(enum.Enum):
#     CV_RESNET18 = ("resnet18", 512, Implementation.CLASSY_VISION)
#     CV_RESNET50 = ("resnet50", 2048, Implementation.CLASSY_VISION)
#     CV_RESNEXT101 = ("resnext101_32x4d", 2048, Implementation.CLASSY_VISION)

#     TV_RESNET18 = (resnet18, 512, Implementation.TORCHVISION)
#     TV_RESNET50 = (resnet50, 2048, Implementation.TORCHVISION)
#     TV_RESNEXT101 = (resnext101_32x8d, 2048, Implementation.TORCHVISION)

#     def build(self, dims: int):
#         impl = self.value[2]
#         if impl == Implementation.CLASSY_VISION:
#             model = build_model({"name": self.value[0]})
#             # Remove head exec wrapper, which we don't need, and breaks pickling
#             # (needed for spawn dataloaders).
#             return model.classy_model
#         if impl == Implementation.TORCHVISION:
#             return self.value[0](num_classes=dims, zero_init_residual=True)
#         raise AssertionError("Model implementation not handled: %s" % (self.name,))


# class L2Norm(nn.Module):
#     def forward(self, x):
#         return F.normalize(x)


# class Model(nn.Module):
#     def __init__(self, backbone: str, dims: int, pool_param: float):
#         super().__init__()
#         self.backbone_type = Backbone[backbone]
#         self.backbone = self.backbone_type.build(dims=dims)
#         impl = self.backbone_type.value[2]
#         if impl == Implementation.CLASSY_VISION:
#             self.embeddings = nn.Sequential(
#                 GlobalGeMPool2d(pool_param),
#                 nn.Linear(self.backbone_type.value[1], dims),
#                 L2Norm(),
#             )
#         elif impl == Implementation.TORCHVISION:
#             if pool_param > 1:
#                 self.backbone.avgpool = GlobalGeMPool2d(pool_param)
#                 fc = self.backbone.fc
#                 nn.init.xavier_uniform_(fc.weight)
#                 nn.init.constant_(fc.bias, 0)
#             self.embeddings = L2Norm()

#     def forward(self, x):
#         x = self.backbone(x)
#         return self.embeddings(x)

#     @classmethod
#     def add_arguments(cls, parser: argparse.ArgumentParser):
#         parser = parser.add_argument_group("Model")
#         parser.add_argument(
#             "--backbone", default="TV_RESNET50", choices=[b.name for b in Backbone]
#         )
#         parser.add_argument("--dims", default=512, type=int)
#         parser.add_argument("--pool_param", default=3, type=float)


# if __name__ == "__main__":
#     # model = ViTForImageClassification.from_pretrained("facebook/vit-mae-base")
#     model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
#     import pdb; pdb.set_trace()
