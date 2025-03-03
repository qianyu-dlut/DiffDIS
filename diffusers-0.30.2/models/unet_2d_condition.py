# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from ..utils import deprecate
from .unets.unet_2d_condition import UNet2DConditionModel, UNet2DConditionOutput
from .unets.unet_2d_condition_ import UNet2DConditionModel_
from .unets.unet_2d_condition_vaefeats import UNet2DConditionModel_extracross
from .unets.unet_2d_condition_extracross import UNet2DConditionModel_extracross_dec
from .unets.unet_2d_condition_midcross import UNet2DConditionModel_midcross
from .unets.unet_2d_condition_sideout import UNet2DConditionModel_sideout
from .unets.unet_2d_condition_diffdis import UNet2DConditionModel_diffdis



class UNet2DConditionOutput(UNet2DConditionOutput):
    deprecation_message = "Importing `UNet2DConditionOutput` from `diffusers.models.unet_2d_condition` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput`, instead."
    deprecate("UNet2DConditionOutput", "0.29", deprecation_message)


class UNet2DConditionModel(UNet2DConditionModel):
    deprecation_message = "Importing `UNet2DConditionModel` from `diffusers.models.unet_2d_condition` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel`, instead."
    deprecate("UNet2DConditionModel", "0.29", deprecation_message)

class UNet2DConditionModel_(UNet2DConditionModel_):
    deprecation_message = "Importing `UNet2DConditionModel_` from `diffusers.models.unet_2d_condition` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel`, instead."
    deprecate("UNet2DConditionModel_", "0.29", deprecation_message)

class UNet2DConditionModel_extracross(UNet2DConditionModel_extracross):
    deprecation_message = "Importing `UNet2DConditionModel_` from `diffusers.models.unet_2d_condition` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel`, instead."
    deprecate("UNet2DConditionModel_extracross", "0.29", deprecation_message)

class UNet2DConditionModel_extracross_dec(UNet2DConditionModel_extracross_dec):
    deprecation_message = "Importing `UNet2DConditionModel_` from `diffusers.models.unet_2d_condition` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel`, instead."
    deprecate("UNet2DConditionModel_extracross_dec", "0.29", deprecation_message)

class UNet2DConditionModel_midcross(UNet2DConditionModel_midcross):
    deprecation_message = "Importing `UNet2DConditionModel_midcross` from `diffusers.models.unet_2d_condition` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel`, instead."
    deprecate("UNet2DConditionModel_midcross", "0.29", deprecation_message)

class UNet2DConditionModel_sideout(UNet2DConditionModel_sideout):
    deprecation_message = "Importing `UNet2DConditionModel_sideout` from `diffusers.models.unet_2d_condition` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel`, instead."
    deprecate("UNet2DConditionModel_sideout", "0.29", deprecation_message)

class UNet2DConditionModel_diffdis(UNet2DConditionModel_diffdis):
    deprecation_message = "Importing `UNet2DConditionModel_sideout` from `diffusers.models.unet_2d_condition` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel`, instead."
    deprecate("UNet2DConditionModel_sideout", "0.29", deprecation_message)
