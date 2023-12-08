# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Base class for classifiers.
"""
from abc import ABC
from typing import Any, Dict

from texar.torch.module_base import ModuleBase

__all__ = [
    "ClassifierBase",
]


class ClassifierBase(ModuleBase, ABC):
    r"""Base class inherited by all classifier classes.
    """

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        r"""Returns a dictionary of hyperparameters with default values.
        """
        return {
            "name": "classifier"
        }
