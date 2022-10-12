# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.fluid.framework import dygraph_only


@dygraph_only
def dist_load(path_prefix, dist_model, load_optimzier=False):
    """
    Description:
        Load parameters from saved model
    """

    if load_optimzier:
        _dist_load_optimizer_state(path_prefix, dist_model)
    _dist_load_parameters(path_prefix, dist_model)


def _dist_load_optimizer_state(path_prefix, dist_model):
    """
    Load optimzier state.
    """
    pass


def _dist_load_parameters(path_preifx, dist_model):
    """
    Load parameters.
    """
    pass
