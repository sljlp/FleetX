"""
module for saving for auto parallel infer
"""

import paddle.distributed as dist
import paddle.distributed.fleet as fleet
import re
import paddle
from paddle.distributed.fleet.utils.log_util import logger
import time
import os
import pickle
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_stage3 import GroupShardedStage3

def _is_wrapped_module(model):
    return hasattr(model, "_layer") or hasattr(model, "_layers")

def _get_wrapped_state_dict(dist_model, single_model):
    """ 
    Trasform dist parameters' names in dist model's state_dict to single-card names
    """
    # Insure the dist model are wrapped with distributed apis and the single model not
    assert _is_wrapped_module(dist_model)
    assert not _is_wrapped_module(single_model)

    name_mapping = _get_name_mapping(dist_model, single_model)
    wrapped_state_dict = {}
    state_dict = dist_model.state_dict()

    for k, v in state_dict.items():
        if k in name_mapping:
            logger.debug(f"saving: {k} as: {name_mapping[k]}")
            wrapped_state_dict[name_mapping[k]] = v
    return wrapped_state_dict

# to keep the same parameter name in single models of both dygraph and autoparallel mode, 
# the single model must be created before the dist model
# Question: Why do we need this?
#   Answer: to save model for autoparallel,we need to create model twice -- the single mode and the distributed mode.
#           However, the parameters' name are generated automatically and in a unified order. 
#           For example, if there are 5 linears, the a parameter's name in the first model may be linear_5 
#           while the recosponding name in another model may be linear_10, even though they are the samely 
#           located parameters.
# Question: How to garanetee this?
#   Answer: To check the indices of common name,
#          if there is no index of single is greater than the dist models', 
#          we say the single model is created before the dist modle.
def _check_single_card_model_formmer(single_name, dist_name):
    """
    Description:
        check whether the parameter with single_name is created before the parameter with dist_name.
    Args:
        single_name: A parameter's name in a single model
        dist_name: A parameter's name in a distributed model
    """

    # The names format between splited linear and single model linear are different, so we skip these names and always return True. 
    if re.search("^(column|row)_parallel_linear", dist_name):
        return True

    # match the format as [opname]_[numbers] and extract the number to make a comparation.
    reg = re.compile("[a-z]+_[0-9]+")
    matched1 = reg.search(single_name)
    matched2 = reg.search(dist_name)
    logger.debug(f"checking , {single_name}, {dist_name}")

    assert matched1 is not None and matched2 is not None, f"Cannot find the pattern '[name]_[number]'. " \
        f"single name: {single_name}, dist name: {dist_name}"
    
    name1, idx1 = single_name[matched1.start():matched1.end()].split("_")
    name2, idx2 = dist_name[matched2.start():matched2.end()].split("_")
    assert name1 == name2, f"the input parameters are not same, please check. " \
        f"single name: {name1}, dist name: {name2}"
    return int(idx1) <= int(idx2)
    
def _is_first_used(param):
    return not (hasattr(param, "is_firstly_shared") and not param.is_firstly_shared)

def _is_first_shared(param):
    return hasattr(param, "is_firstly_shared") and param.is_firstly_shared

def _get_name_mapping(dist_model, single_model):
    """
    Description:
        get name mapping from names in dist model to those in single model
    Args:
        dist_model: model in distributed mode
        single_model: model in single mode
    Return:
        dict mapping names in two modles
    """
    
    hcg = fleet.get_hybrid_communicate_group()
    mp_group = hcg.get_model_parallel_group()
    pp_group = hcg.get_pipe_parallel_group()

    # if pp_degree > 1, the dist_model_layers 's type must be
    if pp_group.nranks > 1:
        assert isinstance(dist_model._layers, paddle.distributed.fleet.meta_parallel.parallel_layers.pp_layers.PipelineLayer), \
            f"the pipleline parallel degree is {pp_group.nranks}(larger than 1), but the _layers are not an instance PipeLineLayer," \
            f" please check what's wrong"

    name_mapping = {}
    logger.info("number of single model: " + str(len(single_model.parameters())))
    logger.info("number of distributed model: " + str(len(dist_model.parameters())))

    p_size = len(dist_model.parameters())
    p_size = paddle.to_tensor(p_size)

    # for pipeline paralllel
    # for pp i, we only save parameters from the accumulation before pp i (acc) to acc + len(current dist_model)
    p_sizes = []
    if pp_group.nranks > 1:
        dist.all_gather(p_sizes, p_size, group = pp_group)
        logger.debug("pp size: "+ str(p_sizes))
        pp_rank = dist.get_rank(pp_group)
        acc = sum(p_sizes[:pp_rank])
    else:
        acc = 0
    
    dist_parameters = [
        d for d in dist_model.parameters() if _is_first_used(d)
    ]

    dist_state_keys = [
        k for k, v in dist_model.state_dict().items() if _is_first_used(v)
    ]

    logger.debug(f"len total single: {len(single_model.parameters())}")
    logger.debug(f"start: {acc}, end: {acc+len(dist_parameters)}")
    single_parameters = list(single_model.parameters())[acc: acc+len(dist_parameters)]

    logger.debug(f"len single: {len(single_parameters)}, len dist: {len(dist_parameters)}, len dist keys: {dist_state_keys}")

    for p, k , dp in zip(single_parameters, dist_state_keys , dist_parameters):
        assert _check_single_card_model_formmer(p.name, dp.name), f" single-card model must be built before distributed model" \
            f" the single-card name is '{p.name}' while the dist name is '{dp.name}'"
        name_mapping[k] = p.name
        setattr(dp, "dims_mapping", _get_dims_mapping(dp, p, mp_group))
    return name_mapping

def _get_all_ranks_of_pp(pp_rank):
    """
    Description:
        get all global ranks involving given pp_rank
    """
    hcg = fleet.get_hybrid_communicate_group()
    dp_degree = hcg.get_data_parallel_world_size()
    mp_degree = hcg.get_model_parallel_world_size()
    pp_degree = hcg.get_pipe_parallel_world_size()

    process_group = []
    for i in range(dp_degree):
        for k in range(mp_degree):
            process_group.append(i * dist.get_world_size() // dp_degree \
                + pp_rank * dist.get_world_size() // dp_degree // pp_degree + k)
    return process_group

def _save_param_attr(state_dict, path):
    """
    Description:
        save params' attr dict
    """
    hcg = fleet.get_hybrid_communicate_group()
    dp_degree = hcg.get_data_parallel_world_size()
    mp_degree = hcg.get_model_parallel_world_size()
    
    dp_group = hcg.get_data_parallel_group()
    mp_group = hcg.get_model_parallel_group()
    pp_group = hcg.get_pipe_parallel_group()
    
    logger.debug(f"dp group: {dp_group}")
    logger.debug(f"mp group: {mp_group}")
    logger.debug(f"pp group: {pp_group}")
    
    pp_rank = dist.get_rank(pp_group)

    # Why condition 'pp_rank < 0' exists?
    # Because if pp_degree = 1, pp_rank is set -1 
    pp_rank = 0 if pp_rank <= 0 else pp_rank

    process_group = _get_all_ranks_of_pp(pp_rank)
    
    attr_dict = {}
    for k, v in state_dict.items():
        dims = len(v.shape)
        print("shape: ", k, dims)
        attr_d = {
            "process_shape": [dp_degree, mp_degree] if hcg else [1],
            "process_group": process_group,
            "dims_mapping": v.dims_mapping
        }
        print(v.dims_mapping)
        attr_dict[k] = attr_d

    with open(path, "wb") as f:
        pickle.dump(attr_dict, f)

def _get_dims_mapping(dist_parameter, single_parameter, mp_group):
    """
    Description:
        return the sliting mapping:
            {tensor_name: spiting_strategy}

    Examples:
        spliting_strategy's format (-1, -1, -1, 0), meaing the dims 
        of  the tennsor is 4 and it is splited along the first strategy axis in mesh

    Mesh Examples: (2, 4) means dp=2, mp=4

    """
    
    import numpy as np
    dist_shape = np.array(dist_parameter.shape)
    single_shape = np.array(single_parameter.shape)
    logger.debug(f"single shape: {single_shape}, dist shape: {dist_shape}")
    assert len(dist_shape) == len(single_shape)
    diff = dist_shape - single_shape

    assert (diff <= 0).all(), f"Dist shape is larger than single shape in some axis, please check the shapes. " \
        f"dist shape: {dist_shape}, single shape: {single_shape}"
    assert np.min(diff) == np.sum(diff), f"There are more than one axis along which the tensor is splited, which are not allowed now. " \
         f"dist shape: {dist_shape}, single shape: {single_shape}"

    index = np.argsort(diff)[0]

    if diff[index] < 0:
        assert single_shape[index] % dist_shape[index] == 0 \
            and dist.get_world_size(mp_group) == single_shape[index] // dist_shape[index]
    
    # only tensor for mp is splited, and the mp axis is 1
    mapping = [-1 if d == 0 else 1 for d in diff]
    return mapping

def _get_abs_saved_prefix(path_prefix):
    """
    Description: 
        Get absolute dir path and basename prefix of path_prefix, with making path_prefix's directories.
        If path_prefix is a directory name, basename is set 'saved_parameters'.
        If path_prefix is a file name, basename is extracted from path_prefix.
    Args:
        path_prefix: str
    Return:
        (dirpath: str, basename: str)
    """
    abs_prefix = os.path.abspath(path_prefix)
    if abs_prefix[-1] == os.path.sep:
        save_dir = abs_prefix
        basename_prefix = "saved_parameters"
    else:
        save_dir = os.path.dirname(abs_prefix)
        basename_prefix = os.path.basename(abs_prefix)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir, basename_prefix

def save_for_auto_inference(path_prefix, dist_model, original_model, cvt2cpu=False):
    """
    Description：
        Save model parameters for auto parallel inference.
        Supporting dp + mp + pp + sharding(stage1), dp + sharding stage2-3.
        MoE not sdupported till MoE is supported in auto parallel mode. 
    Args:
        path_prefix: path prefix to save
                    If `path_preifx` ends with path sepreator, 
                        the path is processed as a directory and parameters will be saved in it, 
                        automatically named saved_parameters.
                    Otherwisw, the parameters will be saved with name 
                        path_preifx_dist{global_rank}.pdparams and  path_preifx_dist{global_rank}.pdattrs
    
        dist_model: 
                model in distributed mode
        original_model: 
                model in single-card mode, with no distributed apis
        cvt2cpu: wheather to move parameters to CPU when using sharding stage 3. 
                The var is invalid if not using sharding stage 3. 
    """

    save_dir, basename_prefix = _get_abs_saved_prefix(path_prefix)
    
    if isinstance(dist_model, GroupShardedStage3):
        dist_model.get_all_parameters(cvt2cpu)
    wrapped_dict = _get_wrapped_state_dict(dist_model, original_model)
    global_rank = paddle.distributed.get_rank()

    # save parameters
    paddle.save(wrapped_dict,
                os.path.join(save_dir, f"{basename_prefix}_dist{global_rank}.pdparams"))

    # save attributes
    _save_param_attr(wrapped_dict, os.path.join(save_dir, f"{basename_prefix}_dist{global_rank}.pdattr"))
