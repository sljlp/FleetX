"""
module for save for auto parallen infer
"""

import paddle.distributed as dist
import paddle.distributed.fleet as fleet
import re
import paddle
from paddle.distributed.fleet.utils.log_util import logger
import time
# logger = get_logger("INFO", "__warp_saver__")

def get_wrapped_state_dict(dist_model, single_model):
    """ 
    Trasform name to single card name according name mapping 
    """
    # assert False
    name_mapping = get_name_mapping(dist_model, single_model)
    wrapped_state_dict = {}
    state_dict = dist_model.state_dict()

    for k, v in state_dict.items():
        if k in name_mapping:
            print("saving:", k, "as:", name_mapping[k])
            wrapped_state_dict[name_mapping[k]] = v
    return wrapped_state_dict

def check_single_card_model_formmer(single_name, dist_name):
    if re.search("^(column|row)_parallel_linear", dist_name):
        return
    reg = re.compile("[a-z]+_[0-9]+")
    matched1 = reg.search(single_name)
    matched2 = reg.search(dist_name)
    logger.info(f"checking , {single_name}, {dist_name}")
    if matched1 is not None and matched2 is not None: 
        idx1 = single_name[matched1.start():matched1.end()].split("_")[-1]
        idx2 = dist_name[matched2.start():matched2.end()].split("_")[-1]
        assert int(idx1) <= int(idx2), f" single-card model must be built before distributed model" \
            f" the single-card name is '{single_name}' while the dist name is '{dist_name}'"

def is_first_used(param):
    return not (hasattr(param, "is_firstly_shared") and not param.is_firstly_shared)

def is_first_shared(param):
    return hasattr(param, "is_firstly_shared") and param.is_firstly_shared

def get_name_mapping(dist_model, single_model):
    """
    get name mapping
    """

    print(type(dist_model))
    assert isinstance(dist_model._layers, paddle.distributed.fleet.meta_parallel.parallel_layers.pp_layers.PipelineLayer)
    
    hcg = fleet.get_hybrid_communicate_group()
    if hcg:
        mp_group = hcg.get_model_parallel_group()
        pp_group = hcg.get_pipe_parallel_group()
    # step one no pipeline parallel
    name_mapping = {}
    print(len(single_model.parameters()))
    print(len(dist_model.parameters()))
    p_size = len(dist_model.parameters())
    p_size = paddle.to_tensor(p_size)
    logger.info(p_size)
    p_sizes = []
    dist.all_gather(p_sizes, p_size, group = pp_group)
    print("pp size: ", p_sizes)
    pp_rank = dist.get_rank(pp_group)
    acc = sum(p_sizes[:pp_rank])
    dist_parameters = [
        d for d in dist_model.parameters() if is_first_used(d)
    ]

    dist_state_keys = [
        k for k, v in dist_model.state_dict().items() if is_first_used(v)
    ]

    print("len total single:", len(single_model.parameters()))
    print("start: ", acc, "end:", acc+len(dist_parameters))
    single_parameters = list(single_model.parameters())[acc: acc+len(dist_parameters)]

    print("len single:", len(single_parameters), "len dist:", len(dist_parameters), "len dist keys:", dist_state_keys)

    for p, k , dp in zip(single_parameters, dist_state_keys , dist_parameters):
        # if is_first_shared(dp):
            # for pp_rank in pp_group.ranks:
            #     process_group += get_all_ranks_of_pp(pp_rank)
            # process_group = list(set(process_group))
        check_single_card_model_formmer(p.name, dp.name)
        name_mapping[k] = p.name
        print("key:", k, p.name)
        setattr(dp, "dims_mapping", get_dims_mapping(dp, p, mp_group))
    return name_mapping

def get_all_ranks_of_pp(pp_rank):
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

def save_param_attr(state_dict, path):
    """
    save parap attr dict
    """
    try:
        hcg = fleet.get_hybrid_communicate_group()
        dp_degree = hcg.get_data_parallel_world_size()
        mp_degree = hcg.get_model_parallel_world_size()
        pp_degree = hcg.get_pipe_parallel_world_size()
        
        dp_group = hcg.get_data_parallel_group()
        mp_group = hcg.get_model_parallel_group()
        pp_group = hcg.get_pipe_parallel_group()
        print("dp group:", dp_group)
        print("mp group:", mp_group)
        print("pp group:", pp_group)

    except:
        hcg = None
    
    pp_rank = dist.get_rank(pp_group)
    pp_rank = 0 if pp_rank <= 0 else pp_rank

    process_group = get_all_ranks_of_pp(pp_rank)
    
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
    import pickle
    with open(path, "wb") as f:
        pickle.dump(attr_dict, f)

def get_dims_mapping(dist_parameter, single_parameter, mp_group):
    """
    Description:
        return the sliting mapping:
            {tensor_name: spiting_strategy}
        Examples:
            spliting_strategy's format (-1, -1, -1, 0), meaing the dims of  the tennsor is 4 and it is splited along the first strategy axis in mesh
            mesh examples: (2, 4) may means dp=2, mp=4

    """
    
    import numpy as np
    dist_shape = np.array(dist_parameter.shape)
    single_shape = np.array(single_parameter.shape)
    print(single_shape, dist_shape)
    assert len(dist_shape) == len(single_shape)
    diff = dist_shape - single_shape
    assert (diff <= 0).all()
    assert np.min(diff) == np.sum(diff)
    print("diff:", diff)
    index = np.argsort(diff)[0]
    print("index:", index)
    if diff[index] < 0:
        assert single_shape[index] % dist_shape[index] == 0 \
            and dist.get_world_size(mp_group) == single_shape[index] // dist_shape[index]
    mapping = [-1 if d == 0 else 1 for d in diff]
    return mapping
