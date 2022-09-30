"""
module for save for auto parallen infer
"""

import paddle.distributed as dist
import paddle.distributed.fleet as fleet
import re
from paddle.distributed.fleet.utils.log_util import logger
# logger = get_logger("INFO", "__warp_saver__")

def get_wrapped_state_dict(dist_model, single_model):
    """ 
    Trasform name to single card name according name mapping 
    """

    # if not name_mapping:
    name_mapping = get_name_mapping(dist_model, single_model)
    wrapped_state_dict = {}
    state_dict = dist_model.state_dict()
    for k in state_dict:
        print(k)
    for k in name_mapping:
        print(k)
    for k, v in state_dict.items():
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

def get_name_mapping(dist_model, single_model):
    """
    ge name mapping
    """
    # step one no pipeline parallel
    name_mapping = {}
    print(len(single_model.parameters()))
    print(len(dist_model.parameters()))
    assert len(single_model.parameters()) == len(dist_model.parameters())
    print("=================================================")
    hcg = fleet.get_hybrid_communicate_group()
    if hcg:
        mp_group = hcg.get_model_parallel_group()

    for p, k , dp in zip(single_model.parameters(), dist_model.state_dict().keys(), dist_model.parameters()):
        check_single_card_model_formmer(p.name, dp.name)
        name_mapping[k] = p.name
        print("key:", k, dp.name, dp.shape, p.shape)
        setattr(dp, "dims_mapping", get_dims_mapping(dp, p, mp_group))
    return name_mapping

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
    
    process_group = []
    pp_rank = dist.get_rank(pp_group)
    pp_rank = 0 if pp_rank <= 0 else pp_rank
    for i in range(dp_degree):
        for k in range(mp_degree):
            process_group.append(i * dist.get_world_size() // dp_degree \
                + pp_rank * dist.get_world_size() // dp_degree // pp_degree + k)
    
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
