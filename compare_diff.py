import pickle
import numpy as np
import copy
import re

def strip(data):
    assert isinstance(data, dict)
    data.pop("StructuredToParameterName@@", None)

def load_params(path):
    with open(path, "rb") as f:
        data = pickle.load(f, encoding='latin1')
    return data

def parse_rank_auto(dp, mp, pp):
    ranks = list(range(8))
    ranks = ranks[pp*4:pp*4+4]
    ranks = ranks[dp*2:dp*2+2]
    rank = ranks[mp]
    return rank

def test(strategy):
    mp1 = parse_rank_auto(*strategy)
    mp_model = load_params(f"before_train/static_saved_dist{mp1}.pdparams")

    pp0_static = mp_model

    mp1 = parse_rank_dygraph(*strategy)
    mp_model_dy = load_params(f"output_dp2mp2pp2/auto_dist{mp1}.pdparams")

    pp0_dygraph = mp_model_dy

    for k in pp0_static.keys():
        if k in pp0_dygraph:
            d1 = pp0_static[k]
            d2 = pp0_dygraph[k]
            assert np.allclose(d1, d2)
        else:
            print("not matched key:", k)

def parse_rank_dygraph(dp, mp, pp):
    ranks = list(range(4))
    ranks = ranks[pp*2:pp*2+2]
    rank = ranks[mp]
    return rank

if __name__ == "__main__":

    # opts=[]
    # keys = []
    # for i in range(0,4):
    #     opts.append(load_params(f"output_dp2sharding4/epoch_0_step_0/mp_00_sharding_0{i}_pp_00/dist_saved_dist{i}.pdopt"))
    #     print((len(opts[i].keys()) - 2)  % 4)
    #     print((len(opts[i].keys())-2)//4)

    #     opts[i].pop("master_weights", None)
    #     opts[i].pop("LR_Scheduler", None)

    #     keys += list(opts[i].keys())

    # print(keys)

    # merged = load_params("output_dp2sharding4/epoch_0_step_0/mp_00_sharding_00_pp_00/dist_saved_dist0.pdmergedopt")
    m1 = load_params("output_dp2sharding4/epoch_0_step_0/mp_00_sharding_00_pp_00/model_state.pdopt")
    m2 = load_params("output_dp2sharding4/epoch_0_step_0/mp_00_sharding_01_pp_00/model_state.pdopt")
    m3 = load_params("output_dp2sharding4/epoch_0_step_0/mp_00_sharding_02_pp_00/model_state.pdopt")
    m4 = load_params("output_dp2sharding4/epoch_0_step_0/mp_00_sharding_03_pp_00/model_state.pdopt")
    # meta1 = load_params("output_dp2sharding4/epoch_0_step_0/mp_00_sharding_01_pp_00/meta_state.pdopt")
    # meta2 = load_params("output_dp2sharding4/epoch_0_step_0/mp_00_sharding_02_pp_00/meta_state.pdopt")
    # dp2 = load_params("output_dp2/epoch_0_step_0/mp_00_sharding_00_pp_00/model_state.pdopt")
    shdm = load_params("output_dp2sharding4/epoch_0_step_0/mp_00_sharding_00_pp_00/dist_saved.pdopt")
    # shd = load_params("output_dp2sharding4_save/epoch_0_step_0/mp_00_sharding_00_pp_00/dist_saved.pdopt")

    m1.update(m2)
    m1.update(m3)
    m1.update(m4)

    assert len(shdm) == len(m1)

    for k, v in m1.items():
        assert k in shdm, k
        if isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[1], np.ndarray):
            v2 = shdm[k][1]
            assert v[1].dtype == v2.dtype, f"{v[1].dtype} vs {v2.dtype}"
    #     if k not in shd:
    #         head_match = re.search("^.*\.", k)
    #         head = k[head_match.start(): head_match.end(0)]
    #         for k2 in k_list:
    #             matched = re.search(f"^{head}", k2)
    #             # assert matched
    #             if matched:
    #                 print("matched: ", k2, shd[k2])

    # print("--------")
    # for k, v in dp2.items():
    #     if isinstance(v, tuple):
    #         print(k)

    # print(len(dp2), len(shd))
    # m1.update(m2)
    # m1.update(m3)
    # m1.update(m4)
    # print(m1["LR_Scheduler"])

    # for k, v in meta1.items():
    #     print(k, v)
    # for k2, v2 in meta2.items():
        # print(k2, v2)
    # for k, v in merged.items():
    #     if isinstance(v, np.ndarray):
    #         print(k, v.dtype)
    #     if k == "master_weights":
    #         for k1, v1 in v.items():
    #             print(k1, v1.dtype)

    # print(len(merged))

