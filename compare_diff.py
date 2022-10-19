import pickle
import numpy as np
import copy

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

    model = load_params("output_345/epoch_0_step_0/model.pdparams")
    saved = load_params("pretrained/PaddleFleetX_GPT_345M_220826/mp_00_sharding_00_pp_00/model.pdparams")
    for k1, k2 in zip(model.keys(), saved.keys()):
        print(k1, k2)
        assert k1 == k2
        diff = model[k1].astype("float32") - saved[k2]
        print(np.max(diff), np.min(diff))
    print(len(model.keys()), len(saved.keys()))

    # attr = load_params("")