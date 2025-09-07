import joblib
import tyro

import phc
from phc.xy_utils import *


def load_dataset(fn, seq_cnt=1919):
    """seq_cnt is a hack since the dumped results from PHC are redundant in the end."""
    raw_data = joblib.load(fn)
    key_names = raw_data["key_names"][:seq_cnt]
    seq_lens = raw_data["motion_lengths"][:seq_cnt]
    actions = raw_data["clean_action"][:seq_cnt]  # list of (L, 69), since L differs
    obs = raw_data["obs"][:seq_cnt]  # list of (L, 945), since L differs

    data = {
        "key_names": key_names,
        "seq_lens": seq_lens,
        "actions": actions,
        "obs": obs,
    }

    return data


def main(
    input_path: str = "output/HumanoidIm/phc_shape_mcp_iccv/phc_act/amass_xy/AMASS_CMU_act.pkl",
    output_path: str = "data/yx_data/AMASS_CMU_act_processed.npy",
):
    data = load_dataset(input_path)


if __name__ == "__main__":
    tyro.cli(main)
