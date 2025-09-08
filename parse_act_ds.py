import tyro
import torch
import numpy as np
import smplx
import joblib
from tqdm import tqdm

import phc
from phc.xy_utils import *
from phc.xy_wis3d import HWis3D as Wis3D

SMPL_PATH = 'data/smpl'
smpl_models = {
    'neutral' : smplx.SMPL(model_path=SMPL_PATH, gender='neutral'),
    'male'    : smplx.SMPL(model_path=SMPL_PATH, gender='male'),
    'female'  : smplx.SMPL(model_path=SMPL_PATH, gender='female'),
}

wis3d = Wis3D(seq_name='vis_smpl_params')


def load_dataset(act_fn, smpl_fn, seq_cnt=1919):
    """seq_cnt is a hack since the dumped results from PHC are redundant in the end."""

    # Load action dataset.
    act_raw_data = joblib.load(act_fn)
    key_names = act_raw_data["key_names"][:seq_cnt]
    seq_lens = act_raw_data["motion_lengths"][:seq_cnt]
    actions = act_raw_data["clean_action"][:seq_cnt]  # list of (L, 69), since L differs
    obs = act_raw_data["obs"][:seq_cnt]  # list of (L, 945), since L differs

    # Load SMPL dataset.
    smpl_raw_data = joblib.load(smpl_fn)

    # Repack the data.
    data = {}
    for i, key in enumerate(tqdm(key_names)):
        assert key in smpl_raw_data.keys()
        data[key] = {
            "key_name" : key,
            "seq_len"  : seq_lens[i],
            "action"   : actions[i],
            "obs"      : obs[i],
        }

        cur_smpl_raw_data = smpl_raw_data[key]
        # dict_keys([
        #     "pose_quat_global",  # (L, 24, 4)
        #     "pose_quat", # (L, 24, 4)
        #     "trans_orig", #̦ (L, 3)
        #     "root_trans_offset",  # (L, 3)
        #     "beta",  # (16,)
        #     "gender", # str, in ['neutral', 'male', 'female']
        #     "pose_aa",  (72)
        #     "fps",  # all AMASS propcessed with ZL's script should be 30 fps
        # ])

        data[key]['smpl'] = {
            'poses'  : cur_smpl_raw_data['pose_aa'],     # (L, 72)
            'beta'   : cur_smpl_raw_data['beta'][:10],   # (10,)
            'transl' : cur_smpl_raw_data['trans_orig'],  # (L, 3)
            'gender' : cur_smpl_raw_data['gender'],
        }

    return data


def debug_vis_smpl(data_all, motion_id=0):
    key = list(data_all.keys())[motion_id]
    motion = data_all[key]
    print(f"Visualizing motion {motion_id}: {key}, length {motion['seq_len']}")

    smpl_params = motion['smpl']
    gender = smpl_params['gender']
    smpl_model = smpl_models[gender.lower()]

    transl = torch.from_numpy(smpl_params['transl']).float()  # (L, 3)
    betas  = torch.from_numpy(motion['smpl']['beta']).float()[None].repeat(len(transl), 1)  # (L, 10)
    poses  = torch.from_numpy(motion['smpl']['poses']).float().reshape(-1, 24, 3)  # (L, 24, 3)
    global_orient = poses[:, :1]  # (L, 1, 3)
    body_pose     = poses[:, 1:]  # (L, 21, 3)

    smpl_output = smpl_model(
        betas         = betas,
        body_pose     = body_pose,
        global_orient = global_orient,
        transl        = transl
    )

    # Z-Up
    # wis3d.add_motion_mesh(
    #     verts = smpl_output.vertices.numpy()[:1000],
    #     faces = smpl_model.faces,
    #     name  = 'mesh',
    # )


def main(
    act_ds_path  : str = "output/HumanoidIm/phc_shape_mcp_iccv/phc_act/amass_xy/AMASS_CMU_act.pkl",
    smpl_ds_path : str = "data/amass/amass_xy.pkl",
    output_path  : str = "data/yx_data/AMASS_CMU_act_processed.npy",
):
    data = load_dataset(act_ds_path, smpl_ds_path)
    debug_vis_smpl(data, motion_id=0)


if __name__ == "__main__":
    tyro.cli(main)
