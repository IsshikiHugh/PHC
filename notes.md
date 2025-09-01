```shell
python \
    phc/run_hydra.py \
    learning=im_mcp_big \
    exp_name=phc_comp_3 \
    env=env_im_getup_mcp \
    robot=smpl_humanoid \
    env.zero_out_far=False \
    robot.real_weight_porpotion_boxes=False \
    env.num_prim=3 \
    env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
    env.models=['output/HumanoidIm/phc_3/Humanoid.pth'] \
    env.num_envs=1 \
    headless=False \
    im_eval=True
```


```shell
python \
    phc/run_hydra.py \
    learning=im_mcp \
    exp_name=phc_shape_mcp_iccv \
    epoch=-1 \
    test=True \
    env=env_im_getup_mcp \
    robot=smpl_humanoid_shape \
    robot.freeze_hand=True \
    robot.box_body=False \
    env.z_activation=relu \
    env.motion_file=data/amass/amass_train_take6_upright.pkl \
    env.models=['output/HumanoidIm/phc_shape_pnn_iccv/Humanoid.pth'] \
    env.num_envs=512 \
    headless=False \
    im_eval=True
```


```shell
CUDA_VISIBLE_DEVICES=3 python \
	phc/run_hydra.py \
    learning=im_mcp \
    exp_name=phc_shape_mcp_iccv \
    epoch=-1 \
    test=True \
    env=env_im_getup_mcp \
    robot=smpl_humanoid_shape \
    robot.freeze_hand=True \
    robot.box_body=False \
    env.z_activation=relu \
    env.motion_file=data/amass/amass_train_take6_upright.pkl \
    env.models=['output/HumanoidIm/phc_shape_pnn_iccv/Humanoid.pth'] \
    env.num_envs=2048 \
    headless=True \
    im_eval=True
    collect_dataset=True
```

```shell
CUDA_VISIBLE_DEVICES=3 python \
	phc/run_hydra.py \
    learning=im_mcp \
    exp_name=phc_shape_mcp_iccv \
    epoch=-1 \
    test=True \
    env=env_im_getup_mcp \
    robot=smpl_humanoid_shape \
    robot.freeze_hand=True \
    robot.box_body=False \
    env.z_activation=relu \
    env.motion_file=data/amass/amass_xy.pkl \
    env.models=['output/HumanoidIm/phc_shape_pnn_iccv/Humanoid.pth'] \
    env.num_envs=512 \
    headless=True \
    im_eval=True \
    collect_dataset=True
```