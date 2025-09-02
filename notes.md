## Imitation

```
python phc/run_hydra.py \
    learning=im_mcp_big \
    exp_name=phc_comp_3 \
    env=env_im_getup_mcp \
    robot=smpl_humanoid \
    env.zero_out_far=False \
    robot.real_weight_porpotion_boxes=False \
    env.num_prim=3 \
    env.motion_file=data/amass/amass_xy.pkl \
    env.models=['output/HumanoidIm/phc_3/Humanoid.pth'] \
    env.num_envs=1 \
    headless=False \
    epoch=-1 \
    test=True
```

## Eval AMASS

```
python phc/run_hydra.py \
    learning=im_mcp_big \
    exp_name=phc_comp_3 \
    env=env_im_getup_mcp \
    robot=smpl_humanoid \
    env.zero_out_far=False \
    robot.real_weight_porpotion_boxes=False \
    env.num_prim=3 \
    env.motion_file=data/amass/amass_xy.pkl \
    env.models=['output/HumanoidIm/phc_3/Humanoid.pth'] \
    env.num_envs=512 \
    epoch=-1 \
    headless=True \
    test=True \
    im_eval=True \
    collect_dataset=True \
    no_virtual_display=True
```