Credit to [PHC](https://github.com/ZhengyiLuo/PHC)

Modifications:
- for numpy >= 1.24, we manully define some missing numpy types
- change matplotlib backend to `TkAgg` for correct 3D plot
- set OMP_NUM_THREADS to 1 for multiprocessing
- simplified the config for retargeting process

For fitting motion data from AMASS:

```bash
cd phc
python scripts/1-fit_smpl_shape.py
python scripts/2-fit_smpl_motion.py +amass_root=./data/AMASS
python scripts/3-vis_q_mj.py +motion_file=./data/g1/amass_all.pkl
```

For motion files or HOI sequences that have keypoints, we can directly solve the dof values from the keypoints instead of forward through smpl model:

```bash
cd phc
python scripts/0-fit_keypoints.py +motion_file=<path_to_motion_file>
python scripts/3-vis_q_mj.py +motion_file=<path_to_motion_file>
```

Feel free to modify this template file and config file for specific needs.