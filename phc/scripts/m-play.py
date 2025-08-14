import jax
import jax.numpy as jnp
import mujoco

from loco_mujoco.environments import UnitreeG1
from loco_mujoco.trajectory import Trajectory, TrajectoryInfo, TrajectoryModel, TrajectoryData, TrajectoryHandler

# create the environment
env = UnitreeG1(init_state_type="DefaultInitialStateHandler")

# reset the env
key = jax.random.PRNGKey(0)
env.reset(key)

# get the model and data of the environment
model = env.get_model()
data = env.get_data()

fps = 30.0
traj_handler = TrajectoryHandler(
    model=model,
    traj_path="./datasets/Loco-mj/balance.npz",
    # traj_path="./datasets/Lafan1/dance1_subject1.npz",
    control_dt=1 / fps
)

qpos = traj_handler.traj.data.qpos
qvel = traj_handler.traj.data.qvel

# create a trajectory info -- this stores basic information about the trajectory
njnt = model.njnt
jnt_type = model.jnt_type
jnt_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(njnt)]
traj_info = TrajectoryInfo(jnt_names, model=TrajectoryModel(njnt, jnp.array(jnt_type)), frequency=fps)

# create a trajectory data -- this stores the actual trajectory data
traj_data = TrajectoryData(jnp.array(qpos), jnp.array(qvel), split_points=jnp.array([0, len(qpos)]))

# combine them to a trajectory
traj = Trajectory(traj_info, traj_data)

# example: save the trajectory
#traj.save("trajectory.npz")
#traj = Trajectory.load("trajectory.npz")

# add the trajectory to the environment
env.load_trajectory(traj)

# replay the trajectory
env.play_trajectory(n_steps_per_episode=len(qpos))

""" 
    For G1 23 DOF
    -----------------
    ============ ============================= 
    Index in Obs Name                          
    ============ ============================= 
    0 - 6        q_root                        
    ------------ ----------------------------- 
    7            q_left_hip_pitch_joint        
    ------------ ----------------------------- 
    8            q_left_hip_roll_joint        
    ------------ ----------------------------- 
    9            q_left_hip_yaw_joint         
    ------------ ----------------------------- 
    10           q_left_knee_joint            
    ------------ ----------------------------- 
    11           q_left_ankle_pitch_joint     
    ------------ ----------------------------- 
    12           q_left_ankle_roll_joint      
    ------------ ----------------------------- 
    13           q_right_hip_pitch_joint      
    ------------ ----------------------------- 
    14           q_right_hip_roll_joint       
    ------------ ----------------------------- 
    15           q_right_hip_yaw_joint        
    ------------ ----------------------------- 
    16           q_right_knee_joint           
    ------------ ----------------------------- 
    17           q_right_ankle_pitch_joint    
    ------------ ----------------------------- 
    18           q_right_ankle_roll_joint     
    ------------ ----------------------------- 
    19           q_waist_yaw_joint            
    ------------ ----------------------------- 
    20           q_left_shoulder_pitch_joint  
    ------------ ----------------------------- 
    21           q_left_shoulder_roll_joint   
    ------------ ----------------------------- 
    22           q_left_shoulder_yaw_joint    
    ------------ ----------------------------- 
    23           q_left_elbow_joint           
    ------------ ----------------------------- 
    24           q_left_wrist_roll_joint      
    ------------ ----------------------------- 
    25           q_right_shoulder_pitch_joint  
    ------------ ----------------------------- 
    26           q_right_shoulder_roll_joint  
    ------------ ----------------------------- 
    27           q_right_shoulder_yaw_joint   
    ------------ ----------------------------- 
    28           q_right_elbow_joint          
    ------------ ----------------------------- 
    29           q_right_wrist_roll_joint     
    ------------ ----------------------------- 
"""