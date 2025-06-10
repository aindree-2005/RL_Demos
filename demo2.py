import gym
import numpy as np
import time
from f110_gym.envs.base_classes import Integrator

try:
    env = gym.make('f110_gym:f110-v0', 
                    map="maps/levine",
                    map_ext=".png",
                    num_agents=1,
                    render_mode='human',
                    ego_idx=0,
                    timestep=0.01,
                    integrator=Integrator.RK4 
                    )
except Exception as e:
    print(f"Error creating environment: {e}")
    print("\nTROUBLESHOOTING:")
    print("1. Make sure you have installed f110_gym with GUI dependencies: pip install f110_gym[gui]")
    print("2. Ensure the map file 'maps/levine.png' exists relative to where you run the script.")
    exit()

initial_pose = np.array([[0.0, 0.0, 0.0]])

# Changed this line to only unpack 'obs' as env.reset() seems to return only one value
obs = env.reset(poses=initial_pose)

def random_policy(obs):
    steering_angle = np.random.uniform(-0.3, 0.3)
    velocity = np.random.uniform(0.5, 2.5)
    return np.array([[steering_angle, velocity]])

lap_time = 0.0
step_count = 0
max_steps = 2000

print("Starting F1TENTH simulation with rendering and state monitoring...")

done = False
while not done and step_count < max_steps:
    action = random_policy(obs)
    
    obs, reward, done, info = env.step(action)
    
    env.render()  
    
    lap_time += env.timestep
    step_count += 1
    
    if step_count % 100 == 0 or done:
        print(f"\n--- Step: {step_count} ---")
        print(f"Current Lap Time (seconds): {lap_time:.2f}")
        print(f"Reward (progress): {reward:.4f}")
        pose_x = obs['poses_x'][0] if 'poses_x' in obs else 'N/A'
        pose_y = obs['poses_y'][0] if 'poses_y' in obs else 'N/A'
        pose_theta = obs['poses_theta'][0] if 'poses_theta' in obs else 'N/A'
        print(f"Pose: x={pose_x:.2f}, y={pose_y:.2f}, theta={pose_theta:.2f} rad")
        linear_vel_x = obs['linear_vels_x'][0] if 'linear_vels_x' in obs else 'N/A'
        angular_vel_z = obs['ang_vels_z'][0] if 'ang_vels_z' in obs else 'N/A'
        print(f"Velocity: linear_x={linear_vel_x:.2f} m/s, angular_z={angular_vel_z:.2f} rad/s")
        if 'scans' in obs and obs['scans'][0] is not None:
            lidar_data = obs['scans'][0]
            print(f"LIDAR Scan: {len(lidar_data)} readings, Min: {np.min(lidar_data):.2f}, Max: {np.max(lidar_data):.2f}")
        else:
            print("LIDAR Scan: Not available in observation.")
        collision_status = obs['collisions'][0] if 'collisions' in obs else 'N/A'
        print(f"Collision: {collision_status}")
        if done:
            print(f"\n--- Simulation Terminated at Step {step_count} ---")
            if 'collisions' in obs and obs['collisions'][0] == 1:
                print("Reason: Agent collided.")
            elif 'lap_counts' in info and info['lap_counts'][0] >= 1: 
                 print(f"Reason: Agent completed a lap in {lap_time:.2f} seconds.")
            else:
                print("Reason: Episode terminated (e.g., off-track or max steps reached).")
            print(f"Final Lap Time: {lap_time:.2f} seconds")
            break
    time.sleep(0.01) 
print("\nSimulation ended.")
env.close()
