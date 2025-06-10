import gym
import numpy as np
import time
from f110_gym.envs.base_classes import Integrator

# Attempt to create the environment
try:
    env = gym.make(
        'f110_gym:f110-v0',
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

# --------DON'T TOUCH ANYTHING ABOVE THIS---------
# -------- RACE CAR STATE CLASS --------
class RacecarState:
    def __init__(self, obs):
        print("\n[RacecarState] Observation received:")
        print(type(obs))
        if isinstance(obs, dict):
            scan = obs['scans'][0]
            num_beams = len(scan)
            self.selected_indices = np.linspace(0, num_beams - 1, num=15, dtype=int)
            self.lidar_data = scan[self.selected_indices]
            self.v_x = obs['linear_vels_x'][0]
            self.w_z = obs['ang_vels_z'][0]
            self.lap_time = obs['lap_times'][0]
        else:
            raise TypeError(f"Unexpected observation type: {type(obs)}")

    def get_state_vector(self):
        state_vector = np.concatenate([
            self.lidar_data,
            [self.v_x],
            [self.w_z]
        ])
        return state_vector
#-------Rewards System---
class RewardSystem:
    def __init__(self, collision_penalty=20.0, lap_completion_bonus=50.0, speed_reward_weight=0.5, stability_penalty_weight=0.2):
        self.collision_penalty = collision_penalty
        self.lap_completion_bonus = lap_completion_bonus
        self.speed_reward_weight = speed_reward_weight
        self.stability_penalty_weight = stability_penalty_weight
        self.prev_lap_time = 0.0
    def calc_rewards(self, obs, done, info):
        if done and obs['collisions'][0]:
            return -self.collision_penalty
        if info['lap_counts'][0] > 0:
            return self.lap_completion_bonus
        reward = obs['linear_vels_x'][0] * self.speed_reward_weight
        reward -= abs(obs['ang_vels_z'][0]) * self.stability_penalty_weight
        return reward
#example
initial_pose = np.array([[0.0, 0.0, 0.0]])  
reset_result = env.reset(poses=initial_pose)
if isinstance(reset_result, tuple):
    obs = reset_result[0]
else:
    obs = reset_result
state_processor = RacecarState(obs)
nn_input = state_processor.get_state_vector()
print(f"\nFinal state vector shape: {nn_input.shape}")