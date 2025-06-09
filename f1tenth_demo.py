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
# Don't touch env setup above this

initial_pose = np.array([[0.0, 0.0, 0.0]])

obs = env.reset(poses=initial_pose)

def random_policy(obs):
    steering_angle = np.random.uniform(-0.3, 0.3)
    velocity = np.random.uniform(0.5, 2.5)
    return np.array([[steering_angle, velocity]])

lap_time = 0.0
step_count = 0
max_steps = 2000

print("Starting F1TENTH simulation with rendering...")

done = False
while not done and step_count < max_steps:
    action = random_policy(obs)
    obs, reward, done, info = env.step(action)
    env.render()  
    lap_time += env.timestep
    step_count += 1
    if step_count % 100 == 0:
        print(f"Step: {step_count}, Lap Time (seconds): {lap_time:.2f}")
    time.sleep(0.01) 
print("\nSimulation ended.")
env.close()