import gym
import numpy as np

try:
    env = gym.make('f110_gym:f110-v0', render_mode='human', num_agents=1)
except Exception as e:
    print(f"Error creating environment: {e}")
    print("Please ensure 'f110_gym' is installed and configured correctly.")
    print("You might need to run: pip install f110_gym")
    exit()

ego_pose = np.array([[0.0, 0.0, 0.0]])

obs, reward, done, info = env.reset(ego_pose)

def random_policy(current_obs):
    steering_angle = np.random.uniform(-0.3, 0.3)
    velocity = np.random.uniform(0.5, 1.5)
    return np.array([[steering_angle, velocity]])

lap_time = 0.0
step_count = 0
max_steps = 2000

print("Starting F1TENTH simulation...")

while not done and step_count < max_steps:
    action = random_policy(obs)

    obs, reward, done, info = env.step(action)

    lap_time += reward

    if step_count % 100 == 0:
        print(f"Step: {step_count}, Current Lap Time: {lap_time:.2f}, Done: {done}")

    step_count += 1

print("\nSimulation ended.")
print(f"Final Lap Time: {lap_time:.2f}")

if done:
    print("Reason for termination: Car finished lap or went off track.")
else:
    print(f"Reason for termination: Maximum steps ({max_steps}) reached.")

env.close()
print("Environment closed.")
