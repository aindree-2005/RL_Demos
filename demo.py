import gymnasium as gym
env = gym.make("CartPole-v1", render_mode="human")  
obs, info = env.reset()
done = False
while not done:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
    print(f"Observation: {obs}, Reward: {reward}")
env.close()
