import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

class ValueNetwork(nn.Module):
    def __init__(self, num_features, hidden_size):
        super(ValueNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PPOPolicyNetwork(nn.Module):
    def __init__(self, num_features, hidden_sizes, num_actions):
        super(PPOPolicyNetwork, self).__init__()
        layers = []
        input_dim = num_features
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


def discount_rewards(rewards, gamma):
    discounted = []
    running_total = 0
    for r in reversed(rewards):
        running_total = r + gamma * running_total
        discounted.insert(0, running_total)
    return torch.tensor(discounted, dtype=torch.float32)


def calculate_advantages(rewards, values, gamma, lam):
    rewards = torch.tensor(rewards, dtype=torch.float32)
    values = torch.tensor(values, dtype=torch.float32)
    advantages = torch.zeros_like(rewards)

    next_value = 0
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        next_value = values[t]

    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)


def run_ppo(env):
    num_features = env.observation_space.shape[0]
    num_actions = env.action_space.n

    value_net = ValueNetwork(num_features, hidden_size=100)
    policy_net = PPOPolicyNetwork(num_features, [40, 35, 30], num_actions)

    value_optimizer = optim.Adam(value_net.parameters(), lr=0.001)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=0.01)

    gamma = 0.98
    lam = 1.0
    epsilon = 0.2

    episode = 1
    running_reward = []

    while True:
        s0, _ = env.reset()
        done = False
        ep_rewards, ep_states, ep_actions = [], [], []
        score = 0

        while not done:
            s0_tensor = torch.tensor(s0, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action_probs = policy_net(s0_tensor).squeeze(0).numpy()
            action = np.random.choice(num_actions, p=action_probs)

            s1, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            ep_states.append(s0)
            ep_actions.append(action)
            ep_rewards.append(r)
            s0 = s1
            score += r

        ep_states_tensor = torch.tensor(np.array(ep_states), dtype=torch.float32)
        ep_actions_tensor = torch.tensor(ep_actions, dtype=torch.long)
        discounted_rewards = discount_rewards(ep_rewards, gamma)

        # Update value network
        for state, target in zip(ep_states_tensor, discounted_rewards):
            value = value_net(state)
            loss = (value - target).pow(2).mean()
            value_optimizer.zero_grad()
            loss.backward()
            value_optimizer.step()

        # Compute advantages
        values = value_net(ep_states_tensor).detach().numpy()
        advantages = calculate_advantages(ep_rewards, values, gamma, lam)

        # Update policy network using clipped PPO objective
        old_probs = policy_net(ep_states_tensor).detach()
        new_probs = policy_net(ep_states_tensor)
        chosen_new_probs = new_probs.gather(1, ep_actions_tensor.unsqueeze(1)).squeeze()
        chosen_old_probs = old_probs.gather(1, ep_actions_tensor.unsqueeze(1)).squeeze()

        ratio = chosen_new_probs / (chosen_old_probs + 1e-8)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        running_reward.append(score)
        if episode % 25 == 0:
            avg_score = np.mean(running_reward[-25:])
            print(f"Episode {episode}, Average Score: {avg_score:.2f}")
            if avg_score >= 500:
                print("Solved!")
        episode += 1


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    run_ppo(env)
