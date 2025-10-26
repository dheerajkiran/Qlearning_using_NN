import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from AgentDQN import AgentDQN
from Gridworld import Gridworld


def evaluate_policy(agent, n_episodes, max_steps):
    """
    Run a short greedy evaluation and return the average episode reward.
    """
    total_reward_across_runs = 0
    for _ in range(n_episodes):
        state = env.reset()
        episode_done = False
        episode_reward = 0
        step_counter = 0

        while not episode_done and step_counter < max_steps:
            step_counter += 1
            action = agent.select_greedy_action(state)
            next_state, reward, episode_done = env.step(action)
            episode_reward += reward
            state = next_state

        total_reward_across_runs += episode_reward

    return total_reward_across_runs / n_episodes


if __name__ == "__main__":
    # Try a few discount factors and see how training curves compare.
    discount_choices = [0.90, 0.95, 0.99]

    hidden_layer_count = 2
    hidden_width = 128

    total_episodes = 3000
    max_steps_per_episode = 30
    eval_every = 100  # run a small greedy eval every N episodes

    results_by_gamma = {}

    for gamma in discount_choices:
        print(f"Training DQN with discount factor = {gamma}")

        # Fresh environment for each gamma, with normalized coordinates.
        env = Gridworld(goal=1, penalty=-1)

        # Make runs reproducible-ish.
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)

        agent = AgentDQN(
            state_dim=2,
            action_dim=1,
            n_actions=4,
            layer_size=hidden_width,
            lr=0.001,
            gamma=gamma,
            tau=0.01,
            buffer_size=100000,
            batch_size=32,
            use_cuda=False,
            n_hidden_layers=hidden_layer_count
        )

        eval_scores = np.zeros(total_episodes // eval_every)

        for episode_idx in range(total_episodes):
            state = env.reset()
            episode_reward = 0.0
            episode_done = False
            step_counter = 0

            while not episode_done and step_counter < max_steps_per_episode:
                step_counter += 1

                # Let it explore randomly for the very first few steps to seed the buffer.
                if step_counter < 50:
                    action = np.random.randint(0, agent.n_actions)
                else:
                    # Mild epsilon so it still explores a bit.
                    action = agent.select_action(state, epsilon=0.1)

                next_state, reward, episode_done = env.step(action)

                agent.store_experience(state, action, next_state, reward, episode_done)
                agent.train()

                episode_reward += reward
                state = next_state

            # Periodic quick evaluation (greedy only)
            if (episode_idx + 1) % eval_every == 0:
                eval_slot = ((episode_idx + 1) // eval_every) - 1
                eval_scores[eval_slot] = evaluate_policy(
                    agent, n_episodes=10, max_steps=max_steps_per_episode
                )

        results_by_gamma[gamma] = eval_scores
        print(f"Completed training with discount factor = {gamma}\n")

    # Compare learning curves across discount factors
    plt.figure(figsize=(8, 5))
    for gamma, scores in results_by_gamma.items():
        plt.plot(
            range(eval_every, total_episodes + 1, eval_every),
            scores,
            linewidth=2,
            label=f'gamma={gamma}'
        )
    plt.title('Effect of Discount Factor (gamma) on DQN Learning')
    plt.xlabel('Training Episodes')
    plt.ylabel('Average Evaluation Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig('discount_factor_comparison.png', dpi=300)
    plt.show()
