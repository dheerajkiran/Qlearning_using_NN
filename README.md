# DQN Agent in Gridworld

This project trains a **Deep Q-Network (DQN)** agent to solve a small **3×4 Gridworld** environment with slip dynamics.
The environment includes a goal cell (+1 reward), a penalty cell (−1 reward), and a small step penalty (−0.04).

The main focus is to study how **different discount factors (γ)** influence the agent’s learning behavior.

---

# Project Structure


Qlearning_using_NN/
│
├── src/
│ ├── main.py # Main training and evaluation loop
│ ├── agent_dqn.py # DQN agent with replay buffer and soft updates
│ ├── q_network.py # Neural network for Q-value approximation
│ ├── replay_buffer.py # Experience replay buffer
│ └── gridworld_env.py # 3×4 Gridworld environment with stochastic slips
│
├── results/
│ └── discount_factor_comparison.png # Saved training comparison plot
│
├── README.md
└── requirements.txt


---

# How to Run

1. Clone or download this repository.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the main script:
   cd src
   python main.py

The script will train the DQN agent under multiple discount factors (0.90, 0.95, 0.99) and generate a comparison plot.


---

# Key Components

Replay Buffer for off-policy learning

Soft Target Network Updates (τ = 0.01)

MLP-based Q-Network

Epsilon-Greedy Exploration

Slip Dynamics to make transitions stochastic

---

# Example Output

A sample comparison of learning curves under different discount factors:

results/discount_factor_comparison.png

---

# Dependencies

Python ≥ 3.9

PyTorch

NumPy

Matplotlib

---

# Author Notes

This project was developed as an assignment for academic exploration of reinforcement learning concepts (EEE598: Reinforcement Learning in Robotics), focusing on the trade-off between short-term and long-term rewards in a simple yet interpretable environment.

The uploaded version specifically demonstrates the discount factor comparison, but with minor tweaks to hyperparameters in the code, similar plots for learning rate, exploration rate, batch size, and network depth were obtained.



