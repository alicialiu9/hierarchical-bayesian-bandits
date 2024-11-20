import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, IntSlider

# 0. Initialize
lambda_ = 0.1
d = 10
V0 = lambda_ * np.eye(d)
theta0 = np.zeros(d)
# set number of iterations
n_rounds = 100

# theta* from a multivariable Gaussian distribution
mu = np.zeros(d)  # mean = 0
sigma = np.eye(d)  # covariance matrix = I
theta_star = np.random.multivariate_normal(mu, sigma)


# 1. decision set
def generate_decision_set(d, num_actions=100):
    actions = np.random.randn(num_actions, d)
    actions = actions / np.linalg.norm(actions, axis=1, keepdims=True)  # unit ball
    return actions


# 2. regularized least-squares estimate
def compute_theta(V, actions, rewards):
    return np.linalg.inv(V).dot(actions.T.dot(rewards))


# 3. confidence set
def construct_confidence_set(theta, V, beta):
    return theta, beta * np.linalg.inv(V)  # ellipsoid


# 4. UCB computation
def compute_ucb(actions, theta, invV, beta):
    ucb_values = []
    for a in actions:
        ucb_value = theta.dot(a) + beta * np.sqrt(a.T.dot(invV).dot(a))
        ucb_values.append(ucb_value)
    return np.array(ucb_values)


# compute beta_t (confidence bound parameter)
def compute_beta_t(t, d, lambda_, delta=0.001):
    return np.sqrt(lambda_) + np.sqrt(
        2 * np.log(1 / delta) + d * np.log((1 + t / (lambda_ * d)))
    )


# Run the algorithm for n rounds
actions_list = []
rewards_list = []
total_possible_reward = 0
total_reward = 0
regrets = []
distances = []

for t in range(1, n_rounds + 1):
    At = generate_decision_set(d)
    if t == 1:
        Vt = V0
        theta_t = theta0
    else:
        Vt = V0 + np.dot(np.array(actions_list).T, np.array(actions_list))
        theta_t = compute_theta(Vt, np.array(actions_list), np.array(rewards_list))
    invVt = np.linalg.inv(Vt)
    beta_t = compute_beta_t(t, d, lambda_)
    confidence_set_center, confidence_set_matrix = construct_confidence_set(
        theta_t, Vt, beta_t
    )
    ucb_values = compute_ucb(At, confidence_set_center, invVt, beta_t)
    # 5. select action with highest UCB
    At_selected = At[np.argmax(ucb_values)]
    # 6. stimulate observe reward
    Xt = At_selected.dot(theta_star) + np.random.randn()  # add Gaussian noise

    # optimal reward
    optimal_reward = max(At.dot(theta_star))
    total_possible_reward += optimal_reward
    total_reward += Xt

    regret = total_possible_reward - total_reward
    regrets.append(regret)

    # Calculate distance to true theta
    distance = np.linalg.norm(theta_t - theta_star)
    distances.append(distance)

    # 7. Update history
    actions_list.append(At_selected)
    rewards_list.append(Xt)

# Outputs
print("Estimated parameter vector:", theta_t)
print("True parameter vector:", theta_star)
print("Total Reward:", total_reward)
print("Total Possible Reward:", total_possible_reward)
print("Regret:", regret)

# Plot 1: Visualization of the estimated and true parameter vectors
fig1, ax1 = plt.subplots()
indices = np.arange(d)
width = 0.35

rects1 = ax1.bar(indices - width / 2, theta_star, width, label="True Theta")
rects2 = ax1.bar(indices + width / 2, theta_t, width, label="Estimated Theta")

ax1.set_xlabel("Dimension")
ax1.set_ylabel("Value")
ax1.set_title(f"True vs. Estimated Theta after {n_rounds} Iterations")
ax1.legend()
ax1.set_xticks(indices)
ax1.set_xticklabels(np.arange(1, d + 1))

plt.show()

cumulative_regret = np.cumsum(regrets)
regret_per_trial = cumulative_regret / np.arange(1, n_rounds + 1)

# Plot 2: Regret per trial vs. trials
fig2, ax2 = plt.subplots()
ax2.plot(range(1, n_rounds + 1), regret_per_trial, label="Regret/Trial")
ax2.set_xlabel("Trial")
ax2.set_ylabel("Regret/Trial")
ax2.set_title("Average Regret per Trial vs. Trials")
ax2.legend()

plt.show()

# Plotting the distance to true theta over iterations
plt.figure(figsize=(10, 6))
plt.plot(
    range(1, n_rounds + 1),
    distances,
    label="Distance to True Theta",
    color="red",
)
plt.xlabel("Number of Iterations", fontsize=14, labelpad=10)
plt.ylabel("Distance to True Theta", fontsize=14, labelpad=10)
plt.title("Closeness to True Parameter (Theta)", fontsize=18, pad=20)
plt.show()
