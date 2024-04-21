# Simple Example for PPO2 Algorithm

This repository contains a simple implementation of the Proximal Policy Optimization (PPO2) algorithm applied to the LunarLander-v2 environment from OpenAI Gym. The code is structured aim for easy understanding and readability.

## Objective Function Evolution

### Policy Gradient (PG):

$J^{\theta'}(\theta) = E_{(s_t, a_t) \sim \pi_{\theta'}} \left[ \dfrac{P_\theta(a_t \vert s_t)}{P_{\theta'}(a_t \vert s_t)} A^{\theta'}(s_t, a_t) \right]$

$J_{PG}^{\theta^k}(\theta) \approx \sum_{s_t, a_t} \dfrac{p_\theta(a_t \vert s_t)}{p_{\theta^k}(a_t \vert s_t)} A^{\theta^k}(s_t, a_t)$

### Trust Region Policy Optimization (TRPO):

$J^{\theta'}(\theta) = E_{(s_t, a_t) \sim \pi_{\theta'}} \left[ \dfrac{P_\theta(a_t \vert s_t)}{P_{\theta'}(a_t \vert s_t)} A^{\theta'}(s_t, a_t) \right]$

where $KL(\theta' \| \theta) < \delta$

### Proximal Policy Optimization (PPO):

$J^{\theta'}(\theta) = E_{(s_t, a_t) \sim \pi_{\theta'}} \left[ \dfrac{P_\theta(a_t \vert s_t)}{P_{\theta'}(a_t \vert s_t)} A^{\theta'}(s_t, a_t) \right] - \beta KL(\theta' \| \theta)$

$J_{PPO}^{\theta^k}(\theta) \approx \sum_{s_t, a_t} \dfrac{p_\theta(a_t \vert s_t)}{p_{\theta^k}(a_t \vert s_t)} A^{\theta^k}(s_t, a_t) - \beta KL(\theta^k \| \theta)$

### Proximal Policy Optimization 2 (PPO2):

$J_{PPO2}^{\theta^k}(\theta) \approx \sum_{s_t, a_t} \min \left( \dfrac{p_\theta(a_t \vert s_t)}{p_{\theta^k}(a_t \vert s_t)} A^{\theta^k}(s_t, a_t), clip\left(\dfrac{p_\theta(a_t \vert s_t)}{p_{\theta^k}(a_t \vert s_t)}, 1 - \epsilon, 1 + \epsilon \right) A^{\theta^k (s_t, a_t)} \right)$

## Code Structure

- `ppo2.py`: This script contains the implementation of the PPO2 model.
- `memory_collector.py`: Memory collector module for collecting training data.
- `helper.py`: Helper functions for saving checkpoints.
- `main.py`: Entry point for running the PPO2 algorithm.

## Acknowledgments

This implementation is based on the Proximal Policy Optimization (PPO) algorithm introduced by Schulman et al. in their paper "Proximal Policy Optimization Algorithms" (https://arxiv.org/abs/1707.06347).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
