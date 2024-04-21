# Simple Example for PPO2 Algorithm

This repository contains a simple implementation of the Proximal Policy Optimization (PPO2) algorithm applied to the LunarLander-v2 environment from OpenAI Gym. The code is structured aim for easy understanding and readability.

## Requirements

To run this code, you need:

- Python 3.x
- PyTorch
- NumPy
- Gym
- tqdm

## Usage

1. Clone this repository:

    ```bash
    git clone https://github.com/example_repo.git
    ```

2. Navigate to the cloned directory:

    ```bash
    cd example_repo
    ```

3. Run the `ppo2.py` script:

    ```bash
    python ppo2.py
    ```

## Code Structure

- `ppo2.py`: This script contains the implementation of the PPO2 algorithm.
- `memory_collector.py`: Memory collector module for collecting training data.
- `helper.py`: Helper functions for saving checkpoints.

## Acknowledgments

This implementation is based on the Proximal Policy Optimization (PPO) algorithm introduced by Schulman et al. in their paper "Proximal Policy Optimization Algorithms" (https://arxiv.org/abs/1707.06347).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.