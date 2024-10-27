# Correlation-induced Finite Difference Estimator

## Overview

This repository contains the code implementation for the paper [*A Correlation-induced Finite Difference Estimator*](https://arxiv.org/abs/2405.05638) by Guo Liang, Guangwu Liu, and Kun Zhang. The paper proposes a novel correlation-based finite difference estimator.

### Note on Parameter `pd`

In the `Cor_CFD.m`, the parameter `pd` represents `P0` from the paper, with its standard deviation `sigma`. This parameter may need to be adjusted based on the specific problem context. In general, setting `sigma` to 1 works well for most applications.

## Citation

If you use this code for academic research, please cite the following paper:

```bibtex
@misc{liang2024correlationinducedfinitedifferenceestimator,
      title={A Correlation-induced Finite Difference Estimator}, 
      author={Guo Liang and Guangwu Liu and Kun Zhang},
      year={2024},
      eprint={2405.05638},
      archivePrefix={arXiv},
      primaryClass={stat.ME},
      url={https://arxiv.org/abs/2405.05638}, 
}
