# Hierarchical Bayesian Approach to Linear Bandits for Decision Support
---
## Overview

This project is an implementation of a Hierarchical Bayesian model for multi-armed bandit problems for decision support, including code for Linear UCB and multi-client UCB-type Bayesian models, parameter tuning, and regret analysis.

---

## Table of Contents

1. [Why This Project?](#why-this-project)
2. [Features](#features)
3. [How to Run](#how-to-run)
4. [How to Shutdown](#how-to-shutdown)
5. [Files and Structure](#files-and-structure)
6. [Authors](#authors)

---

## Why This Project?

Multi-armed bandit problems are a cornerstone of decision-making under uncertainty, with applications ranging from risk management to resource allocation. By enhancing existing models with hierarchical Bayesian approaches, this project aims to:
- Reduce parameter assumptions.
- Improve model adaptability to diverse, real-world datasets.
- Demonstrate the benefits of collaborative multi-client training.

---

## Features

- **Linear UCB Algorithm**: Implements a single-client model for baseline analysis.
- **Hierarchical Bayesian Model**: Introduces a multi-client model with Gaussian hyperparameters for improved accuracy.
- **Cumulative and Average Regret Analysis**: Measures algorithm performance over time.
- **Collaborative Training**: Shares knowledge across clients to optimize decisions.
- **Parameter Tuning**: Easily adjust parameters like lambda, delta, and noise variance.

---

## How to Run

1. **Clone the Repository**:  
   ```bash
   git clone https://github.com/username/hierarchical-bayesian-bandits.git
   cd hierarchical-bayesian-bandits

   Run the command `python linear_ucb.py` to execute the Linear UCB model.

   Run the command `python linear_slider.ipynb` to execute the Linear UCB model with a slider for parameters.

   Run the command `python model2.ipynb` to execute the Multi-client model.

   

## Acknowledgments

Special thanks to the FNM REU Site and UCLA Information Theory and Systems Lab for guiding this project, including Professor Suhas Diggavi and Bruce Huang, for their invaluable insights and support.
