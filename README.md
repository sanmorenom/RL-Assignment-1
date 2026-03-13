# Assignment 1 - Tabular Reinforcement Learning

## Group Members
- Jasper Scheel (s4888456)
- Muhamad Iqbal Arsa (s4859049)
- Santiago Moreno Mercado (s4614151)

## Work Division
All team members collaborated closely and contributed equally to every part of this assignment. We actively discussed the concepts and experiments together, combined our code implementations, and shared equal responsibility in writing, editing, and revising the final scientific report.

## Setup & Installation
This code is intended to be run on a Linux machine (DSLab/computer lab environment) using Python 3. 
To install the required dependencies, open your terminal, navigate to this directory, and run the following command:

    pip install -r requirements.txt

## How to Run the Experiments
Below are the commands to easily rerun our experiments and reproduce the learning curves from the report.

### 1. Dynamic Programming (Part 1.4)
To execute the Q-value iteration algorithm and view the Q-value estimates progression, run:
    
    python DynamicProgramming.py

### 2. Model-Free Reinforcement Learning (Parts 1.5, 1.6, and 1.7)
All model-free RL experiments (Exploration, Backup: On-policy vs Off-policy, and Backup: Target Depth) are integrated into a single execution file. To run them all and generate the respective learning curves, simply run:
    
    python Experiment.py

*(Note: Please wait a few moments as the program will execute the training loops for all three parts and plot the graphs sequentially).*
