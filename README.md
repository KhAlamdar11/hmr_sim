# hmr_sim

[![Python](https://img.shields.io/badge/Python-3.7%20or%20later-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://gymnasium.farama.org/assets/images/logo-dark.svg)](https://github.com/Farama-Foundation/Gymnasium)


sending frontier detector function to swarm and then to the robot itself! 


This package provides a hmr_sim simulator built with Gymnasium for testing and developing multi-agent system algorithms. 


## Installation

1. **Dependencies**: 

    Install dependencies via the requirements file.

    ```bash
    cd hmr_sim
    pip install -r requirements.txt
    ```

2. **Setup the package:**

    ```bash
    pip install -e . 
    ```

## Directory Structure

This package is organized as follows:



## Availabe Environments

This section lists available environments, their usage and test scripts/

### Homogenous Environment

Name: Homo-v0

Navigate to the `/tests` dir.

```bash
cd hmr_sim/tests
```


## Usage

The base environments are defined in the `hmr_sim/hmr_sim/envs` directory. These environments are responsible for setting up the simulation, maps, robots, sensory methods, and other components. Please refer to the **Environments** section for a detailed description of the available environments and maps.


## How to test algorithms

## TODOs

Road to gym:
- [x] Update battery colors (fix vis function)
- [x] Add method to remove agents
- [x] Add add agent pipeline, reuse gym functions
- [x] Add add agent near
- [ ] Add lattice based initialization
- [ ] Recheck if rest of the scenerios work well
- [x] Add dotted circles for agents below critical battery and agents with full battery!
- [ ] Rewrite agent additions, now with paths shown! Add method to show path of specific agents only (new agent)
 

- [x] All config variables related to hetro swarm increase an order... num agents becomes [3,4] meaning 3 agents of type 0 and 4 of type 1.
- [x] Formation init: Allow formation for some, and manual initialization for others. Change both init_pos and init_form logic, and use a dictionary for it! {0: [POSITIONS], 1: [cIRCLE,]} ...
- [x] Add options for diff controllers!
- [x] Make connectivity controller distributed!
- [x] Add local obstacle avoidance

Anon:
- [ ] Create UML of everything!!!
- [ ] Fix path speed bug

Error handling:
- [ ] Raise error if formation is initialized within any obstacle
- [ ] Create a function to check if all config variable combos make sense!


## Bugs

- [ ] Error: Possibly when agents enter the obstacle.
yellow
    adjusted_position = self.obstacle_avoidance(proposed_position=proposed_position, 
  File "/home/anton-superior/hmr_sim/hmr_sim/utils/agent.py", line 217, in obstacle_avoidance
    if is_free_path_fn(current_position, check_point):
  File "/home/anton-superior/hmr_sim/hmr_sim/envs/hetro/base.py", line 74, in is_line_of_sight_free
    end = position_to_grid(position2)
  File "/home/anton-superior/hmr_sim/hmr_sim/envs/hetro/base.py", line 69, in position_to_grid
    grid_x = int((position[0] - self.origin['x']) / self.resolution)
ValueError: cannot convert float NaN to integer

- [ ] RRT KNOWS THE WHOLE MAP!

- [ ] Sometimes old maps remain!