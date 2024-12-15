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

- [ ] Sometimes something goes horibly wrong when network breaks

nton-superior@antonsuperior:~/hmr_sim/hmr_sim/tests$ python3 hetro/test_hetro.py thesis1.yaml
Loading configuration from: /home/anton-superior/hmr_sim/hmr_sim/tests/hetro/thesis1.yaml
Initializing environment: Hetro-v0
Using initialization formation: {'shape': 'lattice', 'origin': [0.0, 0.0], 'major_radius': 3.0}
Number of agents is 12, which is <= 12
base: [-3.5  0. ]
Adding new agent at position: [-3.11058837  1.70612385]
Type: 2, ID: 2, batetery type: 1.0
Agent 2 removed due to low battery.
Number of agents is 12, which is <= 12
base: [-3.5  0. ]
Adding new agent at position: [-4.59110715  1.36820509]
Type: 2, ID: 3, batetery type: 1.0
Agent 3 removed due to low battery.
Number of agents is 12, which is <= 12
base: [-3.5  0. ]
Adding new agent at position: [-3.88941163 -1.70612385]
Type: 2, ID: 4, batetery type: 1.0
Agent 4 removed due to low battery.
Number of agents is 12, which is <= 12
base: [-3.5  0. ]
Adding new agent at position: [-5.07669552  0.75929654]
Type: 2, ID: 5, batetery type: 1.0
Agent 5 removed due to low battery.
Number of agents is 12, which is <= 12
base: [-3.5  0. ]
Adding new agent at position: None
Type: 2, ID: 6, batetery type: 1.0
Traceback (most recent call last):
  File "/home/anton-superior/hmr_sim/hmr_sim/tests/hetro/test_hetro.py", line 73, in <module>
    main()
  File "/home/anton-superior/hmr_sim/hmr_sim/tests/hetro/test_hetro.py", line 70, in main
    run(config)
  File "/home/anton-superior/hmr_sim/hmr_sim/tests/hetro/test_hetro.py", line 40, in run
    env.render()
  File "/home/anton-superior/.local/lib/python3.10/site-packages/gymnasium/core.py", line 332, in render
    return self.env.render()
  File "/home/anton-superior/.local/lib/python3.10/site-packages/gymnasium/wrappers/common.py", line 409, in render
    return super().render()
  File "/home/anton-superior/.local/lib/python3.10/site-packages/gymnasium/core.py", line 332, in render
    return self.env.render()
  File "/home/anton-superior/.local/lib/python3.10/site-packages/gymnasium/wrappers/common.py", line 303, in render
    return self.env.render()
  File "/home/anton-superior/hmr_sim/hmr_sim/envs/hetro/hetro_v0.py", line 56, in render
    self.render_func.render()
  File "/home/anton-superior/hmr_sim/hmr_sim/utils/vis.py", line 184, in render
    self.update_adjacency_lines()
  File "/home/anton-superior/hmr_sim/hmr_sim/utils/vis.py", line 91, in update_adjacency_lines
    adjacency_matrix = self.swarm.compute_adjacency_matrix()
  File "/home/anton-superior/hmr_sim/hmr_sim/utils/swarm.py", line 139, in compute_adjacency_matrix
    distance = euclidean(positions[i], positions[j])
  File "/home/anton-superior/.local/lib/python3.10/site-packages/scipy/spatial/distance.py", line 520, in euclidean
    return minkowski(u, v, p=2, w=w)
  File "/home/anton-superior/.local/lib/python3.10/site-packages/scipy/spatial/distance.py", line 480, in minkowski
    dist = norm(u_v, ord=p)
  File "/home/anton-superior/.local/lib/python3.10/site-packages/scipy/linalg/_misc.py", line 146, in norm
    a = np.asarray_chkfinite(a)
  File "/home/anton-superior/.local/lib/python3.10/site-packages/numpy/lib/function_base.py", line 628, in asarray_chkfinite
    raise ValueError(
ValueError: array must not contain infs or NaNs