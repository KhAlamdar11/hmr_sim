# hmr_sim

[![Python](https://img.shields.io/badge/Python-3.7%20or%20later-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://gymnasium.farama.org/assets/images/logo-dark.svg)](https://github.com/Farama-Foundation/Gymnasium)




This package provides a hmr_sim simulator built with Gymnasium for testing and developing navigation algorithms. Designed as an educational tool, it allows users to quickly implement and evaluate various navigation strategies, making it suitable for introductory robotics courses. The simulator is lightweight and computationally efficient.


<!-- <div align="center">
    <img src="/media/classic.gif" alt="testing" height="240">
</div>

<div align="center">
    <img src="/media/near.gif" alt="testing" height="150"><img src="/media/base.gif" alt="testing" height="150">
</div> -->

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

```
hmr_sim/
├── hmr_sim/                    
│   ├── controllers/               # Directory for implementation of different controllers 
│   ├── envs/                      # Environment definitions for Gymnasium
│   │   └── basic_env/             
│   │       └── basic_env0.py      # Environment files
│   ├── maps/                      # Directory for map data used in environments
│   │   └── mapx/                  # Example map folder (e.g., map0, map1)
│   │       ├── data.yaml          # YAML configuration file for map properties (e.g., origin, resolution)
│   │       └── map.bmp            # Bitmap image representing the occupancy grid
│   └── utils/                     # Utility functions for visualization and helpers
├── tests/                         # Directory for test scripts and configurations
│   ├── cfg/                       # Configuration files for test runs
│   ├── utils/                     # Test-specific utility functions
│   └── test_xx.py                # Main test script for validating the environment and controllers
```

## Availabe Environments

This section lists available environments, their usage and test scripts/

Navigate to the `/tests` dir.

```bash
cd hmr_sim/tests
```


| **Environment**       | **Test Run Command**                                    | **Animation (GIF)**                     |
|------------------|---------------------------------------------|-------------------------------------------|
| `Nav-v0` <br> An environment for omnidirectional agents and testing with deliberate (path planners) and reactive algorithms. <br> Environment directory: `envs/omni/nav.py` | `python3 omni_nav/test_rrt.py rrt.cfg` <br> `python3 omni_nav/test_bug0.py bug0.cfg` | ![OmniNav Animation](path_to_omni_nav.gif) |
| `Explore-v0` <br> An environment for omnidirectional robots and testing with exploration algorithms. It immitates an agent with a lidar sensor to map the environment. As an example, frontier exploration algorithm is implemented in the environment class. <br> Environment directory: `envs/omni/explore.py` | `python3 omni_explore/test_explore.py explore.cfg` | ![OmniNav Animation](path_to_omni_nav.gif) |
| `NavDiffDrive-v0` <br> An environment for differential drive robots and testing with navigation algorithms. The kinematics of the agent are handled by the `follow_path()` function of the environment. <br> Environment directory: `envs/diff_drive/nav_dd.py` | `python3 diff_drive_nav/test_rrt_dd.py rrt_dd.cfg` | ![OmniNav Animation](path_to_omni_nav.gif) |

## Usage

The base environments are defined in the `hmr_sim/hmr_sim/envs` directory. These environments are responsible for setting up the simulation, maps, robots, sensory methods, and other components. Please refer to the **Environments** section for a detailed description of the available environments and maps.

The goal of this project is to facilitate quick testing of navigation controllers, which are located in the `hmr_sim/hmr_sim/controllers` directory. These controllers can be broadly categorized into two main types: **deliberative** and **reactive**.

For **deliberative control**, the map is provided as an occupancy grid. Refer to the RRT example in `.../controllers/deliberative/rrt.py` for usage and utility functions.

For **reactive control**, a local circular field of view (FoV) is created based on a configuration file parameter. This enables the detection of obstacles as they enter the agent's FoV. The FoV can be represented in two ways: (1) as a local occupancy grid-like structure highlighting free and unknown areas, or (2) as a vector of angles denoting the presence of obstacles at certain angles relative to the agent, similar to raw readings from a lidar sensor.
An example of (1) can be seen in the Bug 0 algorithm implementation in `.../controllers/reactive/bug0.py`, and an example of (2) can be found in the local potential fields implementation in `.../controllers/reactive/pf.py`.

## How to test algorithms

### Omni & Differential Drive Environments

To test your custom algorithms, put your controllers under `hmr_sim/controllers` directory. See the current functions for usage. Implement you own class in a seperate python file. Import your class in `hmr_sim/envs/omni/base.py`. Then add the option for your custom controller in the `load_config` function as is done for rrt and bug0.



## Available Environments

### Connectivity Battery v0


```bash
cd hmr_sim/tests/
python3 test_bv1.py cfg/custom.cfg
```

If the stubborn agents are to be controlled manually using the keyboard instead of following a pre-defined trajectory, the same script can be run with a different configuration file:

```bash
cd hmr_sim/tests/
python3 test_connectivitybattery_v0.py cfg/cfg_test_keyboard.cfg
```

The following keys can be used to control the two stubborn agents:

<div align="center">
    <img src="/media/keyboard.png" alt="testing" height="130">
</div>

**Additional Information**

If `Ctrl+C` is pressed where the script is running, a simulation video will be saved along with some data files. These data files include numpy files for tracking:

- Fiedler value
- Distance measurements
- Number of agents 

Refer to the configuration files for more details on the simulation parameters and settings.

### Connectivity Battery v1

env-name: ConnectivityBattery-v1

Decoupled controller with same features as above. Here, the controller only outsputs connectivity forces, and the seperation force is computed in the step function. This is done to allow learning only part of the control law instead of all end to end.

To run the test script:

```bash
cd hmr_sim/tests/
python3 test_connectivitybattery_v1.py cfg/cfg_test_random.cfg
```

### Connectivity3D-v0

env-name: Connectivity3D-v0

Extension of the connectivity controller to 3D.

To run the test script:

```bash
cd hmr_sim/tests/
python3 test_connectivity3d_v0.py cfg/cfg_test_3d.cfg
```

⚠️ Some funcionalities, such as spawning of new agents are not well defined in 3D space.

⚠️ Agents in 3D space are more unstable and gains require to be tuned extensively. 

## Custom Usage

To use the environment in your own script:

~~~~
import gym  
import hmr_sim
env = gym.make("Connectivity-v0") 
~~~~

`env.reset()` and `env.step()` can be used to interface with the environment, just like with other OpenAI Gym environments. Additionally, the `env.controller()` function computes the connectivity controller output for the agents.

For more examples, consult the test scripts under the ```hmr_sim/tests/``` directory.

## Acknowledgements

This codebase is structured similarly to and inspired by the work of katetolstaya (https://github.com/katetolstaya/gym-flock).

## TODOs
- [ ] Update all variables to be object-agnostic (e.g., use "agents" instead of "UAV/robot/base/node") for consistency.
- [ ] Upgrade the lattice generation algorithm to fix the recursion bug (see [Issue](https://github.com/KhAlamdar11/gym-connect/issues/1)).
- [ ] Create a new environment that allows dynamic allotment of stubborn agents, instead of presetting them to 2.
- [ ] Upgrade to support trajectories for multiple stubborn agents simultaneously (currently, only one agent can move in a circle).
- [ ] Upgrade 3D environment to include agent addition strategies in 3D.
- [ ] Upgrade the entire codebase to use Gymnasium.



## References

past path