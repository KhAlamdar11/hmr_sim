# hmr_sim

[![Python](https://img.shields.io/badge/Python-3.7%20or%20later-blue.svg)](https://www.python.org/downloads/)

> ⚠ **Documentation is currently in progress.** In the meantime, please refer to the [Usage Examples](#usage-examples) section.

This package provides **`hmr_sim`**, a simulator with **Gymnasium** environments, for testing and developing **multi-agent system algorithms**. It is designed to facilitate the rapid creation and evaluation of different mission scenarios.

This project is an extension of the [gym_connect](https://github.com/KhAlamdar11/gym-connect) package, originally developed for [<a href="#ref1">1</a>]. It expands the framework to support multiple mission scenarios and introduces a more modular code structure. Some features have not yet been ported — see checklist <a href="#yet-to-port">here</a> for details.


## Installation

```bash
# Clone the package
git clone https://github.com/KhAlamdar11/hmr_sim

# Navigate to the root of the project
cd hmr_sim

# Install dependencies via your python package manager
pip install -r requirements.txt

# Install the package
pip install -e . 
```

## Availabe Environments

For simplicity, a single Gymnasium environment is created to handle all multiple mission scenerios and parameters using a config file.
This environment is named ```Hetero-V0``` and inherits from the  ```Base``` environment. The ```Base``` environment handles all the map related behaviours and is created seperately from ```Hetero-V0``` to allow extensions to multiple environments if needed in the future.

<a id="usage-examples"></a>
## Usage Examples

The table below illustrates examples of creating different mission scenarios. The provided YAML files can be used as samples to experiment with various mission setups and parameters.

First navigate to the test folder: ```cd hmr_sim/tests/```

| Mission Scenario                                                                                                                                                                                                | Command                                                                                              | Visualization                                                                                                                                                            |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Connectivity control with addition of new agents when the connectivity is low.                                                                                                                                  | ```python3 test_hetro.py add_agent_base.yaml``` <br> ```python3 test_hetro.py add_agent_near.yaml``` | <img src="hmr_sim/media/add_agent_base.gif" alt="Battery Management" width="250"> <br> <img src="hmr_sim/media/add_agent_near.gif" alt="Battery Management" width="250"> |
| An example of a heterogeneous system with a stationary base station (red), 4 mobile robots (yellow) that follow an RRT-defined path to goals, and UAVs (blue) that maintain connectivity of the entire network. | ```python3 test_hetro.py hetro.yaml```                                                               | <img src="hmr_sim/media/hetro.gif" alt="Battery Management" width="250">                                                                                                 |
| An example where the UAVs (blue) perform obstacle avoidance.                                                                                                                                                    | ```python3 test_hetro.py obstacle.yaml```                                                            | <img src="hmr_sim/media/obstacle.gif" alt="Battery Management" width="250">                                                                                              |
| A mission with 2 robots (yellow) exploring an unknown environment using frontier exploration, while UAVs (blue) maintain connectivity between these robots and the base station (red).                          | ```python3 test_hetro.py explore_x.yaml```                                                           | <img src="hmr_sim/media/multi_agent_exploration.gif" alt="Battery Management" width="250">                                                                               |


## TODOs

**General**

- [ ] **Bug**: For different path types (circle, ellipse, square), agents move at varying speeds even when the speed variable is the same. Redo perimeter sampling logic.
- [ ] **Error Handling**: Raise an error if a formation is initialized within an obstacle.
- [ ] **Validation**: Create a function to validate configuration combinations.   For example: If there are two agents and the `go_to_goal` controller is chosen, ensure that two goal positions are specified.

<a id="yet-to-port"></a>
**Yet to port from gym_connect**

- [ ] **Fiedler based agent addition**: Add the option to add agents based on min fiedler value (minor)
- [ ] **Keyboard control**: Allow a controller option for control by keyboard.

## References

<a id="ref1"></a>
[1]: K.G. Alamdar, “Connectivity Maintainence for ad-hoc UAV Networks for Multi-robot Missions,'' University of Zagreb, 2024, June.