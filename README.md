# hmr_sim

[![Python](https://img.shields.io/badge/Python-3.7%20or%20later-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://gymnasium.farama.org/assets/images/logo-dark.svg)](https://github.com/Farama-Foundation/Gymnasium)




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

### Homo
- [ ] Raise error if formation is initialized within any obstacle
- [ ] Integrate path tracking

### Hetro
- [x] All config variables related to hetro swarm increase an order... num agents becomes [3,4] meaning 3 agents of type 0 and 4 of type 1.
- [ ] Formation init: Allow formation for some, and manual initialization for others. Change both init_pos and init_form logic, and use a dictionary for it! {0: [POSITIONS], 1: [cIRCLE,]} ...
- [ ] Create UML of everything!!!
- [ ] Add options for diff controllers!
- [x] Make connectivity controller distributed!
- [ ] Fix path speed bug

- [ ] Add local obstacle avoidance


## References

past path