# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-agent robotic simulator for UAV swarm missions with connectivity maintenance, built on Gymnasium (OpenAI Gym). Used for testing decentralized connectivity control algorithms where UAVs maintain network connectivity while mobile robots perform exploration tasks.

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Run simulation with a configuration
cd hmr_sim/tests/
python3 test_hetro.py <config_file.yaml>

# Example configurations available in hmr_sim/tests/:
# - hetro.yaml          (heterogeneous multi-agent demo)
# - explore_x.yaml      (frontier exploration with connectivity)
# - obstacle.yaml       (obstacle avoidance)
# - add_agent_base.yaml (dynamic agent addition)
```

## Architecture

### Layered Design

1. **Gymnasium Environment** (`envs/`): `Hetro-v0` environment with state space [x, y, vx, vy] per agent
2. **Swarm Management** (`utils/swarm.py`): Manages heterogeneous agent groups, neighbor updates, adjacency matrix
3. **Agent Class** (`utils/agent.py`): Individual agents with controller type, battery, path tracking
4. **Controllers** (`utils/connectivity_controller.py`, `utils/rrt.py`, `utils/frontier_explore.py`): Interchangeable control strategies
5. **Visualization** (`utils/vis.py`): Real-time matplotlib rendering

### Key Patterns

- **Configuration-driven**: Mission scenarios defined entirely in YAML files
- **Callback handlers**: Map functions (collision check, frontier detection) passed as callbacks to agents
- **Strategy pattern**: Controller types (`connectivity_controller`, `go_to_goal`, `explore`, `path_tracker`, `do_not_move`) are swappable via config

### Agent Types

- Type 0: Base station (stationary)
- Type 1: UAVs (connectivity maintainers)
- Type 2: Mobile robots (task performers)

## Configuration Structure

YAML configs define complete mission scenarios:

```yaml
env: Hetro-v0
map_name: mapx           # Map from hmr_sim/maps/
vis_radius: 5.0          # Communication radius
dt: 0.1                  # Timestep

agent_config:
  0:                     # Agent type
    num_agents: 1
    controller_type: do_not_move
    init_position: [[0.0, 0.0]]
  1:
    num_agents: 14
    controller_type: connectivity_controller
    init_formation:
      shape: Circle      # Circle, Elipse, Square, Lattice
      origin: [0.0, 0.0]
      radius: 1.0

controller_params:       # Connectivity controller tuning
  delta: 0.2
  gainConnectivity: 1.0
  gainRepel: 0.08
```

## Map Format

Each map in `hmr_sim/maps/` contains:
- `map.bmp`: Binary occupancy grid (grayscale image)
- `data.yaml`: Origin coordinates and resolution

## Core Files

- `hmr_sim/envs/base.py`: Core environment with occupancy grid, collision detection
- `hmr_sim/utils/swarm.py`: Swarm initialization, neighbor computation, state updates
- `hmr_sim/utils/agent.py`: Agent state, controller dispatch, path following
- `hmr_sim/utils/connectivity_controller.py`: Fiedler-based decentralized connectivity control
- `hmr_sim/utils/add_agents.py`: Dynamic agent addition logic
- `hmr_sim/tests/test_hetro.py`: Main test/demo runner

## Known TODOs (from README)

- Bug: Different formation shapes cause varying agent speeds
- Need to raise error if formation initialized within obstacle
- Config validation function needed
- Features to port: Fiedler-based agent addition, keyboard control
