import numpy as np

def init_homo_formation(formation, num_agents=5):
    """
    Initialize agent positions based on the given formation.

    Args:
        formation (list): Formation type and parameters:
                         - ["Circle", [x, y], radius]
                         - ["Elipse", [x, y], major_radius, minor_radius (optional)]
                         - ["Square", [x, y], side_length]
        num_agents (int): Number of agents.

    Returns:
        np.ndarray: Array of positions as [[x, y], [x, y], ...].
    """
    formation_type = formation[0].lower()
    origin = np.array(formation[1])  # Extract the origin
    
    if formation_type == "circle":
        radius = formation[2]
        angles = np.linspace(0, 2 * np.pi, num_agents, endpoint=False)
        positions = np.array([[radius * np.cos(angle), radius * np.sin(angle)] for angle in angles])
        return positions + origin  # Offset positions by origin
    
    elif formation_type == "elipse":
        major_radius = formation[2]
        minor_radius = formation[3] if len(formation) > 3 else major_radius / 2  # Default minor radius
        angles = np.linspace(0, 2 * np.pi, num_agents, endpoint=False)
        positions = np.array([[major_radius * np.cos(angle), minor_radius * np.sin(angle)] for angle in angles])
        return positions + origin  # Offset positions by origin
    
    elif formation_type == "square":
        side_length = formation[2]
        half_side = side_length / 2
        
        # Distribute agents along the four sides of the square
        per_side = max(1, num_agents // 4)
        remainder = num_agents % 4
        
        # Generate points along each side
        top = np.linspace(-half_side, half_side, per_side + (1 if remainder > 0 else 0), endpoint=False)
        right = np.linspace(half_side, -half_side, per_side + (1 if remainder > 1 else 0), endpoint=False)
        bottom = np.linspace(half_side, -half_side, per_side + (1 if remainder > 2 else 0), endpoint=False)
        left = np.linspace(-half_side, half_side, per_side, endpoint=True)
        
        # Combine the coordinates for each edge
        positions = np.concatenate([
            np.column_stack((top, np.full_like(top, half_side))),  # Top edge
            np.column_stack((np.full_like(right, half_side), right)),  # Right edge
            np.column_stack((bottom, np.full_like(bottom, -half_side))),  # Bottom edge
            np.column_stack((np.full_like(left, -half_side), left))  # Left edge
        ])
        
        # Trim to the number of agents
        return positions[:num_agents] + origin  # Offset positions by origin
    
    else:
        raise ValueError(f"Unsupported formation type: {formation_type}")
