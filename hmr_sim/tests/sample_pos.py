import numpy as np


def sample_points_in_ellipse(n, major_radius, origin=(0, 0), minor_radius=None, min_distance=0.1):
    """
    Generate n points uniformly distributed within an ellipse while ensuring minimum distance between points.

    Parameters:
        n (int): Number of points to generate.
        major_radius (float): The major radius of the ellipse.
        origin (tuple): The center of the ellipse (x, y).
        minor_radius (float, optional): The minor radius of the ellipse. 
                                        Defaults to major_radius (circle).
        min_distance (float): Minimum distance between points.

    Returns:
        numpy.ndarray: An (n, 2) array of points within the ellipse.
    """
    if minor_radius is None:
        minor_radius = major_radius

    points = []

    while len(points) < n:
        # Generate a single random point within the ellipse
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.sqrt(np.random.uniform(0, 1))
        x = radius * major_radius * np.cos(angle)
        y = radius * minor_radius * np.sin(angle)

        # Translate point by the origin
        x += origin[0]
        y += origin[1]

        # Check minimum distance constraint
        if all(np.linalg.norm(np.array([x, y]) - np.array(p)) >= min_distance for p in points):
            points.append([x, y])
        # print(len(points))

    return points


# Example usage
if __name__ == "__main__":
    n = 50  # Number of points
    major_radius = 3.0
    minor_radius = 3.0
    origin = (0, 0)
    min_distance = 0.6  # Minimum distance between points

    points = sample_points_in_ellipse(n, major_radius, origin, minor_radius, min_distance)
    # print(points)

    # Save the points to a file if needed
    # np.save("ellipse_points.npy", points)

    # Print or visualize the points
    print(points)
