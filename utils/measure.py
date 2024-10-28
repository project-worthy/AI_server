import math

from itertools import combinations, permutations;

import numpy as np

def calculate_distance(point1, point2):
    # 두 점 사이의 유클리드 거리 계산
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

def degrees_to_radians(degrees):
    """
    Converts an angle from degrees to radians.

    Parameters:
    - degrees: The angle in degrees.

    Returns:
    - The angle in radians.
    """
    return degrees * (math.pi / 180)

def radians_to_degrees(radians):
    """
    Converts an angle from radians to degrees.

    Parameters:
    - radians: The angle in radians.

    Returns:
    - The angle in degrees.
    """
    return radians * (180 / math.pi)

def rotate_point_2d(point, angle):
    """
    Rotates a point in 2D space around the origin by a given angle.

    Parameters:
    - point: A tuple (x, y) representing the coordinates of the point.
    - angle: The rotation angle in radians.

    Returns:
    - A tuple representing the new coordinates of the point after rotation.
    """
    
    # Convert angle to radians if needed
    theta = angle
    
    # Define the 2D rotation matrix
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    # Convert the point to a numpy array and perform the rotation
    rotated_point = rotation_matrix.dot(np.array(point))
    
    return tuple(rotated_point)

def distance_3d(point1, point2):
    """
    Calculates the distance between two points in 3D space.

    Parameters:
    - point1: The first point as a tuple (x1, y1, z1).
    - point2: The second point as a tuple (x2, y2, z2).

    Returns:
    - The distance between point1 and point2.
    """
    
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def rectangle_line_segments(width, height):
    # 사각형의 꼭짓점 정의 (원점을 기준으로 한 사각형)
    vertices = [(0, 0), (width, 0), (width, height), (0, height)]
    verticesLabel = [0,1,2,3]
    
    # 모든 꼭짓점 쌍의 조합을 계산 (2개씩)
    line_combinations = list(permutations(verticesLabel, 2))
    
    # 각 선분의 길이를 계산하여 저장
    line_lengths = [(combo[0], combo[1], calculate_distance(vertices[combo[0]], vertices[combo[1]])) for combo in line_combinations]
    
    return line_lengths

def get_square_distance_map(width,height):
  lines = rectangle_line_segments(width,height)
  print(lines)
  distanceMap = dict()
  for line in lines:
    if(not distanceMap.get(line[0])):
       distanceMap[line[0]] = {}
    distanceMap[line[0]][line[1]] = line[2]
    # pass

  print(distanceMap)
  return distanceMap

point1 = (316,0,0)
point2 = (316,290,0)
point3 = (0,290,0)

def triangulate_3d(p1, p2, p3, d1, d2, d3):
    """
    Triangulates the position of a point in 3D space given three known points and distances to each.
    
    Parameters:
    - p1, p2, p3: Coordinates of the known points as tuples (x, y, z)
    - d1, d2, d3: Distances from the target point to p1, p2, and p3
    
    Returns:
    - A tuple (x, y, z) representing the triangulated coordinates of the target point.
    """
    
    # Convert points to numpy arrays
    P1, P2, P3 = np.array(p1), np.array(p2), np.array(p3)
    
    # Calculate ex, ey, and ez (unit vectors)
    ex = (P2 - P1) / np.linalg.norm(P2 - P1)
    i = np.dot(ex, P3 - P1)
    ey = (P3 - P1 - i * ex) / np.linalg.norm(P3 - P1 - i * ex)
    ez = np.cross(ex, ey)
    
    # Distances between known points
    d = np.linalg.norm(P2 - P1)
    j = np.dot(ey, P3 - P1)
    
    # Calculate coordinates
    x = (d1**2 - d2**2 + d**2) / (2 * d)
    y = (d1**2 - d3**2 + i**2 + j**2) / (2 * j) - (i / j) * x
    z = np.sqrt(d1**2 - x**2 - y**2)
    
    # Calculate target position
    target = P1 + x * ex + y * ey + z * ez
    
    return tuple(target)


def triangulate_3d_with_range(p1, p2, p3, d1, d2, d3, error_rate=0.0, num_samples=100):
    """
    Triangulates the range of positions of a point in 3D space given three known points and distances to each.
    
    Parameters:
    - p1, p2, p3: Coordinates of the known points as tuples (x, y, z)
    - d1, d2, d3: Distances from the target point to p1, p2, and p3
    - error_rate: Fractional error rate to apply to each distance (e.g., 0.05 for ±5% error)
    - num_samples: Number of random samples to generate within the error range
    
    Returns:
    - A dictionary with 'min_position' and 'max_position' tuples representing the range of possible target coordinates.
    """
    
    # Convert points to numpy arrays
    P1, P2, P3 = np.array(p1), np.array(p2), np.array(p3)
    # multiple 100 all distances
    d1 *= 100
    d2 *= 100
    d3 *= 100
    
    # Store all sampled target positions
    target_positions = []

    for _ in range(num_samples):
        # Apply random error to each distance
        d1_sample = d1 * (1 + np.random.uniform(-error_rate, error_rate))
        d2_sample = d2 * (1 + np.random.uniform(-error_rate, error_rate))
        d3_sample = d3 * (1 + np.random.uniform(-error_rate, error_rate))
        
        # Calculate ex, ey, and ez (unit vectors)
        ex = (P2 - P1) / np.linalg.norm(P2 - P1)
        i = np.dot(ex, P3 - P1)
        ey = (P3 - P1 - i * ex) / np.linalg.norm(P3 - P1 - i * ex)
        ez = np.cross(ex, ey)
        
        # Distances between known points
        d = np.linalg.norm(P2 - P1)
        j = np.dot(ey, P3 - P1)
        
        # Calculate coordinates
        x = (d1_sample**2 - d2_sample**2 + d**2) / (2 * d)
        y = (d1_sample**2 - d3_sample**2 + i**2 + j**2) / (2 * j) - (i / j) * x
        z = np.sqrt(d1_sample**2 - x**2 - y**2)
        
        # Calculate target position and append to list
        target = P1 + x * ex + y * ey + z * ez
        target_positions.append(target)
    
    # Convert to array for easier manipulation
    target_positions = np.array(target_positions)
    
    # Determine min and max bounds for each axis
    min_position = target_positions.min(axis=0)
    max_position = target_positions.max(axis=0)
    
    return {
        "min_position": tuple(min_position),
        "max_position": tuple(max_position)
    }


def trilaterate_3d_with_error(p1, p2, p3, d1, d2, d3, error_rate=0.05, num_samples=100):
    """
    Determines the range of positions of a point in 3D space based on distances to three known points, with error consideration.

    Parameters:
    - p1, p2, p3: Known points in 3D space as tuples (x, y, z)
    - d1, d2, d3: Distances from the target point to p1, p2, and p3
    - error_rate: Fractional error rate for distance adjustments (e.g., 0.05 for ±5% error)
    - num_samples: Number of samples to calculate with the error rate

    Returns:
    - A dictionary with 'estimated_position', 'min_position', and 'max_position' representing the possible range of the target coordinates.
    """
    
    # Convert points to numpy arrays
    P1, P2, P3 = np.array(p1), np.array(p2), np.array(p3)
    
    # Store all sampled target positions
    target_positions = []

    for _ in range(num_samples):
        # Apply random error to each distance
        d1_sample = d1 * (1 + np.random.uniform(-error_rate, error_rate))
        d2_sample = d2 * (1 + np.random.uniform(-error_rate, error_rate))
        d3_sample = d3 * (1 + np.random.uniform(-error_rate, error_rate))
        
        # Calculate ex, ey, and ez (unit vectors)
        ex = (P2 - P1) / np.linalg.norm(P2 - P1)
        i = np.dot(ex, P3 - P1)
        ey = (P3 - P1 - i * ex) / np.linalg.norm(P3 - P1 - i * ex)
        ez = np.cross(ex, ey)
        
        # Distances between known points
        d = np.linalg.norm(P2 - P1)
        j = np.dot(ey, P3 - P1)
        
        # Calculate coordinates
        x = (d1_sample**2 - d2_sample**2 + d**2) / (2 * d)
        y = (d1_sample**2 - d3_sample**2 + i**2 + j**2) / (2 * j) - (i / j) * x
        z = np.sqrt(max(0, d1_sample**2 - x**2 - y**2))  # Ensure non-negative under sqrt
        
        # Calculate target position and append to list
        target = P1 + x * ex + y * ey + z * ez
        target_positions.append(target)
    
    # Convert positions to numpy array
    target_positions = np.array(target_positions)
    
    # Calculate mean, min, and max positions for each axis
    estimated_position = target_positions.mean(axis=0)
    min_position = target_positions.min(axis=0)
    max_position = target_positions.max(axis=0)
    
    return {
        "estimated_position": tuple(estimated_position),
        "min_position": tuple(min_position),
        "max_position": tuple(max_position)
    }

import numpy as np

def trilateration_3d(p1,p2,p3,r1,r2,r3):
    face1 = intersection_spheres(p1,r1,p2,r2)
    face2 = intersection_spheres(p2,r2,p3,r3)
    print(face1,face2)
    a = line_of_intersection(face1,face2)
    # print(a)
    return a


def intersection_spheres(center1, radius1, center2, radius2):
    """
    Finds the plane of intersection of two spheres in 3D space.

    Parameters:
    - center1: Center of the first sphere as a tuple (x1, y1, z1).
    - radius1: Radius of the first sphere.
    - center2: Center of the second sphere as a tuple (x2, y2, z2).
    - radius2: Radius of the second sphere.

    Returns:
    - A list [a, b, c, d] representing the plane of intersection in the form ax + by + cz = d,
      or None if there is no intersection.
    """
    radius1 *= 100
    radius2 *= 100
    # Convert centers to numpy arrays
    C1 = np.array(center1)
    C2 = np.array(center2)
    
    # Calculate the normal vector of the plane of intersection
    normal_vector = C2 - C1
    d = np.linalg.norm(normal_vector)  # Distance between sphere centers
    
    # Check if spheres intersect
    if d > radius1 + radius2 or d < abs(radius1 - radius2):
        return None  # Spheres do not intersect
    
    # Calculate the distance from the first sphere's center to the plane
    p = (radius1**2 - radius2**2 + d**2) / (2 * d)
    
    # Calculate the point on the plane
    point_on_plane = C1 + (p / d) * normal_vector
    
    # Extract the components of the normal vector and the point on the plane
    a, b, c = normal_vector
    x0, y0, z0 = point_on_plane
    
    # Calculate the value of d in the plane equation ax + by + cz = d
    d_plane = a * x0 + b * y0 + c * z0
    
    return [a, b, c, d_plane]

import numpy as np

def line_of_intersection(plane1, plane2):
    """
    Finds the line of intersection between two planes in 3D space.
    
    Parameters:
    - plane1: Coefficients of the first plane as a tuple (a1, b1, c1, d1) representing the plane a1*x + b1*y + c1*z = d1.
    - plane2: Coefficients of the second plane as a tuple (a2, b2, c2, d2) representing the plane a2*x + b2*y + c2*z = d2.
    
    Returns:
    - A dictionary with:
        - 'point_on_line': A point on the intersection line as a tuple (x, y, z)
        - 'direction_vector': The direction vector of the intersection line as a tuple (dx, dy, dz)
      Returns None if the planes are parallel (no intersection).
    """
    
    # Unpack plane coefficients
    a1, b1, c1, d1 = plane1
    a2, b2, c2, d2 = plane2
    
    # Convert the normal vectors to numpy arrays
    N1 = np.array([a1, b1, c1])
    N2 = np.array([a2, b2, c2])
    
    # Calculate the direction vector of the line of intersection (cross product of normals)
    direction_vector = np.cross(N1, N2)
    
    # Check if planes are parallel (direction vector is zero)
    if np.allclose(direction_vector, 0):
        return None  # Planes are parallel and do not intersect
    
    # Find a point on the line of intersection by solving for x, y, z
    # We solve for two coordinates and express the third in terms of them
    # Choose the largest absolute value in the direction vector to avoid division by zero
    if abs(direction_vector[0]) > abs(direction_vector[1]) and abs(direction_vector[0]) > abs(direction_vector[2]):
        # Solve for y and z, express x in terms of them
        A = np.array([[b1, c1], [b2, c2]])
        b = np.array([d1 - a1 * 0, d2 - a2 * 0])  # Assume x = 0 for simplicity
        y, z = np.linalg.solve(A, b)
        point_on_line = (0, y, z)
    elif abs(direction_vector[1]) > abs(direction_vector[2]):
        # Solve for x and z, express y in terms of them
        A = np.array([[a1, c1], [a2, c2]])
        b = np.array([d1 - b1 * 0, d2 - b2 * 0])  # Assume y = 0 for simplicity
        x, z = np.linalg.solve(A, b)
        point_on_line = (x, 0, z)
    else:
        # Solve for x and y, express z in terms of them
        A = np.array([[a1, b1], [a2, b2]])
        b = np.array([d1 - c1 * 0, d2 - c2 * 0])  # Assume z = 0 for simplicity
        x, y = np.linalg.solve(A, b)
        point_on_line = (x, y, 0)
    
    return {
        "point_on_line": tuple(point_on_line),
        "direction_vector": tuple(direction_vector / np.linalg.norm(direction_vector))  # Normalize direction vector
    }

def intersect_lines(line1, line2):
    """
    Finds the intersection point of two lines in 2D space, given in the form (a, b, c) for each line.

    Parameters:
    - line1: Coefficients of the first line as a tuple (a1, b1, c1) representing the line a1*x + b1*y = c1.
    - line2: Coefficients of the second line as a tuple (a2, b2, c2) representing the line a2*x + b2*y = c2.

    Returns:
    - A tuple (x, y) representing the intersection point if it exists, or None if the lines are parallel.
    """
    
    # Unpack line coefficients
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    
    # Set up coefficient matrix and constant vector
    A = np.array([[a1, b1], [a2, b2]])
    C = np.array([c1, c2])
    
    # Check if the determinant is zero (indicating parallel lines)
    if np.isclose(np.linalg.det(A), 0):
        return None  # Lines are parallel and do not intersect
    
    # Solve the system of equations for (x, y)
    intersection_point = np.linalg.solve(A, C)
    
    return tuple(intersection_point)
# intersection_plane_of_spheres(center1, radius1, center2, radius2)

if __name__ == "__main__":
   get_square_distance_map()
   