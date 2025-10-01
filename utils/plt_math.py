import numpy as np
import math 

def get_rotation_matrix(roll,pitch,yaw):
    # Yaw rotation matrix (rotation around z-axis)
    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Pitch rotation matrix (rotation around y-axis)
    R_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # Roll rotation matrix (rotation around x-axis)
    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # Combined rotation matrix
    R = R_yaw @ R_pitch @ R_roll
    return R


def slope_from_angle(degrees):
    # Convert angle to radians
    radians = math.radians(degrees)
    # Calculate slope
    slope = math.tan(radians)
    return slope