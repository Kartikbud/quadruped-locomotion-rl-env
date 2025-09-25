import math
import numpy as np

###add phase paramater which essentially signifies an offset to apply, either 0 or -1 to the time variable and then modify if statements at bottom so no change is sent if < 0 or > 2
def generate_position_trajectory_point(L_span, rho, angular_vel, f_stand, time, swing, dt, length, width, leg, clearance, penetration): #rho is translation angle

    quadratic_bezier_matrix = np.array([
        [1, 0, 0],
        [-2, 2, 0],
        [1, -2, 1]
    ])

    curve_param = clearance
    support_curve_param = penetration

    swing_points = np.array([
        [0, 0], 
        [L_span, -curve_param], #height is negative because the inverse kinematics solver treats z pointing down as positive
        [2*L_span, 0]
    ])

    support_points = np.array([
        [2*L_span, 0],
        [L_span, support_curve_param],
        [0, 0]
    ])

    l = length
    w = width

    initial_stances = {
        "FL": [l/2, w/2],
        "FR": [l/2, -w/2],
        "BL": [-l/2, w/2],
        "BR": [-l/2, -w/2]
    }

    theta = dt * angular_vel

    pos = initial_stances[leg]

    dx = (pos[0]*math.cos(theta)) - (pos[1]*math.sin(theta)) - pos[0]
    dy = (pos[0]*math.sin(theta)) + (pos[1]*math.cos(theta)) - pos[1]

    yaw_swing_points = np.array([
        [0, 0, 0],
        [dx/2, dy/2, -curve_param],
        [dx, dy, 0]
    ])

    yaw_support_points = np.array([
        [dx, dy, 0],
        [dx/2, dy/2, support_curve_param],
        [0, 0, 0]
    ])
    
    def quadratic_bezier_curve(t, points): #for swing phase
        time_vector = np.array([1, t, t**2])
        point = time_vector @ quadratic_bezier_matrix @ points
        return point

    if swing:
        initial_point = quadratic_bezier_curve(time, swing_points)
        new_point = [(initial_point[0] * math.cos(rho)) + f_stand[0], (initial_point[0] * math.sin(rho)) + (f_stand[1]), initial_point[1] + f_stand[2]]
        yaw_point = quadratic_bezier_curve(time, yaw_swing_points)
        final_point = [new_point[0] + yaw_point[0], new_point[1] + yaw_point[1], new_point[2] + yaw_point[2]]
        return final_point
    else:
        initial_point = quadratic_bezier_curve(time, support_points)
        new_point = [(initial_point[0] * math.cos(rho)) + f_stand[0], (initial_point[0] * math.sin(rho)) + (f_stand[1]), initial_point[1] + f_stand[2]]
        yaw_point = quadratic_bezier_curve(time, yaw_support_points)
        final_point = [new_point[0] + yaw_point[0], new_point[1] + yaw_point[1], new_point[2] + yaw_point[2]]
        return final_point