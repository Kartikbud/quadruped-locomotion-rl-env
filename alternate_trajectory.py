import math
import numpy as np

###add phase paramater which essentially signifies an offset to apply, either 0 or -1 to the time variable and then modify if statements at bottom so no change is sent if < 0 or > 2
def generate_position_trajectory_point(L_span, rho, angular_vel, f_stand, time, swing): #rho is translation angle
    quadratic_bezier_matrix = np.array([
        [1, 0, 0],
        [-2, 2, 0],
        [1, -2, 1]
    ])

    curve_param = 6
    support_curve_param = 2

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
    
    def quadratic_bezier_curve(t, points): #for swing phase
        time_vector = np.array([1, t, t**2])
        point = time_vector @ quadratic_bezier_matrix @ points
        return point

    if swing:
        initial_point = quadratic_bezier_curve(time, swing_points)
        new_point = [initial_point[0] + f_stand[0], 0 + f_stand[1], initial_point[1] + f_stand[2]]
        return new_point
    else:
        initial_point = quadratic_bezier_curve(time, support_points)
        new_point = [initial_point[0] + f_stand[0], 0 + f_stand[1], initial_point[1] + f_stand[2]]
        return new_point