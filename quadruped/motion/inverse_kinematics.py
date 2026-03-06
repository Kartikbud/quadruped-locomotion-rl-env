# comment for testing
import math
import numpy as np

def get_joint_angles(pose):

    def safe_acos(x):
        return math.acos(min(1.0, max(-1.0, x)))

    def safe_asin(x):
        return math.asin(min(1.0, max(-1.0, x)))

    # in cm
    # these were derived from reading the tf-transformations between the joints
    # copy values of og: off1 = 1.1369, off2 = 6.3763, upper = 10.9868, lower = 14.458
    #offset_2 = 6.3  # y offset ##either 4.5 or 6.3
    #offset_1 = 0.9  # z offset
    #u = 11.05893304  # upper leg length
    #l = 12.7  # lower leg length

    #--their offsets--
    offset_2 = 6.3763
    offset_1 = 1.1369
    u = 10.9868
    l = 14.458

    #--gpt offsets--
    # offset_2 = 4.5
    # offset_1 = 0.870
    # u = 11.059
    # l = 12.650

    #---onshape offsets
    # offset_1 = 1.087
    # offset_2 = 4.170
    # u = 10.63058253
    # l = 14.64206068

    #--claude offsets are same as gpt


    # points
    x = pose[0]
    y = pose[1]
    z = pose[2]

    # all angles are radians
    # ----Y-Z View----
    h1 = math.sqrt(offset_1**2 + offset_2**2)
    h2 = math.sqrt(y**2 + z**2)
    alpha_0 = math.atan2(y, z)
    alpha_1 = math.atan(offset_2/offset_1)
    alpha_2 = math.atan(offset_1/offset_2)
    alpha_3 = safe_asin((h1*math.sin(alpha_2 + (math.pi/2)))/h2)
    alpha_4 = (math.pi/2) - (alpha_3 + alpha_2)
    alpha_5 = alpha_1 - alpha_4
    abduction = alpha_0 - alpha_5  # hip abduction joint angle in
    # abduction = math.atan2(y - offset_2, -(z - offset_1))
    r0 = (h1 * math.sin(alpha_4))/math.sin(alpha_3)

    # ----X-Z View----
    h = math.sqrt(r0**2 + x**2)
    beta = safe_asin(x/h)
    hip = safe_acos((u**2 + h**2 - l**2)/(2*u*h)) - beta
    knee = safe_acos((u**2 + l**2 - h**2)/(2*u*l))

    hip_angle_offset = -1.309
    knee_angle_offset = (2*math.pi/3)

    return [abduction, hip + hip_angle_offset, (knee - math.pi) + knee_angle_offset]
    #return [abduction, hip, (knee - math.pi)]
