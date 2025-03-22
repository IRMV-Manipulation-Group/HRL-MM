import math
import numpy as np
import modern_robotics as mr
from scipy.spatial.transform import Rotation

def RotationMatrixToEulerAngles(R) :

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def EulerAnglesToRotationMatrix(theta) :
    
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
        
        
                    
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])

    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

def GetMatrix(p, theta) :
    
    T_x = np.array([[1,         0,                  0                   , 0],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) , 0],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  , 0],
                    [0, 0, 0, 1]
                    ])
        
                    
    T_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  , 0],
                    [0,                     1,      0                   , 0],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  , 0],
                    [0, 0, 0, 1]
                    ])
                
    T_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0, 0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0, 0],
                    [0,                     0,                      1, 0],
                    [0, 0, 0, 1]
                    ])

    T = np.dot(T_z, np.dot( T_y, T_x ))
    T[0][3]=p[0]
    T[1][3]=p[1]
    T[2][3]=p[2]
    return T

def VelTrans(T):
    """Computes the adjoint representation of a homogeneous transformation
    matrix

    :param T: A homogeneous transformation matrix
    :return: The 6x6 adjoint representation [AdT] of T

    Example Input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        np.array([[1, 0,  0, 0, 0,  0],
                  [0, 0, -1, 0, 0,  0],
                  [0, 1,  0, 0, 0,  0],
                  [0, 0,  3, 1, 0,  0],
                  [3, 0,  0, 0, 0, -1],
                  [0, 0,  0, 0, 1,  0]])
    """
    R, p = mr.TransToRp(T)
    # return np.r_[np.c_[R,  -mr.VecToso3(p) @ R],
    #              np.c_[np.zeros((3, 3)), R]]
    return np.r_[np.c_[R, np.zeros((3, 3))],
                 np.c_[np.zeros((3, 3)), R]]

def GetMatrixFromPosAndQuat(pos, Quat):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(Quat).as_matrix()
    T[:3, 3] = pos
    return T