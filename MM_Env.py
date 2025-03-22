import sys
import random
import roboticstoolbox as rtb
from Basic_RL_tools.panda_robot import Panda_new
import qpsolvers as qp
from scipy.spatial.transform import Rotation
import math
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import pybullet as p
import modern_robotics as mr
import pybullet_data
import Basic_RL_tools.trans as trans
sys.path.append('./DMP')
sys.path.append('./Basic_RL_tools')
from dmp_discrete import dmp_discrete # modified version

class MMEnv(gym.Env):
  def __init__(self,
                renders=False,
                seed = None,
                decay_c= 20,
                reward_scale_list = None,
                augment=True
               ):
    self.scale_forcebility = reward_scale_list["scale_forcebility"]
    self.joint_velocity_penalty = reward_scale_list["joint_velocity_penalty"]
    self.box_base_collision_penalty = reward_scale_list["box_base_collision_penalty"]
    self.joint_position_penalty = reward_scale_list["joint_position_penalty"]
    self.base_collison_penalty = reward_scale_list["base_collison_penalty"]
    self.manipulability_penalty = reward_scale_list["manipulability_penalty"]
    self.augment = augment

    # reward config
    self.force_limit = 30
    self._timeStep = 1. / 100.
    self.renders = renders

    self._max_episode_steps = int(200)

    if self.renders:
      cid = p.connect(p.GUI_SERVER)
      if (cid < 0):
        cid = p.connect(p.GUI_SERVER)
      p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # GUI off
      p.resetDebugVisualizerCamera(3.5, 90, -50, [0, 0, 0])
    else:
      p.connect(p.DIRECT)

    self.seed(seed)
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    p.resetSimulation() 
    # action space
    self.delta_x_threshold = 50
    self.delta_y_threshold = self.delta_x_threshold

    action_high = np.array([self.delta_x_threshold, self.delta_y_threshold]) 
    action_low = np.array([-self.delta_x_threshold, -self.delta_y_threshold]) 
    self.action_space = spaces.Box(action_low, action_high)

    # obsevation space
    self.arm_joint_limit_low = np.array([-166, -101, -166, -176, -166, -1, -166]) * np.pi / 180
    self.arm_joint_limit_high = np.array([166, 101, 166, -4, 166, 215, 166]) * np.pi / 180

    self.force_state_low = np.array([-410]*3)
    self.force_state_high = np.array([410]*3)

    self.torque_state_high = np.array([87, 87, 87, 87, 12, 12, 12])

    # ---some parameters for Mobile Manipulation---
    self.arm_motor_joint = [19, 20, 21, 22, 23, 24, 25] # dim 7

    self.EndEffectorIndex = 27
    self.ConstraintIndex = 28
    self.ForceJointIndex = 28
    self.FrontLaserIndex = 16
    self.ArmBaseLinkIndex = 18
    self.EndEffectorName = 'panda_hand'

    self.CollisionLink1 = 29
    self.CollisionLink2 = 30
    self.CollisionLink3 = 31
    self.CollisionLink4 = 32

    # set gravity
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(self._timeStep)   

    self.TablePos = [0, 0, -0.3]
    self.TableID = p.loadURDF('./assets/table_urdf/table.urdf', self.TablePos, globalScaling=1.4)

    self.baseStartPos = [2.273343178993374, 1.6589285278302655, 0.0]
    self.baseStartOrientation = [0.0, 0.0, 0.8807038841465188, -0.4736672549894441]

    self.RobotID = p.loadURDF("./assets/MM_urdf/ridgeback_panda.urdf", self.baseStartPos, self.baseStartOrientation, useFixedBase=True)

    BoxStartPos = [1.68, 0.55, 0.601]
    BoxStartOrientation = p.getQuaternionFromEuler([0, 0, np.pi])

    self.BoxID = p.loadURDF("./assets/box_urdf/box.urdf", BoxStartPos, BoxStartOrientation)
    
    initial_arm_joint_position = [[1.5531803], 
                                  [0.04802554], 
                                  [-1.60343711], 
                                  [-1.6899666], 
                                  [1.77526115], 
                                  [2.49254004], 
                                  [-1.76586893]]

    p.resetJointStatesMultiDof(self.RobotID, self.arm_motor_joint, targetValues=initial_arm_joint_position)

    p.enableJointForceTorqueSensor(self.RobotID, self.ForceJointIndex)
    
    for i in self.arm_motor_joint:
      p.enableJointForceTorqueSensor(self.RobotID, i)

    p.loadURDF("plane100.urdf", useMaximalCoordinates=True)

    for t in range(100):
      p.stepSimulation()
      p.resetJointStatesMultiDof(self.RobotID, self.arm_motor_joint, targetValues=initial_arm_joint_position)
    
    self.EE_goal = [0, 0.8]

    self.EE_pos_start, self.EE_ori_start, _, _, _, _ = p.getLinkState(self.RobotID, self.EndEffectorIndex)
    self.EE_start = [self.EE_pos_start[0], self.EE_pos_start[2]]

    self.C1_to_O_base_frame = np.array([0.49, 0.4, 0])
    self.C2_to_O_base_frame = np.array([0.49, -0.4, 0])
    self.C3_to_O_base_frame = np.array([-0.49, 0.4, 0])
    self.C4_to_O_base_frame = np.array([-0.49, -0.4, 0])

    # generate trajectory
    s_vals = np.linspace(0, 1, self._max_episode_steps)
    self.traj = np.zeros((2, self._max_episode_steps))
    
    self.traj[0,:] = +s_vals* (self.EE_goal[0]-self.EE_start[0]) + self.EE_start[0]
    self.traj[1,:] = +s_vals* (self.EE_goal[1]-self.EE_start[1]) + self.EE_start[1]

    # DMP learning
    self.dmp = dmp_discrete(n_dmps=2, 
                            n_bfs=100, 
                            dt=1.0/self._max_episode_steps, 
                            decay_c= decay_c)
    self.dmp.learning(self.traj, plot=False)

    self.panda = Panda_new()

    self.laser_down_sample = 5
    self.laser_noise_threshold = 0.01
    self.arm_noise_threshold = 1e-4
    w = 1/self.torque_state_high
    self.w = np.diag(w)
    self.u = np.array([-0.707, 0, 0.707, 0, 0, 0]).reshape(-1, 1)

    self.stateId = p.saveState()
    self.RefreshRobot()

    # get high level state space  
    self.x = 0
    high_state = self.get_high_Observation()
    self.observation_space = spaces.Box(-np.ones(high_state.shape), np.ones(high_state.shape))

    # get low level state space and action space 
    self.desire_v = np.zeros(6)   
    low_state = self.get_low_Observation()
    self.low_observation_space = spaces.Box(-np.ones(low_state.shape), np.ones(low_state.shape))

    self.joint_velocity_threshold = 150 / 180 * np.pi
    self.base_linear_velocity_threshold = 0.5
    self.base_angular_velocity_threshold = 0.5

    action_high = np.array([self.joint_velocity_threshold,
                            self.base_linear_velocity_threshold, 
                            self.base_linear_velocity_threshold,
                            self.base_angular_velocity_threshold]) 
    action_low = -action_high
    self.low_action_space = spaces.Box(action_low, action_high)
    
  def reset(self, test=None):
    self.dmp.reset_state()
    self.step_number=0
    info = {}
    p.restoreState(self.stateId)

    p.changeDynamics(self.BoxID, -1, mass=3.3, lateralFriction=0.5)

    Box_pos, Box_ori = p.getBasePositionAndOrientation(self.BoxID)

    Box_euler = p.getEulerFromQuaternion(Box_ori)

    T_box = trans.GetMatrix(Box_pos, Box_euler)
    Hand_pos, Hand_ori, _, _, _, _, _, _ = p.getLinkState(self.RobotID, self.ConstraintIndex, 1)
    Hand_euler = p.getEulerFromQuaternion(Hand_ori)
    T_hand = trans.GetMatrix(Hand_pos, Hand_euler)
    
    T_hand_box = np.linalg.pinv(T_box) @ T_hand

    pos = list(T_hand_box[:3, 3])
    R = T_hand_box[:3, :3]
    euler = list(trans.RotationMatrixToEulerAngles(R))

    cid = p.createConstraint(self.RobotID, 
                              self.ConstraintIndex, 
                              self.BoxID, 
                              -1, 
                              p.JOINT_FIXED,
                              [0, 0, 0],
                              parentFramePosition = [0, 0, 0],
                              childFramePosition = pos,
                              parentFrameOrientation = p.getQuaternionFromEuler([0, 0, 0]),
                              childFrameOrientation = p.getQuaternionFromEuler(euler),
                              )

    self.num_JointLimit = 0

    EE_pos, EE_ori, _, _, _, _, _, _ = p.getLinkState(self.RobotID, self.EndEffectorIndex, 1)

    EE_euler = p.getEulerFromQuaternion(EE_ori)
    self.EE_initial_pos = np.array(EE_pos)
    self.EE_initial_euler = np.array(EE_euler)
    self.EE_initial_rotation_matrix = trans.EulerAnglesToRotationMatrix(EE_euler)

    p.stepSimulation()
    self.RefreshRobot()
    state = self.get_high_Observation()
    box_pos, box_ori, _, _, _, _, _, _ = p.getLinkState(self.RobotID, self.BoxID, 1)
    self.desire_v = np.zeros(6)
    return state, info

  def __del__(self):
    p.disconnect()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    random.seed(seed)
    return [seed]

  def pre_step_high(self, action):
    # delta x, delta y, gain
    lb = self.action_space.low
    ub = self.action_space.high
    scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
    scaled_action = np.clip(scaled_action, lb, ub)

    self.step_number = self.step_number + 1  

    self.force_list = []

    self.EE_desire_pos_old = np.array([self.dmp.y[0], self.EE_initial_pos[1], self.dmp.y[1]])

    self.y_reproduce, self.dy_reproduce, self.ddy_reproduce, self.x = self.dmp.step(scaled_action[:2])
  
    self.EE_desire_pos = np.array([self.y_reproduce[0], self.EE_initial_pos[1], self.y_reproduce[1]])

    self.EE_desire_pos_delta = self.EE_desire_pos - self.EE_desire_pos_old

  def post_step_high(self):
    state = self.get_high_Observation()
    Total_force_ee = np.linalg.norm(self.EndForce)

    box_pos, box_ori = p.getBasePositionAndOrientation(self.BoxID)
    info = {}

    reward = -(Total_force_ee / self.force_limit) ** 2 
    done = False
    # time done
    if self.step_number >= int(self._max_episode_steps-1) :
      done = True
      
    return state, reward, done, None, info

  def Calculate_desire_v(self, t):
    EE_desire_pos_t = self.EE_desire_pos_old + self.EE_desire_pos_delta * (t+1) /10
    T_ee_desire = np.eye(4)
    T_ee_desire[:3, :3] = self.EE_initial_rotation_matrix
    T_ee_desire[:3, 3] = EE_desire_pos_t # self.EE_initial_pos # 
    Gain = [1, 1, 1, 10, 10, 10]

    v, arrived = rtb.p_servo(self.EE_T, T_ee_desire, Gain, 0.01, method='angle-axis')
    self.desire_v = v    
    return v

  def step_low(self, control_arm, control_base):
    # control_arm, control_base, qpdone = self.Calculate_vel_from_qp(v)
    self.control_arm = control_arm
    self.control_base = control_base
    self.take_control_vel(control_arm, control_base)
    p.stepSimulation()
    self.RefreshRobot()
    low_state = self.get_low_Observation()
    info = {}
    laser_info = self.getLaserState(False, self.laser_down_sample, Nosie = self.laser_noise_threshold)
    info['laser_info'] = laser_info
    reward, done = self.get_low_reward()
    return low_state, reward, done, None, info

  def low_reset(self):
    low_obs = self.get_low_Observation()
    laser_info = self.getLaserState(False, self.laser_down_sample, Nosie = self.laser_noise_threshold)
    return low_obs, laser_info

  def take_control_reset(self, control_arm, control_base):
    ArmJointsPositions = self.ArmJointsPositions
    GoalJointPositions = (ArmJointsPositions + np.array(control_arm) * self._timeStep).reshape(-1, 1)
    p.resetJointStatesMultiDof(self.RobotID, self.arm_motor_joint, targetValues=GoalJointPositions)

    # velocity control
    vel_lin=[control_base[0], control_base[1], 0]
    vel_ang=[0, 0, control_base[2]]

    # delta position control 
    base_pos_new = [self.BasePos[0]+vel_lin[0]* self._timeStep, self.BasePos[1]+vel_lin[1]* self._timeStep, 0]
    base_ori_new = [0, 0, self.BaseOri_euler[2]+vel_ang[2]* self._timeStep]
    p.resetBasePositionAndOrientation(self.RobotID, base_pos_new, p.getQuaternionFromEuler(base_ori_new))
  
  def take_control_vel(self, control_arm, control_base):
    p.setJointMotorControlArray(self.RobotID,
                                self.arm_motor_joint,
                                p.VELOCITY_CONTROL,
                                targetVelocities = control_arm)

    vel_lin=[control_base[0], control_base[1], 0]
    vel_ang=[0, 0, control_base[2]]
    p.resetBaseVelocity(self.RobotID, vel_lin, vel_ang)

  def get_high_Observation(self):
    box_pos, box_ori = p.getBasePositionAndOrientation(self.BoxID)
    box_euler = p.getEulerFromQuaternion(box_ori)
    state = np.array(self.EE_euler + self.EE_pos + tuple(self.EndForce) + box_euler + box_pos)
    if self.augment:
      state = np.concatenate((state, np.array([self.x])))
    return state

  def get_low_Observation(self):
    desire_EE_v_Base = self.R6_Base_World.T @ self.desire_v
    self.local_u_armbase = self.R6_World_ArmBase @ self.u
    ArmJointsPositions = self.ArmJointsPositions+np.random.normal(0, self.arm_noise_threshold, 7)
    
    state = np.concatenate((ArmJointsPositions, desire_EE_v_Base.reshape(-1), self.local_u_armbase.reshape(-1)))
    return state
  
  def getKinematics(self):
    base_pos, base_ori = p.getBasePositionAndOrientation(self.RobotID)
    base_ori_euler = p.getEulerFromQuaternion(base_ori)
    theta_z = base_ori_euler[-1]

    link_trn = self.EE_pos
    self.panda.q = self.ArmJointsPositions
    Jaco_arm = self.panda.jacob0(self.panda.q)

    R_w2b = np.array([
      [np.cos(theta_z), np.sin(theta_z), 0],
      [-np.sin(theta_z), np.cos(theta_z), 0],
      [0, 0, 1]
    ] )

    EE_to_base_xy = R_w2b @ (np.array(link_trn) - np.array(base_pos))

    Jaco_base = np.array(
      [[1, 0, -EE_to_base_xy[1]],
      [0, 1, EE_to_base_xy[0]],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 1]]
    )

    Jaco = np.concatenate((Jaco_arm, Jaco_base @  R_w2b), axis=1)
    Jaco = np.r_[np.c_[R_w2b.T, np.zeros((3, 3))],
                  np.c_[np.zeros((3, 3)), R_w2b.T]] @ Jaco

    return Jaco

  def checkJointLimit(self):
    result1 = np.greater(self.JointsPositions, self.arm_joint_limit_low)
    result2 = np.greater(self.arm_joint_limit_high, self.JointsPositions)
    if all(result1) and all(result2):
      return False # in the range
    else:
      return True #
     
  def getCollisionAdj(self):
    R_b2w = self.Base_T[:3, :3]

    p_C1_to_O_world_frame = R_b2w @ self.C1_to_O_base_frame
    p_C2_to_O_world_frame = R_b2w @ self.C2_to_O_base_frame
    p_C3_to_O_world_frame = R_b2w @ self.C3_to_O_base_frame
    p_C4_to_O_world_frame = R_b2w @ self.C4_to_O_base_frame
 
    Adj1 =  np.c_[np.eye(3), mr.VecToso3(-p_C1_to_O_world_frame)]
    Adj2 =  np.c_[np.eye(3), mr.VecToso3(-p_C2_to_O_world_frame)]
    Adj3 =  np.c_[np.eye(3), mr.VecToso3(-p_C3_to_O_world_frame)]
    Adj4 =  np.c_[np.eye(3), mr.VecToso3(-p_C4_to_O_world_frame)]

    return Adj1[:, [0, 1, 5]], Adj2[:, [0, 1, 5]], Adj3[:, [0, 1, 5]], Adj4[:, [0, 1, 5]] 


  def getLaserState(self, render, downsample, RayLength=2, Nosie=0.01):
    front_laser_pos, front_laser_ori, _, _, _, _ = p.getLinkState(self.RobotID, self.FrontLaserIndex)

    T_Radar = trans.GetMatrixFromPosAndQuat(front_laser_pos, front_laser_ori)

    begins = front_laser_pos

    self.RayNum = 1080
    RayNum = int(self.RayNum/downsample)

    rayFroms = [begins for _ in range(RayNum)] 

    angles = list(np.array(range(RayNum)) / RayNum * (2.356194496154785*2) - 2.356194496154785)

    rayTos = [(T_Radar @ 
        np.array([RayLength * math.cos(angle),
        RayLength * math.sin(angle),
        0, 1]))[:3]
    for angle in angles]

    results = p.rayTestBatch(rayFroms, rayTos)

    results = [x[2] for x in results]
    RayResults = np.stack(results) * RayLength
    Add_Nosie_ = np.random.normal(0, Nosie, RayResults.shape)
    index = (RayResults<RayLength)

    RayResults[index] += Add_Nosie_[index]
    RayResults = np.array(RayResults, dtype=np.float32)

    if render:
      hitRayColor = [0, 1, 0]
      missRayColor = [1, 0, 0]
      p.removeAllUserDebugItems()
      for index, result in enumerate(results):
        if result[0] == -1:
            p.addUserDebugLine(rayFroms[index], rayTos[index], missRayColor)
        else:
            p.addUserDebugLine(rayFroms[index], rayTos[index], hitRayColor)
    
    return RayResults
  
  def RefreshRobot(self):
    JointResults = p.getJointStates(self.RobotID, self.arm_motor_joint)

    JointResults = [list(x[:2] + x[-1:]) for x in JointResults]
    JointResults = np.array(JointResults)
    JointsPositions = JointResults[:, 0]
    JointsVelocities = JointResults[:, 1]
    JointTorques = JointResults[:, 2]

    _, _, EndForce, _ = p.getJointState(self.RobotID, self.ForceJointIndex)
    EE_pos, EE_ori, _, _, _, _, _, _ = p.getLinkState(self.RobotID, self.EndEffectorIndex, 1)
    EE_T = trans.GetMatrixFromPosAndQuat(EE_pos, EE_ori)

    BasePos, BaseOri = p.getBasePositionAndOrientation(self.RobotID)
    Base_T = trans.GetMatrixFromPosAndQuat(BasePos, BaseOri)
    BaseOri_euler = p.getEulerFromQuaternion(BaseOri)
    BasePositions3 = [BasePos[0], BasePos[1], BaseOri_euler[-1]]

    BaseVeloLin, BaseVeloAng = p.getBaseVelocity(self.RobotID)
    BaseVelocities = [BaseVeloLin[0], BaseVeloLin[1], BaseVeloAng[2]]

    T_EE_ArmBase = np.array(self.panda.fkine(JointsPositions))

    ArmBasePos, ArmBaseOri, _, _, _, _, _, _ = p.getLinkState(self.RobotID, self.ArmBaseLinkIndex, 1)
    R_ArmBase_World = Rotation.from_quat(ArmBaseOri).as_matrix()

    # -----------collsion pos-------
    c1_pos, _, _, _, _, _, v_c1_lin_true, v_c1_ang_true = p.getLinkState(self.RobotID, self.CollisionLink1, 1)
    c2_pos, _, _, _, _, _, v_c2_lin_true, v_c2_ang_true = p.getLinkState(self.RobotID, self.CollisionLink2, 1)
    c3_pos, _, _, _, _, _, v_c3_lin_true, v_c3_ang_true = p.getLinkState(self.RobotID, self.CollisionLink3, 1)
    c4_pos, _, _, _, _, _, v_c4_lin_true, v_c4_ang_true = p.getLinkState(self.RobotID, self.CollisionLink4, 1)
    collsion_pos = [c1_pos[1], c2_pos[1], c3_pos[1], c4_pos[1]]

    # ------ refresh data------------
    self.ArmJointsPositions = JointsPositions
    # print(JointsPositions)
    self.ArmJointsVelocities = JointsVelocities
    self.BasePos = BasePos
    self.BaseOri = BaseOri
    self.BasePositions3 = BasePositions3
    self.Base_T = Base_T
    self.R6_Base_World = np.r_[np.c_[self.Base_T[:3, :3], np.zeros((3, 3))],
                  np.c_[np.zeros((3, 3)), self.Base_T[:3, :3]]]

    self.BaseOri_euler = BaseOri_euler
    self.EE_pos = EE_pos
    self.EE_euler = p.getEulerFromQuaternion(EE_ori)    
    self.EE_T = EE_T
    self.R_EE_World = self.EE_T[:3, :3]
    
    self.BaseVelocities = BaseVelocities
    self.T_EE_ArmBase = T_EE_ArmBase
    self.R_ArmBase_World = R_ArmBase_World
    self.R_World_ArmBase = self.R_ArmBase_World.T

    self.R6_World_ArmBase = np.r_[np.c_[self.R_World_ArmBase, np.zeros((3, 3))],
                    np.c_[np.zeros((3, 3)), self.R_World_ArmBase]]
    self.local_u_armbase = self.R6_World_ArmBase @ self.u
    self.collsion_pos = collsion_pos

    self.EndForce = -np.array(EndForce[:3])

    self.JointTorques = JointTorques
    self.Jaco = self.getKinematics()
    self.EndForce2World = self.R_EE_World @ self.EndForce
    

  def Calculate_vel_from_base(self, v, other_action):
    base_lin_vel_world = self.Base_T[:3, :3] @ np.array([other_action[1], other_action[2], 0])
    action_base_world = [base_lin_vel_world[0], base_lin_vel_world[1], other_action[-1]]

    J1 = self.Jaco[:, 0].reshape(-1, 1)
    J2 = self.Jaco[:, 1:7]
    J3 = self.Jaco[:, 7:]

    q = np.linalg.pinv(J2) @ (v.reshape(-1, 1) - J1 * other_action[0] -J3 @ np.array(action_base_world).reshape(-1, 1))

    control_arm = [other_action[0]]+ q.reshape(-1).tolist()
    control_base = action_base_world

    return control_arm, control_base
  
  def Calculate_vel_from_qp(self, v):
    # -----------------redundant----------------
    # arm dof
    na = 7

    # base dof
    nb = 3

    # base collsion point number
    nc = 4

    # total dof
    n = na + nb

    # Gain term (lambda) for control minimisation
    Y = 0.01 # 0.01

    # Quadratic component of objective function
    Q = np.eye(n + 6)
    c = np.zeros(n + 6)

    # Joint velocity component of Q
    Q[:n, :n] *= Y

    # Slack component of Q
    Q[n:, n:] = 1000 * np.eye(6)

    # The equality contraints
    Aeq = np.c_[self.Jaco, np.eye(6)]
    beq = v.reshape((6,))

    # --------
    # The inequality constraints for joint limit avoidance
    # and the base avoidance

    Ain = np.zeros((na+nc, n + 6))
    bin = np.zeros(na+nc)

    # The minimum angle (in radians) in which the joint is allowed to approach
    # to its limit
    ps = 0.05

    # The influence angle (in radians) in which the velocity damper
    # becomes active
    pi = 0.9

    # Form the joint limit velocity damper
    Ain[:na, :na], bin[:na] = self.panda.joint_velocity_damper(ps, pi, na)

    # base collision avoidance
    Ac = np.zeros((nc, nb))
    bc = np.zeros(nc)

    c_pos = self.collsion_pos

    c_threshold = 0.67
    gain = 1

    ds = 0.003
    di = 0.05

    adj = self.getCollisionAdj()
    for i in range(nc):
      d = c_pos[i] - c_threshold
      if d <= di:
        bc[i] = -gain * (d - ds) / (di - ds)
        Ac[i, :] = -adj[i][1]

    Ain[na:na+nc, na:na+nb] = Ac
    bin[na:na+nc] = bc
    
    k_force = 100 

    self.forcebility = self.panda.forcebility(self.w, self.u)
    
    # if self.forcebility >= 0.01:  
    c = np.r_[self.panda.jacobf_MM(self.w, self.u, self.R_World_ArmBase, nb).reshape((na+nb,)), np.zeros(6)] * k_force

    # Linear component of objective function: the manipulability Jacobian

    # The lower and upper bounds on the joint velocity and slack variable
    lb = -np.r_[self.panda.qdlim[:na], 1.1 * np.ones(3), 10 * np.ones(6)]
    ub = np.r_[self.panda.qdlim[:na], 1.1 * np.ones(3), 10 * np.ones(6)]
    # print(lb, ub)
    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver='cvxopt')
    if type(qd) == type(None):
      qpdone = True
      control_arm = np.zeros(na)
      control_base = np.zeros(nb)  
    else:
      qpdone = False
      control_arm = qd[:na]
      control_base = qd[na: na+nb]       

    return control_arm, control_base, qpdone


  def _reward_forcebility(self, R_World_ArmBase, ArmJointsPositions):
    self.forcebility = self.panda.forcebility_MM(self.w, self.u, R_World_ArmBase, ArmJointsPositions)[0, 0]

    if self.forcebility >= 0.04:
      reward = -50 
      done = True
    else:
      reward = 0
      done = False

    return reward, done
  
  def _reward_mainipulability(self, threshod, ArmJointsPositions):
    axes = [False, False, True, True, True, False]
    manipulability = self.panda.manipulability(ArmJointsPositions, end=self.EndEffectorName, axes=axes)
    if manipulability <= threshod:
      reward = self.manipulability_penalty
      done = True
    else:
      reward = 0
      done = False
    return reward, done

  def _reward_self_collision(self, rate, ArmJointsPositions):
    during = self.arm_joint_limit_high - self.arm_joint_limit_low 
    new_low = self.arm_joint_limit_low + (1-rate) * during /2
    new_high = self.arm_joint_limit_high - (1-rate) * during /2

    if any(ArmJointsPositions > new_high) or any(ArmJointsPositions < new_low):
      reward = self.joint_position_penalty
      done = True
    else:
      reward = 0
      done = False
    return reward, done
  
  def _reward_joint_velocity_limit(self, control_arm):
    if any(np.array(control_arm)>self.joint_velocity_threshold):
      reward = self.joint_velocity_penalty
      done = True
    else:
      reward = 0 
      done = False
    return reward, done

  def _reward_action_am(self, control_arm):
    return 1- np.linalg.norm(control_arm), False

  def _reward_EE_base(self):
    if self.T_EE_ArmBase[0, 3] <= 0.4:
      reward = self.box_base_collision_penalty
      print('T_EE_ArmBase')
      done = True
    else:
      reward = 0
      done = False
    return reward, done

  def _reward_base_collision(self, threshod):
    # mesh collsion
    pts = p.getClosestPoints(bodyA=self.TableID, 
                             bodyB=self.RobotID, 
                             distance=100)

    if len(pts) > 0:
      distance = pts[0][8]
      if distance <= threshod:
        reward = self.base_collison_penalty
        done = True

        print('base collison')
      else:
        reward = 0
        done = False
    else:
        reward = 0
        done = True

    return reward, done
  
  def _reward_base_collision_easy(self, threshold, collsion_pos):
    collsion_pos = np.array(collsion_pos)
    if any(collsion_pos<threshold):
      return self.base_collison_penalty, True
    return 0, False

  def predict_next_state(self, control_arm, control_base, step=1):
    ArmJointsPositions = self.ArmJointsPositions
    NextArmJointPositions = (ArmJointsPositions + np.array(control_arm) * self._timeStep * step)

    # velocity control
    vel_lin=[control_base[0], control_base[1], 0]
    vel_ang=[0, 0, control_base[2]]

    # delta position control 
    base_pos_new = [self.BasePos[0]+vel_lin[0]* self._timeStep*step, self.BasePos[1]+vel_lin[1]* self._timeStep*step, 0]
    base_ori_new = [0, 0, self.BaseOri_euler[2]+vel_ang[2]* self._timeStep*step]
    Base_T = trans.GetMatrixFromPosAndQuat(base_pos_new, p.getQuaternionFromEuler(base_ori_new))
    R_b2w = Base_T[:3, :3]

    collsion_pos1 = (R_b2w @ self.C1_to_O_base_frame)[1] + base_pos_new[1]
    collsion_pos2 = (R_b2w @ self.C2_to_O_base_frame)[1] + base_pos_new[1]
    collsion_pos3 = (R_b2w @ self.C3_to_O_base_frame)[1] + base_pos_new[1]
    collsion_pos4 = (R_b2w @ self.C4_to_O_base_frame)[1] + base_pos_new[1]

    predict_obs = {}

    predict_obs['R_World_ArmBase'] = R_b2w.T
    predict_obs['ArmJointsPositions'] = NextArmJointPositions
    predict_obs['control_arm'] = control_arm
    predict_obs['collsion_pos'] = [collsion_pos1, collsion_pos2, collsion_pos3, collsion_pos4]
    
    return predict_obs


  def get_low_reward(self, obs=None):
    if type(obs) == type(None):
      R_World_ArmBase = self.R_World_ArmBase
      ArmJointsPositions = self.ArmJointsPositions
      control_arm = self.control_arm
      collsion_pos = self.collsion_pos
    else:
      R_World_ArmBase = obs['R_World_ArmBase']
      ArmJointsPositions = obs['ArmJointsPositions']
      control_arm = obs['control_arm']
      collsion_pos = obs['collsion_pos']

    reward_forcebility, done_forcebility = self._reward_forcebility(R_World_ArmBase, ArmJointsPositions)
    reward_mainipulability, done_mainipulability = self._reward_mainipulability(0.005, ArmJointsPositions)
    reward_self_collision, done_self_collision = self._reward_self_collision(0.95, ArmJointsPositions)
    reward_joint_velocity_limit, done_joint_velocity_limit = self._reward_joint_velocity_limit(control_arm)
    reward_action_am, done_action_am = self._reward_action_am(control_arm)
    reward_base_collision, done_base_collision = self._reward_base_collision_easy(0.05, collsion_pos)

    reward = reward_forcebility + reward_mainipulability + \
              reward_self_collision + \
              reward_joint_velocity_limit + reward_action_am + \
              reward_base_collision
    
    done = done_forcebility or done_mainipulability or \
            done_self_collision or \
            done_joint_velocity_limit or done_action_am or\
            done_base_collision 

    return reward, done
