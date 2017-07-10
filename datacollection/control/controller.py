import os, sys, roslib, rospy, tf, actionlib, PyKDL

import numpy as np
from geometry_msgs.msg import PointStamped, PoseStamped
from trajectory_msgs.msg import JointTrajectoryPoint
from pr2_controllers_msgs.msg import JointTrajectoryGoal, JointTrajectoryAction, JointTrajectoryControllerState, JointControllerState, Pr2GripperCommandAction, Pr2GripperCommandGoal, PointHeadAction, PointHeadGoal
from pykdl_utils.kdl_kinematics import create_kdl_kin # TODO: Remove this dependency
import sandbox_dpark_darpa_m3.lib.hrl_dh_lib as dh # TODO: Remove this dependency that does not come with ROS


class Controller:
    def __init__(self, frame, verticalMovement=False, verbose=False):
        self.frame = frame
        self.verbose = verbose
        self.tf = tf.TransformListener()
        self.leftGripperAngle = 0


        # Set all arm movement parameters
        self.rightJointLimitsMax = np.radians([26.0, 68.0, 41.0, 0.01, 180.0, 0.01, 180.0])
        self.rightJointLimitsMin = np.radians([-109.0, -24.0, -220.0, -132.0, -180.0, -120.0, -180.0])
        self.leftJointLimitsMax = np.radians([109.0, 68.0, 220.0, 0.01, 270.0, 0.01, 180.0])
        self.leftJointLimitsMin = np.radians([-26.0, -24.0, -41.0, -132.0, -270.0, -120.0, -180.0])
        # self.initRightJointGuess = np.array([-0.236, 0.556, -0.091, -1.913, -1.371, -1.538, -3.372])
        self.initRightJointGuess = np.array([0.13, 0.2, 0.63, -3.23, -2.0, -0.96, -0.1])
        # self.initLeftJointGuess = np.array([0.203, 0.846, 1.102, -1.671, 5.592, -1.189, -3.640])
        self.initLeftJointGuess = np.array([0.74, 0.03, 1.05, -0.32, -1.63, -1.28, 4.22])
        if verticalMovement:
            self.initRightJointGuess = np.array([-0.29, -0.35, -1.56, -1.76, -1.22, -1.52, 3.04])
            self.initLeftJointGuess = np.array([0.20, 1.03, 1.22, -1.23, -0.51, -1.69, 8.75])

        self.initJoints()

        self.currentRightJointPositions = None
        self.currentLeftJointPositions = None
        # rospy.Subscriber('/r_arm_controller/state', JointTrajectoryControllerState, self.rightJointsCallback)
        # rospy.Subscriber('/l_arm_controller/state', JointTrajectoryControllerState, self.leftJointsCallback)

        self.rightArmClient = actionlib.SimpleActionClient('/r_arm_controller/joint_trajectory_action', JointTrajectoryAction)
        self.rightArmClient.wait_for_server()
        self.rightGripperClient = actionlib.SimpleActionClient('/r_gripper_controller/gripper_action', Pr2GripperCommandAction)
        self.rightGripperClient.wait_for_server()
        self.leftArmClient = actionlib.SimpleActionClient('/l_arm_controller/joint_trajectory_action', JointTrajectoryAction)
        self.leftArmClient.wait_for_server()
        self.leftGripperClient = actionlib.SimpleActionClient('/l_gripper_controller/gripper_action', Pr2GripperCommandAction)
        self.leftGripperClient.wait_for_server()
        self.pointHeadClient = actionlib.SimpleActionClient('/head_traj_controller/point_head_action', PointHeadAction)
        self.pointHeadClient.wait_for_server()

        # Initialize KDL for inverse kinematics
        self.rightArmKdl = create_kdl_kin(self.frame, 'r_gripper_tool_frame')
        self.rightArmKdl.joint_safety_lower = self.rightJointLimitsMin
        self.rightArmKdl.joint_safety_upper = self.rightJointLimitsMax

        self.leftArmKdl = create_kdl_kin(self.frame, 'l_gripper_tool_frame')
        self.leftArmKdl.joint_safety_lower = self.leftJointLimitsMin
        self.leftArmKdl.joint_safety_upper = self.leftJointLimitsMax

    def initJoints(self):
        msg = rospy.wait_for_message('/r_arm_controller/state', JointTrajectoryControllerState)
        self.rightJointNames = msg.joint_names
        self.initRightJointPositions = msg.actual.positions
        msg = rospy.wait_for_message('/l_arm_controller/state', JointTrajectoryControllerState)
        self.leftJointNames = msg.joint_names
        self.initLeftJointPositions = msg.actual.positions
        if self.verbose:
            print 'Right joint names:', self.rightJointNames
            print 'Left joint names:', self.leftJointNames

    def setJointGuesses(self, rightGuess=None, leftGuess=None):
        if rightGuess is not None:
            self.initRightJointGuess = rightGuess
        if leftGuess is not None:
            self.initLeftJointGuess = leftGuess

    def getGripperPosition(self, rightArm=True):
        # now = rospy.Time.now()
        # self.tf.waitForTransform(self.frame, ('r' if rightArm else 'l') + '_gripper_tool_frame', now, rospy.Duration(10.0))
        # Return the most revent transformation
        if rightArm:
            return self.tf.lookupTransform(self.frame, 'r_gripper_tool_frame', rospy.Time(0))
        else:
            return self.tf.lookupTransform(self.frame, 'l_gripper_tool_frame', rospy.Time(0))

    def grip(self, openGripper=True, maxEffort=200.0, rightArm=True, miniOpen=False, stopMovement=False):
        msg = Pr2GripperCommandGoal()
        if stopMovement:
            pos = rospy.wait_for_message('/r_gripper_controller/state', JointControllerState).process_value
            msg.command.position = pos
        else:
            msg.command.position = 0.1 if openGripper else (0.005 if miniOpen else 0.0)
        msg.command.max_effort = maxEffort if stopMovement or not openGripper else -1.0
        if rightArm:
            self.rightGripperClient.send_goal(msg)
        else:
            self.leftGripperClient.send_goal(msg)
        # self.rightGripperClient.send_goal_and_wait(msg)

    def rotateGripperWrist(self, angle):
        self.leftGripperAngle += angle
        rotatedWristAngles = list(self.initLeftJointPositions)
        rotatedWristAngles[-1] += self.leftGripperAngle
        self.moveToJointAngles(rotatedWristAngles, timeout=2.0, wait=True, rightArm=False)

    def moveGripperTo(self, position, rollpitchyaw=[-np.pi, 0.0, 0.0], timeout=1, useInitGuess=False, wait=False, rightArm=True, ret=False):
        # Move using IK and joint trajectory controller
        # Attach new pose to a frame
        poseData = list(position) + list(rollpitchyaw)
        frameData = PyKDL.Frame()
        poseFrame = dh.array2KDLframe(poseData)
        poseFrame = dh.frameConversion(poseFrame, frameData)
        pose = dh.KDLframe2Pose(poseFrame)

        # Create a PoseStamped message and perform transformation to given frame
        ps = PoseStamped()
        ps.header.frame_id = self.frame
        ps.pose.position = pose.position
        ps.pose.orientation = pose.orientation
        ps = self.tf.transformPose(self.frame, ps)

        # Perform IK
        if rightArm:
            ikGoal = self.rightArmKdl.inverse(ps.pose, q_guess=self.initRightJointGuess, min_joints=self.rightJointLimitsMin, max_joints=self.rightJointLimitsMax)
            # ikGoal = self.rightArmKdl.inverse(ps.pose, q_guess=(self.initRightJointGuess if useInitGuess or self.currentRightJointPositions is None else self.currentRightJointPositions), min_joints=self.rightJointLimitsMin, max_joints=self.rightJointLimitsMax)
        else:
            ikGoal = self.leftArmKdl.inverse(ps.pose, q_guess=self.initLeftJointGuess, min_joints=self.leftJointLimitsMin, max_joints=self.leftJointLimitsMax)
            # ikGoal = self.leftArmKdl.inverse(ps.pose, q_guess=(self.initLeftJointGuess if useInitGuess or self.currentLeftJointPositions is None else self.currentLeftJointPositions), min_joints=self.leftJointLimitsMin, max_joints=self.leftJointLimitsMax)
        if ikGoal is not None:
            if not ret:
                self.moveToJointAngles(ikGoal, timeout=timeout, wait=wait, rightArm=rightArm)
            else:
                return ikGoal
        else:
            print 'IK failed'

    def moveToJointAngles(self, jointStates, timeout=1, wait=False, rightArm=True):
        # Create and send trajectory message for new joint angles
        trajMsg = JointTrajectoryGoal()
        trajPoint = JointTrajectoryPoint()
        trajPoint.positions = jointStates
        trajPoint.time_from_start = rospy.Duration(timeout)
        trajMsg.trajectory.points.append(trajPoint)
        trajMsg.trajectory.joint_names = self.rightJointNames if rightArm else self.leftJointNames
        if not wait:
            if rightArm:
                self.rightArmClient.send_goal(trajMsg)
            else:
                self.leftArmClient.send_goal(trajMsg)
        else:
            if rightArm:
                self.rightArmClient.send_goal_and_wait(trajMsg)
            else:
                self.leftArmClient.send_goal_and_wait(trajMsg)

    def lookAt(self, pos, sim=False):
        goal = PointHeadGoal()

        point = PointStamped()
        point.header.frame_id = self.frame
        point.point.x = pos[0]
        point.point.y = pos[1]
        point.point.z = pos[2]
        goal.target = point

        # Point using kinect frame
        goal.pointing_frame = 'head_mount_kinect_rgb_link'
        if sim:
            goal.pointing_frame = 'high_def_frame'
        goal.pointing_axis.x = 1
        goal.pointing_axis.y = 0
        goal.pointing_axis.z = 0
        goal.min_duration = rospy.Duration(1.0)
        goal.max_velocity = 1.0

        self.pointHeadClient.send_goal_and_wait(goal)

    def printJointStates(self):
        try:
            # now = rospy.Time.now()
            # self.tf.waitForTransform(self.frame, 'r_gripper_tool_frame', now, rospy.Duration(10.0))
            self.tf.waitForTransform(self.frame, 'r_gripper_tool_frame', rospy.Time(), rospy.Duration(10.0))
            currentRightPos, currentRightOrient = self.tf.lookupTransform(self.frame, 'r_gripper_tool_frame', rospy.Time(0))

            msg = rospy.wait_for_message('/r_arm_controller/state', JointTrajectoryControllerState)
            currentRightJointPositions = msg.actual.positions
            print 'Right positions:', currentRightPos, currentRightOrient
            print 'Right joint positions:', currentRightJointPositions

            # now = rospy.Time.now()
            # self.tf.waitForTransform(self.frame, 'l_gripper_tool_frame', now, rospy.Duration(10.0))
            currentLeftPos, currentLeftOrient = self.tf.lookupTransform(self.frame, 'l_gripper_tool_frame', rospy.Time(0))

            msg = rospy.wait_for_message('/l_arm_controller/state', JointTrajectoryControllerState)
            currentLeftJointPositions = msg.actual.positions
            print 'Left positions:', currentLeftPos, currentLeftOrient
            print 'Left joint positions:', currentLeftJointPositions

            # print 'Right gripper state:', rospy.wait_for_message('/r_gripper_controller/state', JointControllerState)
        except tf.ExtrapolationException:
            print 'No transformation available! Failing to record this time step.'

    def rightJointsCallback(self, msg):
        self.currentRightJointPositions = msg.actual.positions

    def leftJointsCallback(self, msg):
        self.currentLeftJointPositions = msg.actual.positions



