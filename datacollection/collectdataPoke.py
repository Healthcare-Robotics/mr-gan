import os, sys, roslib, rospy, argparse, cv2, select, threading, serial, time

import numpy as np
import cPickle as pickle
from PIL import Image as PILImage
from sensor_msgs.msg import Image
from fingertip_pressure.msg import PressureInfo
from cv_bridge import CvBridge, CvBridgeError
from pr2_msgs.msg import PressureState, AccelerometerState
from std_msgs.msg import Float64, Float64MultiArray, String
import matplotlib.pyplot as plt

import control.controller as controller

class CollectData:
    def __init__(self, objectName, datasubscribe=False, sequencesPerObject=25, flat=False, quarterflat=False, verticalMovement=False, rotateonce=False, neverrotate=False, curvedsurface=False, width=0, length=0, height=0, heightOffset=0, initWidth=0, handle=False, startCount=0):
        self.objectName = objectName
        self.datasubscribe = datasubscribe
        self.sequencesPerObject = sequencesPerObject
        self.startCount = startCount
        self.bridge = CvBridge()
        self.hz = 1000
        self.saveBatchSize = 25
        self.zeroing = False
        self.recording = False
        self.waitingForData = False
        self.reheating = False
        self.flat = flat
        self.quarterflat = quarterflat
        self.verticalMovement = verticalMovement
        self.rotateonce = rotateonce
        self.neverrotate = neverrotate
        self.curvedsurface = curvedsurface
        self.handle = handle
        self.width= width
        self.length = length
        self.height = height

        self.control = controller.Controller('torso_lift_link', self.verticalMovement)
        # controller.printJointStates()

        # Starting gripper positions for when robot is holding a platter
        # [+ away from robot, + left (robot's perspective) towards center of robot, + up towards the sky]
        if not self.verticalMovement:
            self.initRightPos = np.array([0.495, -0.1 - self.length, 0.08 + heightOffset])
            self.initRightRPY = np.array([0.0, 0.0, np.pi/2.0])
            self.initLeftPos = np.array([0.5, 0.0, 0.0])
            self.initLeftRPY = np.array([0.0, -np.pi/2.0, np.pi/2.0])
            self.initLeftJoints = np.array([0.192, 0.932, 1.052, -1.62, 11.991, -1.225, 2.548])
        else:
            self.initRightPos = np.array([0.5 + initWidth, -0.12, 0.02 + self.height])
            self.initRightRPY = np.array([0.0, np.pi/2.0, np.pi/2.0])
            self.initLeftPos = np.array([0.5, -0.1, -0.1])
            self.initLeftRPY = np.array([0.0, -np.pi/2.0, np.pi/2.0])

        # Data variables
        self.resetData()

        self.RGripRFingerForce = None
        self.RGripRFingerPressure = None
        self.RGripRFingerForceMean = None
        self.RGripRFingerForceRecent = []

        self.accelMean = None
        self.accelRecent = []

        self.temperatureMean = None
        self.temperatureRecent = []
        self.temperatureDataReceived = False
        self.temperatureReheat = 0.0
        self.collisionTimeTemp = 10000
        self.collisionTimeForce = 10000

        self.contactmicMean = None
        self.contactmicRecent = []
        self.contactmicDataReceived = False

        # Data subscribers
        if self.datasubscribe:
            # Gripper Force
            rospy.Subscriber('/pressure/r_gripper_motor', PressureState, self.rGripperForceCallback)
            # Get sensor information for right finger on the right gripper
            print 'Waiting for gripper pressure info'
            msg = rospy.wait_for_message('/pressure/r_gripper_motor_info', PressureInfo)
            self.forcePerUnit = np.array(msg.sensor[1].force_per_unit)
            # Calculate surface areas for each taxel to convert into kilopascals
            # Originates from /opt/ros/indigo/lib/python2.7/dist-packages/fingertip_pressure/fingertip_panel.py
            halfside1 = np.array([[v.x, v.y, v.z] for v in msg.sensor[1].halfside1])
            halfside2 = np.array([[v.x, v.y, v.z] for v in msg.sensor[1].halfside2])
            quarterarea = np.linalg.norm(halfside1[:, [1, 2, 0]] * halfside2[:, [2, 0, 1]] - halfside1[:, [2, 0, 1]] * halfside2[:, [1, 2, 0]], axis=1)
            self.tactileAreas = quarterarea * 4

            # Gripper Accelerometer
            rospy.Subscriber('/accelerometer/r_gripper_motor', AccelerometerState, self.accelerometerCallback)

            # Temperature Sensors
            rospy.Subscriber('/semihaptics/temperature', Float64MultiArray, self.temperatureCallback)
            rospy.Subscriber('/semihaptics/contactmic', Float64MultiArray, self.contactmicCallback)
            self.statePublisher = rospy.Publisher('/semihaptics/datastate', String, queue_size=10)
            rospy.Subscriber('/semihaptics/collisiontime', Float64, self.collisionTimeCallback)

            print 'All subscribers set up'

    def resetData(self):
        # Collect all measurements, regardless of sampling frequency
        self.dataAll = {'objectImage': None, 'images': [], 'RGripRFingerTime': [], 'RGripRFingerForceRaw': [], 'RGripRFingerForce': [], 'RGripRFingerPressure': [], 'RGripRFingerPressureRaw': [], 'temperatureTime': [], 'temperatureRaw': [], 'temperature': [], 'accelerometerTime': [], 'accelerometerRaw': [], 'accelerometer': [], 'contactmicTime': [], 'contactmicRaw': [], 'contactmic': [], 'collisionTime': []}

    def collisionTimeCallback(self, msg):
        self.collisionTimeTemp = msg.data

    def rGripperForceCallback(self, msg):
        RGripRFingerForceRaw = np.array(msg.r_finger_tip)
        if self.zeroing and self.RGripRFingerForceMean is None:
            # Determine a mean for readings to zero out data
            self.RGripRFingerForceRecent.append(RGripRFingerForceRaw)
            if len(self.RGripRFingerForceRecent) >= 20:
                self.RGripRFingerForceMean = np.mean(self.RGripRFingerForceRecent, axis=0)
        elif self.RGripRFingerForceMean is not None:
            # Process the raw electronic readings into newton force and kilopascal pressure readings
            # Originates from /opt/ros/indigo/lib/python2.7/dist-packages/fingertip_pressure/fingertip_panel.py
            self.RGripRFingerForce = (RGripRFingerForceRaw - self.RGripRFingerForceMean) / self.forcePerUnit
            self.RGripRFingerPressure = self.RGripRFingerForce / self.tactileAreas / 1000.0
            RGripRFingerPressureRaw = (RGripRFingerForceRaw / self.forcePerUnit) / self.tactileAreas / 1000.0
            if self.recording:
                # Record data
                self.dataAll['RGripRFingerTime'][-1].append(rospy.get_time() - self.startTime)
                self.dataAll['RGripRFingerForceRaw'][-1].append(RGripRFingerForceRaw)
                self.dataAll['RGripRFingerForce'][-1].append(self.RGripRFingerForce)
                self.dataAll['RGripRFingerPressure'][-1].append(self.RGripRFingerPressure)
                self.dataAll['RGripRFingerPressureRaw'][-1].append(RGripRFingerPressureRaw)

    def accelerometerCallback(self, msg):
        accelAllRaw = [[s.x, s.y, s.z] for s in msg.samples]
        accelRaw = np.mean(accelAllRaw, axis=0)
        if self.zeroing and self.accelMean is None:
            # Determine a mean for readings to zero out data
            self.accelRecent.append(accelRaw)
            if len(self.accelRecent) >= 20:
                self.accelMean = np.mean(self.accelRecent, axis=0)
        elif self.recording and self.accelMean is not None:
            # Record data
            self.dataAll['accelerometerTime'][-1].extend([rospy.get_time() - self.startTime]*len(accelAllRaw))
            self.dataAll['accelerometerRaw'][-1].extend(accelAllRaw)
            self.dataAll['accelerometer'][-1].extend([np.array(a) - self.accelMean for a in accelAllRaw])

    def contactmicCallback(self, msg):
        if self.zeroing and self.contactmicMean is None:
            # Determine a mean for readings to zero out data
            self.contactmicRecent.append(msg.data[0])
            if len(self.contactmicRecent) >= 20:
                self.contactmicMean = np.mean(self.contactmicRecent)
        elif self.waitingForData:
            # Record data
            print 'All contact mic data:', len(msg.data)
            half = len(msg.data)/2
            self.dataAll['contactmicTime'].append(msg.data[:half])
            self.dataAll['contactmicRaw'].append(msg.data[half:])
            self.dataAll['contactmic'].append(msg.data[half:] - self.contactmicMean)
            self.contactmicDataReceived = True

    def temperatureCallback(self, msg):
        if self.zeroing and self.temperatureMean is None:
            # Determine a mean for readings to zero out data
            self.temperatureRecent.append(msg.data)
            if len(self.temperatureRecent) >= 20:
                self.temperatureMean = np.mean(self.temperatureRecent, axis=0)
        elif self.waitingForData:
            # Record data
            third = len(msg.data)/3
            self.dataAll['temperatureTime'].append(msg.data[:third])
            self.dataAll['temperatureRaw'].append(np.reshape(msg.data[third:], (third, 2)))
            self.dataAll['temperature'].append(np.reshape(msg.data[third:], (third, 2)) - self.temperatureMean)
            self.temperatureDataReceived = True
        elif self.reheating:
            # Grab the current temperature in celcius
            self.temperatureReheat = msg.data[-1]

    def grabImage(self, sim=False, viz=False):
        # Grab image from Kinect sensor
        msg = rospy.wait_for_message('/semihaptics/image' if not sim else '/wide_stereo/left/image_color', Image)
        try:
            image = self.bridge.imgmsg_to_cv2(msg)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if viz:
                PILImage.fromarray(np.uint8(image)).show()
            return image
        except CvBridgeError, e:
            print e
            return None

    def zeroData(self):
        self.RGripRFingerForceMean = None
        self.RGripRFingerForceRecent = []
        self.accelMean = None
        self.accelRecent = []
        self.temperatureMean = None
        self.temperatureRecent = []
        self.contactmicMean = None
        self.contactmicRecent = []
        self.zeroing = True
        self.statePublisher.publish('zeroing')
        while self.RGripRFingerForceMean is None or self.accelMean is None or self.temperatureMean is None or self.contactmicMean is None:
            rospy.sleep(0.01)
        self.statePublisher.publish('stop')
        self.zeroing = False
        print 'Data zeroed'

    def beginNewDataSequence(self):
        # Add an empty list to store data for this sequence
        for key, value in self.dataAll.iteritems():
            if 'RGrip' in key or 'accel' in key:
                value.append([])
        self.collisionTimeTemp = 10000
        self.collisionTimeForce = 10000
        # Zero all data readings
        self.zeroData()

    def saveData(self, directory='../data_raw', iteration=-1, batch=-1):
        # Define filename and directory
        filename = os.path.join(directory, 'newdata_%s_%dseqs%s%s' % (self.objectName, self.sequencesPerObject, '_' + str(iteration) if iteration >= 0 else '', '_batchof%d_%d' % (self.saveBatchSize, batch) if batch >= 0 else ''))
        if not os.path.isdir(directory):
            os.makedirs(directory)
        # Save data
        if iteration < 0:
            # Save all data sequences
            data = self.dataAll
        else:
            # Save only the data from the most recent grasping sequence
            data = {key: value[-1] for key, value in self.dataAll.iteritems() if key not in ['objectImage', 'images']}
        with open(filename + '.pkl', 'wb') as f:
            f.write(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))

    def reheat(self):
        self.temperatureReheat = 0.0
        self.reheating = True
        self.statePublisher.publish('zeroing')
        print 'Waiting for temperature sensor to reheat'
        while abs(self.temperatureReheat - 55.0) > 0.5:
            # Temperature sensor still reheating, continue to wait
            rospy.sleep(0.5)
        self.statePublisher.publish('stop')
        self.reheating = False
        print 'Temperature sensor reheated to:', self.temperatureReheat

    def performInteraction(self):
        rospy.sleep(0.5)
        print '-'*15, 'Beginning grasps', '-'*15
        # Close right gripper and open left gripper
        self.control.grip(openGripper=False, rightArm=True, miniOpen=True)
        answer = raw_input('Open left gripper? [y/n] ')
        if answer == 'y':
            self.control.grip(openGripper=True, rightArm=False)

        startPos = np.copy(self.initRightPos)

        # Zero all data readings
        if self.datasubscribe:
            self.zeroData()
            print 'Data zeroed'

        # Move head and Kinect to point at the start location
        self.control.lookAt(self.initLeftPos)

        # Move to starting locations
        print 'Grippers moving to starting positions'
        self.control.moveGripperTo(startPos, self.initRightRPY, timeout=4.0, useInitGuess=True, wait=False, rightArm=True)
        self.control.moveGripperTo(self.initLeftPos, self.initLeftRPY, timeout=4.0, useInitGuess=True, wait=True, rightArm=False)
        rospy.sleep(1.0)
        self.control.initJoints()
        self.control.printJointStates()

        if self.handle:
            self.control.rotateGripperWrist(np.pi/2.0)

        # Save an image of this object
        # self.dataAll['objectImage'] = self.grabImage(sim=sim, viz=False)
        # print 'Image taken'

        if answer == 'y':
            raw_input('Press enter to close the left gripper')
            self.control.grip(openGripper=False, maxEffort=100.0, rightArm=False)
            rospy.sleep(4.0)

        # Sleep until the temperature sensor is heated up
        if self.datasubscribe:
            self.reheat()

        # Rotate if start count is not zero
        if self.startCount != 0 and not self.neverrotate:
            # Rotate left gripper by a small amount
            if (self.flat or self.rotateonce) and self.startCount == int(self.sequencesPerObject/2.0):
                self.control.rotateGripperWrist(np.pi)
            elif self.quarterflat and self.startCount >= int(self.sequencesPerObject/4.0):
                self.control.rotateGripperWrist(np.pi/2.0 * (self.startCount / int(self.sequencesPerObject/4.0)))
            elif not self.flat and not self.quarterflat and not self.rotateonce:
                self.control.rotateGripperWrist(2*np.pi/self.sequencesPerObject*self.startCount if not self.handle else (2*np.pi - np.pi/2.0)/self.sequencesPerObject*self.startCount)

        print 'Press enter at any point to pause the program'

        for i in xrange(self.startCount, self.sequencesPerObject):
            # Check if any key has been pressed. If so, pause the program
            ii, oo, ee = select.select([sys.stdin], [], [], 0.0001)
            if ii:
                sys.stdin.readline().strip()
                raw_input('Program paused. Press enter to continue')

            if self.datasubscribe:
                self.beginNewDataSequence()
            self.rightPos, self.rightOrient = self.control.getGripperPosition(rightArm=True)

            velocities = []
            prevPos = self.rightPos[1]
            rate = rospy.Rate(self.hz)

            self.startTime = rospy.get_time()
            endCriteria = lambda index, distance: (not self.datasubscribe and abs(self.rightPos[index] - self.initLeftPos[index]) < 0.05) or (self.datasubscribe and (self.RGripRFingerForce[3] > 1 or self.RGripRFingerForce[4] > 1 or abs(self.rightPos[index] - self.initLeftPos[index]) < distance)) or self.collisionTimeTemp != 10000 or rospy.get_time() - self.startTime > 7

            # Begin robot gripper motion
            motiontime = np.random.uniform(1.5, 2.5)
            self.control.moveGripperTo(startPos + (np.array([0, 0.1+self.length, 0]) if not self.verticalMovement else np.array([0, 0, -0.1-self.height])), self.initRightRPY, timeout=motiontime, wait=False, rightArm=True)

            self.startTime = rospy.get_time()
            if self.datasubscribe:
                self.recording = True
                self.statePublisher.publish('start')
            lastTime = rospy.get_time()
            index = 1 if not self.verticalMovement else 2
            prevPos = self.rightPos[index]
            while True:
                self.rightPos, self.rightOrient = self.control.getGripperPosition(rightArm=True)
                if endCriteria(1, 0.04) if not self.verticalMovement else endCriteria(2, 0.04):
                    if not self.verticalMovement:
                        self.control.moveGripperTo(self.rightPos + np.array([0, 0.01, 0]), self.initRightRPY, timeout=0.5, wait=False, rightArm=True)
                    else:
                        self.control.moveGripperTo(self.rightPos + np.array([0, 0, -0.01]), self.initRightRPY, timeout=0.5, wait=False, rightArm=True)
                    break
                rate.sleep()

            # Contact has occurred. Turn off temperature control
            if self.datasubscribe:
                self.statePublisher.publish('contact')
            self.collisionTimeForce = rospy.get_time() - self.startTime

            # Wait for 4 seconds of contact
            graspEndTime = rospy.get_time() + 4.0
            while rospy.get_time() < graspEndTime:
                rate.sleep()

            # Send stop code
            if self.datasubscribe:
                self.waitingForData = True
                self.statePublisher.publish('stop')
                self.recording = False
                # Wait to receive temperature and contact mic data
                while not self.contactmicDataReceived or not self.temperatureDataReceived:
                    rospy.sleep(0.001)
                self.contactmicDataReceived = False
                self.temperatureDataReceived = False
                self.waitingForData = False

            # Determine collision time
            self.dataAll['collisionTime'].append(min(self.collisionTimeForce, self.collisionTimeTemp))
            print 'Collision times:', (self.collisionTimeForce, self.collisionTimeTemp)

            # Store an image for this object during interaction
            # self.dataAll['images'].append(self.grabImage(sim=sim, viz=False))

            if self.datasubscribe:
                # Print out data statistics
                print 'Iteration %d collected' % i
                print 'Contact mic frequency: %.3f Hz, Temperature frequency: %.3f Hz' % (len(self.dataAll['contactmicTime'][-1]) / self.dataAll['contactmicTime'][-1][-1], len(self.dataAll['temperatureTime'][-1]) / self.dataAll['temperatureTime'][-1][-1])

            # Move back to initial position
            if self.verticalMovement:
                startPos = np.copy(self.initRightPos) + np.array([np.random.uniform(-self.width/2.0, self.width/2.0) if not self.curvedsurface else np.random.uniform(0, self.width), np.random.uniform(-self.length/2.0, self.length/2.0), 0])
            elif self.flat or self.quarterflat:
                startPos = np.copy(self.initRightPos) + np.array([np.random.uniform(-self.width/2.0, self.width/2.0), 0, np.random.uniform(-0.01, self.height - 0.01)])
            elif self.height > 0:
                startPos = np.copy(self.initRightPos) + np.array([0, 0, np.random.uniform(-0.01, self.height - 0.01)])
            self.control.moveGripperTo(startPos, self.initRightRPY, timeout=1.5, wait=False, rightArm=True)
            rospy.sleep(0.75)
            if not self.neverrotate:
                # Rotate left gripper by a small amount
                if (self.flat or self.rotateonce) and i == int(self.sequencesPerObject/2.0) - 1:
                    self.control.rotateGripperWrist(np.pi)
                elif self.quarterflat and (i+1) % int(self.sequencesPerObject/4.0) == 0:
                    self.control.rotateGripperWrist(np.pi/2.0)
                elif not self.flat and not self.quarterflat and not self.rotateonce:
                    self.control.rotateGripperWrist(2*np.pi/self.sequencesPerObject if not self.handle else (2*np.pi - np.pi/2.0)/self.sequencesPerObject)

            # Save a batch of data every few iterations and empty dictionary to restore memory
            if self.datasubscribe and (i+1) % self.saveBatchSize == 0:
                self.saveData(directory='../data_raw', batch=((i+1)/self.saveBatchSize))
                self.resetData()
                print 'Batch %d saved' % ((i+1)/self.saveBatchSize)

            # Sleep until the temperature sensor is heated up
            if self.datasubscribe:
                self.reheat()
            else:
                rospy.sleep(5.0)

        # Save data to a pickle file
        if self.datasubscribe and (i+1) % self.saveBatchSize != 0:
            self.saveData(directory='../data_raw', batch=((i+1)/self.saveBatchSize))

if __name__ == '__main__':
    # Run: 'python publishimage.py' on PR2 and 'python temperaturepublisher.py' and 'fingerpressureinfo' on desktop
    rospy.init_node('collectdata')

    parser = argparse.ArgumentParser(description='Collecting data from a spinning platter of objects.')
    parser.add_argument('-n', '--name', help='Object name', required=True)
    parser.add_argument('-s', '--seqs', help='Data collection sequences (pokes) per objects', type=int, required=True)
    parser.add_argument('-f', '--flat', help='Is the object flat?', action='store_true')
    parser.add_argument('-qf', '--quarterflat', help='Is the object flat with four sides?', action='store_true')
    parser.add_argument('-v', '--vertmove', help='Should the robot perform vertical touching movements?', action='store_true')
    parser.add_argument('-ro', '--rotateonce', help='Should the robot only rotate the object 180 deg once?', action='store_true')
    parser.add_argument('-nr', '--neverrotate', help='Should the robot never rotate the object?', action='store_true')
    parser.add_argument('-cs', '--curvedsurface', help='Does the object have a curved inner surface during vertical movement?', action='store_true')
    parser.add_argument('-w', '--width', help='Width of the object', type=float, default=0.0)
    parser.add_argument('-l', '--length', help='Length of the object (for vertical movement)', type=float, default=0.0)
    parser.add_argument('-ht', '--height', help='Height of the object', type=float, default=0.0)
    parser.add_argument('-hto', '--heightoffset', help='Height offset for interaction with the object', type=float, default=0.0)
    parser.add_argument('-iw', '--initwidth', help='Initial width starting location for the end effector', type=float, default=0.0)
    parser.add_argument('-sc', '--startcount', help='Adjusts the Starting iteration number.', type=int, default=0)
    parser.add_argument('-sim', '--simulation', help='Simulation mode?', action='store_true')
    parser.add_argument('-hndl', '--handle', help='Does the object have a handle? If so, If so, orient it to face the right grippers left finger.', action='store_true')
    args = parser.parse_args()

    print args.name, args.seqs, args.flat, args.quarterflat, args.vertmove, args.rotateonce, args.neverrotate, args.curvedsurface, args.width, args.length, args.height, args.initwidth, args.heightoffset, args.simulation, args.handle, args.startcount

    collect = CollectData(args.name, datasubscribe=not args.simulation, sequencesPerObject=args.seqs, flat=args.flat, quarterflat=args.quarterflat, verticalMovement=args.vertmove, rotateonce=args.rotateonce, neverrotate=args.neverrotate, curvedsurface=args.curvedsurface, width=args.width, length=args.length, height=args.height, heightOffset=args.heightoffset, initWidth=args.initwidth, handle=args.handle, startCount=args.startcount)
    collect.performInteraction()

