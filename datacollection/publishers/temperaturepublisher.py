#!/usr/bin/env python

import serial, time, gc
import numpy as np

import rospy
from std_msgs.msg import Float64, Float64MultiArray, String

'''
Run on machine with Teensy connected and master set to PR2
'''

def setupSerial(devName, baudrate):
    serialDev = serial.Serial(port=devName, baudrate=baudrate, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE)
    if serialDev is None:
        raise RuntimeError('[%s]: Serial port %s not found!\n' % (rospy.get_name(), devDame))

    serialDev.write_timeout = .05
    serialDev.timeout= 0.05
    serialDev.flushOutput()
    serialDev.flushInput()
    return serialDev

def getData(serialDev, numOutputs):
    numAttempts = 4
    for i in xrange(numAttempts):
        try:
            line = serialDev.readline()
            try:
                values = map(float, line.split(','))
                if len(values) == numOutputs:
                    return values
                else:
                    print 'Read values did not matched expected outputs:', line
            except ValueError:
                print 'Received suspect data: %s from socket.' % line
        except:
            print 'Unable to read line. Recommended to setup serial again.'
        serialDev.flush()
    return []

state = 'stop'
serialTeensy = None
contact = False
def datastate(msg):
    global state, contact
    if msg.data.lower() == 'contact':
        # Send hold command
        print 'Turning off temperature control due to contact'
        serialTeensy.write('H'.encode('utf-8'))
        contact = True
    else:
        state = msg.data.lower()

if __name__ == '__main__':
    # Setup serial port and ROS publisher
    baudrate = 115200
    serialTeensy = setupSerial('/dev/teensytemperature', baudrate)
    pub = rospy.Publisher('/semihaptics/temperature', Float64MultiArray, queue_size=10000)
    pubCollisionTime = rospy.Publisher('/semihaptics/collisiontime', Float64, queue_size=10)
    rospy.Subscriber('/semihaptics/datastate', String, datastate)
    rospy.init_node('temperaturepublisher')

    # Read a few lines to get things rolling
    for _ in range(25):
        serialTeensy.readline()

    print 'Started publishing ADC data'
    data = []
    times = []

    t = time.time()
    while not rospy.is_shutdown():
        if state == 'zeroing':
            fa = Float64MultiArray()
            fa.data = getData(serialTeensy, 2)
            pub.publish(fa)
        elif state == 'start':
            t = time.time()
            print 'Start recording'
            data = []
            times = []
            state = 'record'
        elif state == 'record':
            data.append(getData(serialTeensy, 2))
            times.append(time.time())
            # If contact occurs, turn off temperature control
            if not contact and len(data) > 10 and abs(data[-1][-1] - np.mean(np.array(data[:10])[:, -1])) > 1:
                print 'Turning off temperature control'
                # Different between most recent temperature and first few temperatures exceeds 1 celcius. Contact has occurred
                serialTeensy.write('H'.encode('utf-8'))
                contact = True
                pubCollisionTime.publish(time.time() - t)
        else:
            if contact:
                # Turn on temperature control
                print 'Turning on temperature control'
                serialTeensy.write('C'.encode('utf-8'))
                contact = False
            if data or times:
                # We have stopped recording. Publish all data
                fa = Float64MultiArray()
                fa.data = (np.array(times) - t).tolist() + np.array(data).flatten().tolist()
                pub.publish(fa)
                data = []
                times = []
                gc.collect()
                print 'Published all data'
            rospy.sleep(0.0001)

    serialTeensy.close()
