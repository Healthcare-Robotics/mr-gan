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

def getData(serialDev):
    numAttempts = 4
    for i in xrange(numAttempts):
        try:
            value = float(serialDev.readline())
            if value < 10000:
                return value
        except:
            print 'Unable to read line. Recommended to setup serial again.'
        serialDev.flush()
    return []

state = 'stop'
def datastate(msg):
    global state
    if msg.data.lower() != 'contact':
        state = msg.data.lower()

if __name__ == '__main__':
    # Setup serial port and ROS publisher
    baudrate = 115200
    serialTeensy = setupSerial('/dev/teensycontactmic', baudrate)
    pub = rospy.Publisher('/semihaptics/contactmic', Float64MultiArray, queue_size=10000)
    rospy.Subscriber('/semihaptics/datastate', String, datastate)
    rospy.init_node('contactmicpublisher')

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
            fa.data = [getData(serialTeensy)]
            pub.publish(fa)
        elif state == 'start':
            t = time.time()
            print 'Start recording'
            data = []
            times = []
            state = 'record'
        elif state == 'record':
            data.append(getData(serialTeensy))
            times.append(time.time())
        else:
            if data or times:
                # We have stopped recording. Publish all data
                fa = Float64MultiArray()
                fa.data = (np.array(times) - t).tolist() + data
                pub.publish(fa)
                data = []
                times = []
                gc.collect()
                print 'Published all data'
            rospy.sleep(0.0001)

    serialTeensy.close()
