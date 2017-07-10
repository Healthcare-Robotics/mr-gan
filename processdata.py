import os, sys, glob, time, joblib
import cPickle as pickle
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
# Librosa for audio
import librosa

for durationOfContact, contactAccelLength in zip([4, 3, 2, 1, 0.5, 0.2, 0.1, 4, 4, 4, 4, 4, 4, 4], [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05]):
    windowSize = 100 * durationOfContact
    windowContact = 48000 * contactAccelLength
    windowAccel = 3000 * contactAccelLength
    materials = ['plastic', 'glass', 'fabric', 'metal', 'wood', 'ceramic']
    print '-'*50
    print 'Force/temperature duration:', durationOfContact, '| Contact mic/accel duration:', contactAccelLength
    print '-'*50
    for m, material in enumerate(materials):
        filenames = glob.glob('data_raw/newdata_%s*.pkl' % material)
        allData = dict()
        for filename in filenames:
            objectname = '_'.join(filename.split('/')[-1].split('_')[1:3])
            if objectname not in allData:
                allData[objectname] = dict()
                allData[objectname]['forceTime'] = []
                allData[objectname]['force0'] = []
                allData[objectname]['force1'] = []
                allData[objectname]['pressureTime'] = []
                allData[objectname]['pressure0'] = []
                allData[objectname]['pressure1'] = []
                allData[objectname]['temperatureTime'] = []
                allData[objectname]['temperature'] = []
                allData[objectname]['contactTime'] = []
                allData[objectname]['contact'] = []

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                print 'Processing:', filename
                tt = time.time()

                for temp, tempTime, force, pressure, forceTime, contact, contactTime, accel, accelTime, impactTime in zip(data['temperatureRaw'], data['temperatureTime'], data['RGripRFingerForce'], data['RGripRFingerPressure'], data['RGripRFingerTime'], data['contactmic'], data['contactmicTime'], data['accelerometer'], data['accelerometerTime'], data['collisionTime']):
                    # force: bottom = 3, top = 4
                    temp = np.array(temp)
                    tempTime = np.array(tempTime)
                    force = np.array(force)
                    pressure = np.array(pressure)
                    forceTime = np.array(forceTime)
                    contact = np.array(contact)
                    contactTime = np.array(contactTime)

                    # Determine which two force taxels to use
                    taxelIndex1 = 3
                    taxelIndex2 = 4

                    # Start 0.5 seconds before contact
                    preForceIndex = np.argmax(forceTime > impactTime-0.1)
                    postForceIndex = len(forceTime) if forceTime[-1] <= impactTime+durationOfContact else np.argmax(forceTime > impactTime+durationOfContact)
                    # Interpolate to ensure consistent window size
                    newForceTime = np.linspace(forceTime[preForceIndex], forceTime[postForceIndex-1], num=windowSize, endpoint=True)
                    xForce0 = interp1d(forceTime[preForceIndex:postForceIndex], force[preForceIndex:postForceIndex, taxelIndex1])(newForceTime).tolist()
                    xForce1 = interp1d(forceTime[preForceIndex:postForceIndex], force[preForceIndex:postForceIndex, taxelIndex2])(newForceTime).tolist()
                    allData[objectname]['forceTime'].append(newForceTime)
                    allData[objectname]['force0'].append(xForce0)
                    allData[objectname]['force1'].append(xForce1)
                    xPressure0 = interp1d(forceTime[preForceIndex:postForceIndex], pressure[preForceIndex:postForceIndex, taxelIndex1])(newForceTime).tolist()
                    xPressure1 = interp1d(forceTime[preForceIndex:postForceIndex], pressure[preForceIndex:postForceIndex, taxelIndex2])(newForceTime).tolist()
                    allData[objectname]['pressureTime'].append(newForceTime)
                    allData[objectname]['pressure0'].append(xPressure0)
                    allData[objectname]['pressure1'].append(xPressure1)

                    # Use just the temperature in celcius (index 1), not the raw temperature readings (index 0)
                    preTempIndex = np.argmax(tempTime > impactTime-0.1)
                    postTempIndex = len(tempTime) if tempTime[-1] <= impactTime+durationOfContact else np.argmax(tempTime > impactTime+durationOfContact)
                    newTempTime = np.linspace(tempTime[preTempIndex], tempTime[postTempIndex-1], num=windowSize, endpoint=True)
                    xTemp = interp1d(tempTime[preTempIndex:postTempIndex], temp[preTempIndex:postTempIndex, 1])(newTempTime).tolist()
                    allData[objectname]['temperatureTime'].append(newTempTime)
                    allData[objectname]['temperature'].append(xTemp)

                    preContactIndex = np.argmax(contactTime > impactTime-(contactAccelLength/2.0))
                    postContactIndex = np.argmax(contactTime > impactTime+(contactAccelLength/2.0))
                    # Upsample to 44.1 kHz using librosa (This uses scipy.resample which is worse than interpolate)
                    newContactTime = np.linspace(contactTime[preContactIndex+1], contactTime[postContactIndex-1], num=windowContact, endpoint=True)
                    xContact = interp1d(contactTime[preContactIndex:postContactIndex], contact[preContactIndex:postContactIndex])(newContactTime).tolist()
                    allData[objectname]['contactTime'].append(newContactTime)
                    allData[objectname]['contact'].append(xContact)

                data = None
                print 'Done processing file', time.time() - tt, 's'
                sys.stdout.flush()

        with open('data_processed/custom_processed_0.1sbefore_%s_times_%.2f_%.2f.pkl' % (material, durationOfContact, contactAccelLength), 'wb') as f:
            pickle.dump(allData, f, pickle.HIGHEST_PROTOCOL)


