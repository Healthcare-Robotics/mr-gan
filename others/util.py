import glob, csv, os
import numpy as np

def loadLuminiDataset():
    data = []
    wavelengths = None
    filenames = glob.glob(os.path.join('data', 'lumini', '*', '*', '*_*_*.txt'))
    for filename in filenames:
        filenameParts = filename.split(os.sep)
        material = filenameParts[-3]
        objectName = filenameParts[-2]
        exposure = int(filenameParts[-1].split('.')[0].split('_')[-1])
        with open(filename, 'rb') as f:
            # Find file line that splits original from sensitivity corrected data
            lines = f.read().splitlines()
            split = np.argmax([1 if 'sensitivity' in l else 0 for l in lines])
            # print split, lines[1:split], lines[split]
            measurements_orig = np.array([l.split('\t') for l in lines[1:split]], dtype=np.float)
            measurements_corrected = np.array([l.split('\t') for l in lines[split+1:]], dtype=np.float)
            if wavelengths is None:
                wavelengths = measurements_orig[:, 0]
            elif not np.array_equal(wavelengths, measurements_orig[:, 0]) or not np.array_equal(wavelengths, measurements_corrected[:, 0]):
                print 'Found a file with inconsistent wavelengths', filename
                exit()
            data.append([material, objectName, exposure] + measurements_orig[:, 1].tolist() + measurements_corrected[:, 1].tolist())
    return data, wavelengths

def processLuminiDataset(data, materialNames, objectNames, sampleCount=20, exposure=100, correctedValues=True):
    X = []
    y = []
    counts = dict()
    for d in data:
        material = d[0]
        obj = d[1]
        exp = d[2]
        if material not in materialNames or exp != exposure:
            continue
        index = materialNames.index(material)
        if obj not in objectNames[index]:
            continue
        values = d[3:]
        if correctedValues:
            # Use only corrected readings
            values = values[:len(values)/2]
        else:
            # Use only original signal readings
            values = values[len(values)/2:]
        if material + obj not in counts:
            counts[material+obj] = 0
        if counts[material+obj] < sampleCount:
            X.append(values)
            y.append(index)
            counts[material+obj] += 1
    return X, y

def firstDeriv(x, wavelengths):
    # First derivative of measurements with respect to wavelength
    x = np.copy(x)
    for i, xx in enumerate(x):
        dx = np.zeros(xx.shape, np.float)
        dx[0:-1] = np.diff(xx)/np.diff(wavelengths)
        dx[-1] = (xx[-1] - xx[-2])/(wavelengths[-1] - wavelengths[-2])
        x[i] = dx
    return x


