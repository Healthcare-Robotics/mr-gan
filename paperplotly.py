import os, sys, inspect, plotly
import plotly.graph_objs as go
import numpy as np
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
sys.path.insert(0, parentdir)
import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import librosa
import librosa.display

if not os.path.isdir('plots'):
    os.makedirs('plots')

x = [1, 2, 4, 8, 16, 50, 100]
force = [62.1, 70.4, 72.2, 77.7, 79.8, 85.8, 87.9]
temp = [53.8, 59.0, 64.1, 68.1, 69.0, 80.0, 82.1]
contact = [42.9, 53.9, 62.6, 67.5, 73.4, 79.8, 83.1]
forcetemp = [74.3, 81.4, 85.6, 88.5, 90.2, 94.2, 95.3]
forcecontact = [58.2, 67.5, 73.8, 80.2, 84.7, 89.7, 91.8]
tempcontact = [52.4, 68.3, 79.2, 84.9, 87.4, 91.2, 92.2]
forcetempcontact = [62.8, 75.4, 85.6, 89.4, 92.0, 95.4, 96.2]

data = []
data.append(go.Scatter(x=x, y=force, name='Force', line=dict(dash='lines'), mode='lines'))
data.append(go.Scatter(x=x, y=temp, name='Temperature', line=dict(dash='lines'), mode='lines'))
data.append(go.Scatter(x=x, y=contact, name='Contact mic', line=dict(dash='lines'), mode='lines'))
data.append(go.Scatter(x=x, y=forcetemp, name='Force, Temperature', line=dict(dash='lines'), mode='lines'))
data.append(go.Scatter(x=x, y=forcecontact, name='Force, Contact mic', line=dict(dash='lines'), mode='lines'))
data.append(go.Scatter(x=x, y=tempcontact, name='Temperature, Contact mic', line=dict(dash='lines'), mode='lines'))
data.append(go.Scatter(x=x, y=forcetempcontact, name='Force, Temperature, Contact mic', line=dict(dash='lines'), mode='lines'))

layout = dict(title='Accuracy with Varying Labeled Training Data',
              titlefont=dict(size=20),
              xaxis=dict(title='Percent of Training Data Labeled (%)', showgrid=True, titlefont=dict(size=18), tickfont=dict(size=18)),
              yaxis=dict(title='Accuracy (%)', showgrid=True, titlefont=dict(size=18), tickfont=dict(size=18)),
              width=1280,
              height=720,
              legend=dict(font=dict(size=18)),
              showlegend=True)

plotly.offline.plot({'data': data, 'layout': layout}, filename='plots/table1.html')

# ---------------------------------------

forcetempTime = 4
contactmicTime = 0.2
materials = ['plastic', 'glass', 'fabric', 'metal', 'wood', 'ceramic']
objects = ['coffee', 'candle', 'pillowcase', 'drinkingmug', 'soapdispenser', 'largemug']
dataForce = []
dataTemperature = []
dataContactmic = []
for material, objName in zip(materials, objects):
    with open('data_processed/processed_0.1sbefore_%s_times_%.2f_%.2f.pkl' % (material, forcetempTime, contactmicTime), 'rb') as f:
        allData = pickle.load(f)
        allData['accel0'] = None; allData['accel1'] = None; allData['accel2'] = None; allData['accelTime'] = None
        objData = allData['%s_%s' % (material, objName)]
        dataForce.append(go.Scatter(x=(objData['forceTime'][5] - objData['forceTime'][5][0]), y=objData['force0'][5], name=material, line=dict(dash='lines'), mode='lines'))
        dataTemperature.append(go.Scatter(x=(objData['temperatureTime'][5] - objData['temperatureTime'][5][0]), y=objData['temperature'][5], name=material, line=dict(dash='lines'), mode='lines'))
        # NOTE: Convert contact mic signal to voltage by dividing by 2048 then multiplying by 5 (volts).
        dataContactmic.append(go.Scatter(x=(objData['contactTime'][5] - objData['contactTime'][5][0] + 0.4), y=np.array(objData['contact'][5])/2048.0*5.0, name=material, line=dict(dash='lines', width=1), mode='lines'))
        allData = None

layout = dict(title='Force Measurements',
              titlefont=dict(size=20),
              xaxis=dict(title='Time (s)', showgrid=True, titlefont=dict(size=20), tickfont=dict(size=20)),
              yaxis=dict(title='Force (N)', showgrid=True, titlefont=dict(size=20), tickfont=dict(size=20)),
              width=500,
              height=350,
              legend=dict(font=dict(size=18)),
              showlegend=False)
plotly.offline.plot({'data': dataForce, 'layout': layout}, filename='plots/force_measurements.html')

layout = dict(title='Temperature Measurements',
              titlefont=dict(size=20),
              xaxis=dict(title='Time (s)', showgrid=True, titlefont=dict(size=20), tickfont=dict(size=20)),
              yaxis=dict(title='Temperature (C)', showgrid=True, titlefont=dict(size=20), tickfont=dict(size=20)),
              width=500,
              height=350,
              legend=dict(font=dict(size=14)),
              showlegend=True)
plotly.offline.plot({'data': dataTemperature, 'layout': layout}, filename='plots/temperature_measurements.html')

layout = dict(title='Contact Microphone Measurements',
              titlefont=dict(size=20),
              xaxis=dict(title='Time (s)', showgrid=True, titlefont=dict(size=20), tickfont=dict(size=20)),
              yaxis=dict(title='Contact Mic Signal (V)', showgrid=True, titlefont=dict(size=20), tickfont=dict(size=20)),
              width=500,
              height=350,
              legend=dict(font=dict(size=18)),
              showlegend=False)
plotly.offline.plot({'data': dataContactmic, 'layout': layout}, filename='plots/contactmic_measurements.html')

# ---------------------------------------

def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []
    for k in range(pl_entries):
        C = map(np.uint8, np.array(cmap(k*h)[:3])*255)
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])
    return pl_colorscale

magma_cmap = matplotlib.cm.get_cmap('magma')
norm = matplotlib.colors.Normalize(vmin=0, vmax=255)
magma_rgb = [matplotlib.colors.colorConverter.to_rgb(magma_cmap(norm(i))) for i in xrange(0, 255)]
magma = matplotlib_to_plotly(magma_cmap, 255)

forcetempTime = 4
contactmicTime = 0.2
plotsPerRow = 4
# TODO: 3 objects from each material
materials = ['plastic', 'plastic', 'glass', 'glass', 'fabric', 'fabric', 'metal', 'metal', 'wood', 'wood', 'ceramic', 'ceramic']
objects = ['coffee', 'bowl', 'plate', 'bowl', 'jeans', 'woolsocks', 'drinkingmug', 'bowl', 'cuttingboard', 'largebambooplate', 'whiteplate', 'largemug']
names = ['Plastic Coffee Container', 'Plastic Bowl', 'Glass Plate', 'Glass Bowl', 'Fabric Jeans', 'Fabric Wool Socks', 'Metal Coffee Mug', 'Metal Bowl', 'Wood Cutting Board', 'Bamboo Plate', 'Ceramic Plate', 'Ceramic Mug']

plots = []
plotNames = []
colors = ['rgb(53, 118, 203)', 'rgb(120, 91, 188)']
for i, (material, objName, name) in enumerate(zip(materials, objects, names)):
    with open('data_processed/processed_0.1sbefore_%s_times_%.2f_%.2f.pkl' % (material, forcetempTime, contactmicTime), 'rb') as f:
        allData = pickle.load(f)
        allData['accel0'] = None; allData['accel1'] = None; allData['accel2'] = None; allData['accelTime'] = None
        objData = allData['%s_%s' % (material, objName)]
        plotNames.append(name)

        # NOTE: Convert contact mic signal to voltage by dividing by 2048 then multiplying by 5 (volts).
        dataContactmic = go.Scatter(x=(objData['contactTime'][0] - objData['contactTime'][0][0]), y=np.array(objData['contact'][0])/2048.0*5.0, line=dict(dash='lines', width=2, color=colors[(i%4)/2]), mode='lines')
        plots.append(dataContactmic)


        # Mel-scale spectrogram
        sr = 48000
        S = librosa.feature.melspectrogram(np.array(objData['contact'][0]), sr=sr, n_mels=128)
        # Convert to log scale (dB)
        log_S = librosa.logamplitude(S, ref_power=np.max)
        print np.shape(S), np.shape(log_S)

        allParams = dict(sr=sr, fmin=None, fmax=None, bins_per_octave=12, hop_length=512)
        xCoords = librosa.display.__mesh_coords('time', None, log_S.shape[1], **allParams)
        yCoords = librosa.display.__mesh_coords('log', None, log_S.shape[0], **allParams)
        if i%plotsPerRow == plotsPerRow-1:
            dataSpectrogram = go.Heatmap(z=log_S, x=xCoords, y=yCoords, colorscale=magma, colorbar=dict(tickformat='+.0f', ticksuffix=' dB', tickfont=dict(size=20)), zmin=-60, zmax=0)
        else:
            dataSpectrogram = go.Heatmap(z=log_S, x=xCoords, y=yCoords, colorscale=magma, showscale=False)
        plots.append(dataSpectrogram)

        if i%plotsPerRow == plotsPerRow-1:
            yCoordsMel = librosa.display.__mesh_coords('mel', None, log_S.shape[0], **allParams)
            print [yCoords.min(), yCoords.max()], [yCoordsMel.min(), yCoordsMel.max()]
            text = np.array([0, 512, 1024, 2048, 4096, 8192])
            steps = np.arange(6) / (np.log2(yCoordsMel.max() - yCoordsMel.min()) - 7.5)
            vals = steps * (yCoords.max() - yCoords.min())
            text = np.array(['0', '.5', '1', '2', '4', '8'])

            fig = plotly.tools.make_subplots(rows=2, cols=plotsPerRow, subplot_titles=plotNames, shared_xaxes=False, shared_yaxes=True, vertical_spacing=0.03, horizontal_spacing=0.02)
            for j, plot in enumerate(plots):
                fig.append_trace(plot, j%2+1, j/2+1)

            fig['layout'].update(width=500*plotsPerRow, height=350*2, titlefont=dict(size=20), showlegend=False)
            for j in xrange(plotsPerRow):
                fig['layout']['xaxis'+str(1+j)].update(showticklabels=False, showgrid=True, zeroline=False, range=[0, 0.2])
                fig['layout']['xaxis'+str(plotsPerRow+1+j)].update(showticklabels=False, range=[0, 0.2])
            fig['layout']['xaxis'+str(plotsPerRow+1)].update(title='Time (s)', titlefont=dict(size=20), showticklabels=True, tickfont=dict(size=20))
            fig['layout']['yaxis1'].update(title='Signal (V)', titlefont=dict(size=20), tickfont=dict(size=20), showgrid=True, range=[-5.1, 5.1], dtick=5, zeroline=False)
            fig['layout']['yaxis2'].update(title='Frequency (kHz)', titlefont=dict(size=20), tickfont=dict(size=20), tickmode='array', ticktext=text, tickvals=vals)
            plotly.offline.plot(fig, filename='plots/signal_spectrogram_%dplots_%d.html' % (plotsPerRow, i))
            plots = []
            plotNames = []

        allData = None


