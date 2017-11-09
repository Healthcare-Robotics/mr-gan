# Semi-Supervised Haptic Material Recognition using GANs

Z. Erickson, S. Chernova, and C. C. Kemp, "Semi-Supervised Haptic Material Recognition for Robots using Generative Adversarial Networks", pre-print on [arXiv](https://arxiv.org/abs/1707.02796), 2017.

Project webpage: http://healthcare-robotics.com/mr-gan

## Download the MREO dataset
Compact dataset (1 GB) (can be used to compute tables 1, 2, and 4): https://goo.gl/WiqSjJ  
Full dataset (20 GB) (can be used to compute all tables in paper): https://goo.gl/FnXfgM  
Raw data collected on the PR2: https://goo.gl/DNqPib  
Dataset details can be found on the [project webpage](http://healthcare-robotics.com/mr-gan).

## Running the code
Our generative adversarial network is implemented in Keras and includes the feature matching technique presented by [Salimans et al.](https://arxiv.org/abs/1606.03498v1)  
Results presented in tables 1, 2, and 4 can be recomputed using the command below (requires compact dataset). This takes several hours with a GPU.
```bash
python mr_gan.py --tables 1 2 4
```
Recompute results presented in table 3 (requires full dataset).
```bash
python mr_gan.py --tables 3
```
Generate plots. This requires [plotly](https://plot.ly/python/).
```bash
python paperplotly.py
```
Collect new data with a PR2.
```bash
rosrun fingertip_pressure sensor_info.py &
python contactmicpublisher.py &
python temperaturepublisher.py &
python collectdataPoke.py -n fabric_khakishorts -s 100 -w 0.1 -l 0.1 -ht 0.06 -v
python collectdataPoke.py -n plastic_fullwaterbottle -s 100 -l 0.03 -ht 0.08
```

### Dependencies
Python 2.7  
Keras 2.0.9  
Librosa 0.5.1  
Theano 0.9.0  
Numpy 1.13.3  
Plotly 2.0.11

