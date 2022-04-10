# DeepRaman
A Strong Universal method for Spectral Matching and Compound Identification in Generalized Raman Spectroscopy Based on Deep Learning
----------
Raman spectra contain abundant information from molecules but are difficult to analyze, especially for the mixtures. In this study, a method entitled DeepRaman has been proposed to solve this problems. Essentially, it is a pseudo-Siamese neural network (pSNN) with spatial pyramid pooling (SPP) to predict component(s) in an unknown sample by comparing their Raman spectra. DeepRaman obtains promising results for the analysis of surface-enhanced Raman spectroscopy (SERS) dataset of artificial sweeteners and Raman imaging dataset of gunpowder. Together, it is an accurate, universal and ready-to-use method for spectral matching and compound identification in various application scenarios.

<div align="center">
<img src="https://github.com/XiaqiongFan/DeepRaman/blob/main/workflow.jpg?raw=true" width="70%">
</div>

# Installation

## python and TensorFlow

Python 3.6.8，available at [https://www.python.org.](https://www.python.org/) 

TensorFlow (version 2.0.0-GPU)，available at [https://github.com/tensorflow.](https://github.com/tensorflow) 

## Install dependent packages

The packages mainly include: numpy,Matplotlib,sklearn and os.

These packages are included in the integration tool Anaconda [(https://www.anaconda.com).](https://www.anaconda.com/) 

# Clone the repository and run it directly
[git clone](https://github.com/xiaqiong/DeepRaman) 

**1.Training your model**

Run the file 'training.py'.The corresponding example data have been uploaded to the folder named 'augmented data'.

**2.Predict mixture spectra data**

Run the file 'testing.py'. The input 1 and input 2 represent pure component spectrum and unknown spectrum, respectively.

# Contact

Xiaqiong Fan: xiaqiongfan@csu.edu.cn

