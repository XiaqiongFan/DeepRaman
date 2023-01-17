# DeepRaman
A Universal and Accurate Method for Easily Component identification in Raman Spectroscopy Based on Deep Learning
----------
Raman spectra contain abundant information from molecules but are difficult to analyze, especially for the mixtures. In this study, a method entitled DeepRaman has been proposed to solve this problems. Essentially, it is a pseudo-Siamese neural network (pSNN) with spatial pyramid pooling (SPP) to predict component(s) in an unknown sample by comparing their Raman spectra. DeepRaman obtains promising results for the analysis of surface-enhanced Raman spectroscopy (SERS) dataset of artificial sweeteners and Raman imaging dataset of gunpowder. Together, it is an accurate, universal and ready-to-use method for spectral matching and compound identification in various application scenarios.

<div align="center">
<img src=https://github.com/XiaqiongFan/DeepRaman/blob/main/WF.png?raw=true" width="70%">
</div>

# Depends

```
pip install -r requirements.txt
```

# Usage

**1.Training your model**

Run the file 'training.py'. Since the data exceeded the limit, we have uploaded some example data for training，download at [releases.](https://github.com/XiaqiongFan/DeepRaman/releases)

**2.Predict mixture spectra**

Run the file 'testing.py'. Some data has been upload as examples.
The input 1 and input 2 represent pure component spectrum and unknown spectrum, respectively.
More example data for testing can be download at [releases.](https://github.com/XiaqiongFan/DeepRaman/releases)

# Contact

Xiaqiong Fan: fxq@haut.edu.cn
Zhimin Zhang: zmzhang@csu.edu.cn

