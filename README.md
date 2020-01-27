Titanic - Passenger Survival Prediction

Project Organization
------------

    ├── README.md               <- The top-level README for developers using this project.
    ├── data
    │   ├── train.csv           <- Training dataset provided by Kaggle.
    │   ├── test.csv            <- Test dataset provided by Kaggle.
    │   └── ground-truth.csv    <- Ground truth dataset which contains actual label for both training and test data.
    │
    ├── img                     <- Images inserted to notebook
    │
    ├── 1.0-wsh-titanic-exploratory-data-analysis.ipynb
    │                           <- Jupyter notebooks for exploratory study and feature engineering
    │ 
    ├── 2.0-wsh-titanic-modelling-and-evaluation.ipynb
    │                           <- Jupyter notebooks for model training and parameter tuning
    │
    ├── requirements.txt        <- The requirements file for reproducing the analysis environment, e.g.
    │                              generated with `pip freeze > requirements.txt`
    │
    ├── helps.py                <- Helper script for data preprocessing and parameter tuning
    │ 
    └── submission.csv          <- Prediction of labels for test data submitted to Kaggle.

--------

# Install and start Jupyter Lab
```
$ sudo -H pip3 uninstall -y jupyterlab && sudo -H pip3 install jupyterlab
$ jupyter-lab
```
Follow the guidelines on https://plot.ly/python/getting-started/ to install extensions for the use of plotly with Jupyter Lab.


# Install lightgbm on macOS system
```
$ brew install cmake
$ brew install gcc@7
$ git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
$ mkdir build ; cd build
$ cmake -DCMAKE_CXX_COMPILER=g++-7 -DCMAKE_C_COMPILER=gcc-7 ..
$ make -j
$ cp -r LightGBM/python-package/lightgbm python3.7/site-packages/
$ cp LightGBM/lib_lightgbm.so python3.7/site-packages/lightgbm/
```