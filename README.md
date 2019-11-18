Titanic - Passenger Survival Prediction

# Install lightgbm on macOS system
$ brew install cmake
$ brew install gcc@7
$ git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
$ mkdir build ; cd build
$ cmake -DCMAKE_CXX_COMPILER=g++-7 -DCMAKE_C_COMPILER=gcc-7 ..
$ make -j
$ cp -r LightGBM/python-package/lightgbm python3.7/site-packages/
$ cp LightGBM/lib_lightgbm.so python3.7/site-packages/lightgbm/ 

# Features in use
1. Pclass;
2. Sex;
3. Age: fit a Random Forest model to fill in the 177 missing values in the training data;
4. Number of family members onboard: sum of SibSp and Parch;
5. Fare;
6. Embarked: assign the most popular port S to the 2 missing values;
7. Passenger title: Mrs, Miss, Master, Mr, the rest titles;
8. Cabin: 


# Machine Learning models
1. Logistic Regression;
2. Random Forest;
3. Support Vector Machine;
4. XGBoost;
5. LightGBM.


# References
Exploratory Data Analysis on the Titanic Dataset by Asela
https://www.kaggle.com/aselad/exploratory-data-analysis-on-the-titanic-dataset

5 Ways To Handle Missing Values In Machine Learning Datasets
https://www.analyticsindiamag.com/5-ways-handle-missing-values-machine-learning-datasets/

Finding Important Factors To Survive Titanic
https://www.kaggle.com/jatturat/finding-important-factors-to-survive-titanic

Analyzing Titanic Dataset
https://www.kaggle.com/viveksrinivasan/analyzing-titanic-dataset

Titanic Cabin Features
https://www.kaggle.com/ccastleberry/titanic-cabin-features

Introduction to Ensembling/Stacking in Python
https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python

LightGBM hyperparameter optimisation (LB: 0.761)
https://www.kaggle.com/mlisovyi/lightgbm-hyperparameter-optimisation-lb-0-761

StackingClassifier
http://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/

Benchmark accuracy: 0.8