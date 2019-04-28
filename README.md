Titanic - Passenger Survival Prediction

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
1. Random Forest.
2. Catboost.
3. Support Vector Machine: one-hot encoding for categorical variables is required.


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

Benchmark accuracy: 0.8