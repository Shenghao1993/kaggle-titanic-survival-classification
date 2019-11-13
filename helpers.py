import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc


def extract_title(full_name):
    names = full_name.split(' ')
    for name in names:
        if name[-1] == '.':
            return name[:-1]
    return ''


def engineer_features(df):
    '''Process features '''
    # Cabin: Generate Deck, Room from Cabin
    df['Deck'] = df['Cabin'].str.slice(0,1)
    df['Deck'].fillna('U', inplace=True)
    df['Room'] = df['Cabin'].str.slice(1,5).str.extract("([0-9]+)", expand=False).astype("float")
    df['Room'].fillna(df["Room"].mean(), inplace=True)
    
    # Embarked: Fill in missing values with the most popular port
    df['Embarked'].fillna('S', inplace=True)
    
    # Name: Generate Title
    df['Title'] = df['Name'].apply(extract_title)
    popular_titles = ['Mrs', 'Miss', 'Master', 'Mr']
    df['Title'] = np.where(~df['Title'].isin(popular_titles), 'Other', df['Title'])
    
    return df


def preprocess(training_data, test_data):
    # Combine training and test datasets to have more training data with valid age values
    train_df = pd.read_csv(training_data)
    test_df = pd.read_csv(test_data)
    combined_df = pd.concat([train_df, test_df])
    selected_features = ['PassengerId', 'Age', 'Deck', 'Room', 'Embarked', 'Fare', 'Title',
                         'Parch', 'Pclass', 'Sex', 'SibSp', 'Survived']
    # One-hot encode the engineered features
    processed_combined_df = pd.get_dummies(engineer_features(combined_df)[selected_features])
    print(processed_combined_df.columns)

    # Filter out unwanted features
    features = list(processed_combined_df.columns)
    for feature in ['PassengerId', 'Age', 'Survived']:
        features.remove(feature)

    # Fill in missing value with the average Fare of class 3 
    avg_class3_fare = processed_combined_df[processed_combined_df['Pclass'] == 3]['Fare'].mean()
    processed_combined_df['Fare'].fillna(avg_class3_fare, inplace=True)
    null_age_df = processed_combined_df[processed_combined_df['Age'].isnull()]
    not_null_age_df = processed_combined_df[processed_combined_df.Age.notnull()]
    print("Number of passengers with age data: %s" %not_null_age_df.shape[0])
    print("Number of passengers with no age data: %s" %null_age_df.shape[0])

    # Fit Random Forest model to predict missing age
    age_rf = RandomForestRegressor(n_estimators=1000, random_state=42)
    age_rf.fit(not_null_age_df[features], not_null_age_df['Age'])
    null_age_df.loc[:, 'Predicted_Age'] = age_rf.predict(null_age_df[features])
    not_null_age_df['Predicted_Age'] = not_null_age_df['Age'] 

    # Combine data with predicted age
    age_predicted_df = pd.concat([not_null_age_df, null_age_df], sort=False)
    age_predicted_df = pd.merge(age_predicted_df, combined_df[['PassengerId', 'Name']],
                                how='left', on=['PassengerId']) 
    return age_predicted_df, features


def evaluate(y_true, y_pred):
    '''Evaluate performance of classification
    '''
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    print("Accuracy = ", round(accuracy, 6))
    print("Precision = tp / (tp + fp), Precision = ", round(precision, 6))
    print("Recall = tp / (tp + fn), Recall = ", round(recall, 6))
    print("ROC = ", round(roc_auc, 6))