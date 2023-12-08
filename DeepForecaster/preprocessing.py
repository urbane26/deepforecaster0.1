import pandas as pd
import numpy as np
#from sklearn.preprocessing import StandardScaler
from feature_engine.encoding import OneHotEncoder


def datasetset_preprocessing_pipeline(dataset):
    featureframe = dataset.drop(['ds','y','unique_id'],axis = 1)

    #Identify numeric and categorical features
    numeric_features = featureframe.select_dtypes(include=['float', 'int']).columns
    try:
        categorical_features = featureframe.select_dtypes(include=['object']).columns
        cat = featureframe.select_dtypes(include=['object']).columns.to_list()
    except:
        pass

    #Handle missing values in numeric features
    dataset[numeric_features] = dataset[numeric_features].fillna(dataset[numeric_features].mean())

    #Detect and handle outliers in numeric features using IQR
    for feature in numeric_features:
        Q1 = dataset[feature].quantile(0.25)
        Q3 = dataset[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (1.5 * IQR)
        upper_bound = Q3 + (1.5 * IQR)
        dataset[feature] = np.where((dataset[feature] < lower_bound) | (dataset[feature] > upper_bound),
                                 dataset[feature].mean(), dataset[feature])

    #Normalize numeric features
    #scaler = StandardScaler()
    #scaled_dataset = scaler.fit_transform(dataset[numeric_features])
    #dataset[numeric_features] = scaler.transform(dataset[numeric_features])

    #Handle missing values in categorical features
    try:
        dataset[categorical_features] = dataset[categorical_features].fillna(dataset[categorical_features].mode().iloc[0])
    
        #one hot encoding categorical values
        ohe_enc = OneHotEncoder(top_categories = None,variables = cat,drop_last = False)
        dataset = ohe_enc.fit(dataset)
        dataset = ohe_enc.transform(dataset)
    except:
        pass

    return dataset


def Dataresampling(dataset : pd.DataFrame,frequency : str,timecol : pd.Series):
    data = dataset.resample(frequency, on=timecol).sum()
    return data
