import pandas as pd
import numpy as np
from scipy.special import boxcox1p
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, LabelEncoder

def feature_engineering(train, test):
    """ 
    Extracts features from the train and test datasets, fills missing values,
    creates new useful features and normalizes features. 
    
    Args:
        train (string): filepath name to the train dataset
        test (string): filepath name to the test dataset
        
    Returns:
        housing_train (pandas DataFrame): train dataset ready for models
        housing_test (pandas DataFrame): test dataset ready for prediction
        test_ID (list of ints): test ID's for submission
        
    """
   # if type(train) or type(test) != str:
       # raise TypeError("inputs must be strings!")
        
        
    housing_train = pd.read_csv(train)
    housing_test = pd.read_csv(test)
    
    test_ID = housing_test['Id']
    
    housing_train.drop('Id', axis = 1, inplace = True)
    housing_test.drop('Id', axis = 1, inplace = True)
    
    na_meaning = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'MasVnrType',
    'GarageQual', 'GarageCond', 'GarageFinish', 'GarageType',
    'BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1']
    for var in na_meaning:
        housing_train[var].fillna('None', inplace = True)
        housing_test[var].fillna('None', inplace = True)
        
    mode_replace = ['MSZoning','Electrical','KitchenQual','Exterior1st','Exterior2nd',
                'SaleType']
    for var in mode_replace:
        housing_train[var] = housing_train[var].fillna(housing_train[var].mode()[0])
        housing_test[var] = housing_test[var].fillna(housing_test[var].mode()[0])
        
    housing_train['LotFrontage'] = housing_train.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median()))
    housing_test['LotFrontage'] = housing_train.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median()))
    
    housing_train['MasVnrArea'].fillna(0, inplace = True)
    housing_train['BsmtFullBath'].fillna(0, inplace = True)
    housing_train['BsmtHalfBath'].fillna(0, inplace = True)
    housing_train['BsmtFinSF1'].fillna(0, inplace = True)    
    housing_train['BsmtFinSF2'].fillna(0, inplace = True)
    housing_train['TotalBsmtSF'].fillna(0, inplace = True)
    housing_train['BsmtUnfSF'].fillna(0, inplace = True)
    housing_train['GarageArea'].fillna(0, inplace = True)
    housing_train['GarageCars'].fillna(0, inplace = True)
    housing_test['MasVnrArea'].fillna(0, inplace = True)
    housing_test['BsmtFullBath'].fillna(0, inplace = True)
    housing_test['BsmtHalfBath'].fillna(0, inplace = True)
    housing_test['BsmtFinSF1'].fillna(0, inplace = True)    
    housing_test['BsmtFinSF2'].fillna(0, inplace = True)
    housing_test['TotalBsmtSF'].fillna(0, inplace = True)
    housing_test['BsmtUnfSF'].fillna(0, inplace = True)
    housing_test['GarageArea'].fillna(0, inplace = True)
    housing_test['GarageCars'].fillna(0, inplace = True)
    
    housing_train['Functional'].fillna("Typical")
    housing_test['Functional'].fillna("Typical")
    
    housing_train['GarageYrBlt'].fillna(0, inplace = True)
    housing_test['GarageYrBlt'].fillna(0, inplace = True)
    
    housing_train = housing_train.drop(['Utilities'], axis = 1)
    housing_test = housing_test.drop(['Utilities'], axis = 1)
    
    housing_train['TotalSF'] = housing_train['TotalBsmtSF']+housing_train['2ndFlrSF']+housing_train['1stFlrSF']
    housing_test['TotalSF'] = housing_test['TotalBsmtSF']+housing_test['2ndFlrSF']+housing_test['1stFlrSF']
    
    log_sp = np.log1p(housing_train.SalePrice)
    housing_train.SalePrice = log_sp
    
    housing_train['MSSubClass'] = housing_train['MSSubClass'].apply(str)
    housing_test['MSSubClass'] = housing_test['MSSubClass'].apply(str)
    to_string = ['OverallCond','MoSold']
    for var in to_string:
        housing_train[var] = housing_train[var].astype(str)
        housing_test[var] = housing_test[var].astype(str)
        
        
    var_types = housing_train.dtypes
    var_cat = var_types[var_types == object].index.tolist()
    var_quant = var_types[var_types != object].index.tolist()
    
    
    var_cont = []
    for var in var_quant:
        if housing_train[var].nunique() > 15:
            var_cont.append(var)
            
    
    for var in ['LowQualFinSF','2ndFlrSF',
              'MiscVal','ScreenPorch','WoodDeckSF','OpenPorchSF',
              'EnclosedPorch','MasVnrArea','GarageArea','Fireplaces',             
              'TotalBsmtSF']:
        housing_train['Has' + var] = (housing_train[var] > 0).astype(int)
        housing_test['Has' + var] = (housing_train[var] > 0).astype(int)
    
    for var in var_cont:
        if var not in ['GrLivArea','LotFrontage','LotArea',
                                 'TotalBsmtSF','1stFlrSF',
                                 '2ndFlrSF','SalePrice', 'TotalSF']:
            housing_train[var] = boxcox1p(housing_train[var], .15)
            housing_test[var] = boxcox1p(housing_test[var], .15)
    
    for var in var_cat:
        le = LabelEncoder()
        le.fit(list(housing_train[var].values))
        housing_train[var] = le.transform(list(housing_train[var].values))
        le.fit(list(housing_test[var].values))
        housing_test[var] = le.transform(list(housing_test[var].values))
        
    housing_train['HasPool'] = (housing_train['PoolQC'] > 0).astype(int)
    housing_train['Has3SsnPorch'] = (housing_train['3SsnPorch'] > 0).astype(int)
    housing_test['HasPool'] = (housing_test['PoolQC'] > 0).astype(int)
    housing_test['Has3SsnPorch'] = (housing_test['3SsnPorch'] > 0).astype(int)
    housing_train.drop(['PoolQC','PoolArea','3SsnPorch'], axis = 1, inplace = True)
    housing_test.drop(['PoolQC','PoolArea','3SsnPorch'], axis = 1, inplace = True)
        
    return housing_train, housing_test, test_ID



def prediction_maker(file_name, housing_train, housing_test, test_ID):
    
    """
    Takes the prepared dataFrames and makes a submission file of
    the predicted values
    
    Args:
        file_name (str): the name of the file to be created
        housing_train (DataFrame): the prepared train dataset
        housing_test (DataFrame): the prepared test dataset
        test_ID (list of ints): the ID's of the test dataset for submission
        
    Returns:
        nothing, creates a submission file
    
    """
    
    ENet = make_pipeline(RobustScaler(), ElasticNet(alpha = .0005,l1_ratio = .9,random_state = 52918))
    lasso = make_pipeline(RobustScaler(), Lasso(alpha = .00022, random_state = 42))
    RF = RandomForestRegressor(n_estimators = 50, max_depth = 30)
    train = housing_train.drop(['SalePrice'], axis = 1).values
    train_y = housing_train.SalePrice.values
    test = housing_test.values
    
    ENet.fit(train, train_y)
    lasso.fit(train, train_y)
    RF.fit(train, train_y)
    predictions = [ENet.predict(test), lasso.predict(test), RF.predict(test)]
   
    predictions = np.expm1(predictions) 
    
   
    
    sub = pd.DataFrame()
    sub['ID'] = test_ID
    sub['SalePrice'] = np.mean(predictions, axis = 0) 
    sub.to_csv(file_name, index = False)
    

    
    
    
    