## -*- coding: utf-8 -*-
""" Jaron Whittington
    House Prices Kaggle """
    
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from scipy.special import boxcox1p
from scipy import stats

#import models
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR



""" Read in the train and test datasets """

housing_train = pd.read_csv('train.csv')

housing_test = pd.read_csv('test.csv')

#save ID values
train_ID = housing_train['Id']
test_ID = housing_test['Id']

#delete ID value from our datasets
housing_train.drop('Id', axis = 1, inplace = True)
housing_test.drop('Id', axis = 1, inplace = True)

#combine datasets into one 
house_all = pd.concat([housing_train,housing_test])

#Find columns with NAN values
house_na = house_all.isna().sum()
house_na = house_na[house_na > 0]

""" After investigating, columns with null values that have 
    a meaning (not a missing value) are:
    PoolQC, MiscFeature, Alley, Fence, FireplaceQU, MasVnrType
    GarageQual, GarageCond, GarageFinish, GarageType,
    BsmtExposure, BsmntCond, BsmtQual, BsmtFinType2, BsmtFinType1 """
    
na_meaning = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'MasVnrType',
    'GarageQual', 'GarageCond', 'GarageFinish', 'GarageType',
    'BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1']

#replace NaN with meaning with 'None'

for var in na_meaning:
    housing_train[var].fillna('None', inplace = True)
    housing_test[var].fillna('None', inplace = True)
    
#replace 1 or 2 NA's with mode 
mode_replace = ['MSZoning','Electrical','KitchenQual','Exterior1st','Exterior2nd',
                'SaleType']
for var in mode_replace:
    housing_train[var] = housing_train[var].fillna(housing_train[var].mode()[0])
    housing_test[var] = housing_test[var].fillna(housing_test[var].mode()[0])


#for LotFrontage, we can replace missing values with the mode of the neighborhood
housing_train['LotFrontage'] = housing_train.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median()))
housing_test['LotFrontage'] = housing_train.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median()))
    
 
#Fill all areas and counts with NaN's with 0
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

#Functional: NaN's should be Typical per the description
housing_train['Functional'].fillna("Typical")
housing_test['Functional'].fillna("Typical")

#GarageYrBlt will be replaced with the year the house was built
housing_train['GarageYrBlt'].fillna(0, inplace = True)
housing_test['GarageYrBlt'].fillna(0, inplace = True)

#Utilitis is the same for all of test dataset so we can safely delete it
housing_train = housing_train.drop(['Utilities'], axis = 1)
housing_test = housing_test.drop(['Utilities'], axis = 1)

"""Done with missing values"""

#Add a variable for total square feet
housing_train['TotalSF'] = housing_train['TotalBsmtSF']+housing_train['2ndFlrSF']+housing_train['1stFlrSF']
housing_test['TotalSF'] = housing_test['TotalBsmtSF']+housing_test['2ndFlrSF']+housing_test['1stFlrSF']
#EDA on response variable SalePrice
plt.hist(housing_train.SalePrice, bins = 100)
plt.show()


#SalePrice is right skewed, so we log transorm it
log_sp = np.log1p(housing_train.SalePrice)
plt.hist(log_sp, bins = 100)
plt.show()
#The log transform is much less skewed, so replace SalePrice with log_sp

housing_train.SalePrice = log_sp


#investigate explanatory variables

#Some numerical values should be categorical
housing_train['MSSubClass'] = housing_train['MSSubClass'].apply(str)
housing_test['MSSubClass'] = housing_test['MSSubClass'].apply(str)
to_string = ['OverallCond','MoSold']
for var in to_string:
    housing_train[var] = housing_train[var].astype(str)
    housing_test[var] = housing_test[var].astype(str)

#first sepearate variables between categorical and quantitative

var_types = housing_train.dtypes
var_cat = var_types[var_types == object].index.tolist()
var_quant = var_types[var_types != object].index.tolist()


#find which variables are continuous (more than 15 unique)
var_cont = []
for var in var_quant:
    if housing_train[var].nunique() > 15:
        var_cont.append(var)

#plot the continuous variables to determine if they need to be transformed

i = 0  
plt.figure(figsize = (5, 4*len(var_cont)))      
for var in var_cont:
    if var != 'SalePrice':
        i += 1
        ax = plt.subplot(len(var_cont), 1, i)
        sns.distplot(housing_train[var].dropna(), fit = stats.norm)
        
#Deal with the variables that have a lot of zeros
        
#Pools and 3SsnPorch are very uncommon 
housing_train['HasPool'] = (housing_train['PoolQC'] > 0).astype(int)
housing_train['Has3SsnPorch'] = (housing_train['3SsnPorch'] > 0).astype(int)
housing_test['HasPool'] = (housing_test['PoolQC'] > 0).astype(int)
housing_test['Has3SsnPorch'] = (housing_test['3SsnPorch'] > 0).astype(int)
housing_train.drop(['PoolQC','PoolArea','3SsnPorch'], axis = 1, inplace = True)
housing_test.drop(['PoolQC','PoolArea','3SsnPorch'], axis = 1, inplace = True)

#For other variables with lots of zeros add a dummy variable for Has_
for var in ['LowQualFinSF','2ndFlrSF',
              'MiscVal','ScreenPorch','WoodDeckSF','OpenPorchSF',
              'EnclosedPorch','MasVnrArea','GarageArea','Fireplaces',             
              'TotalBsmtSF']:
    housing_train['Has' + var] = (housing_train[var] > 0).astype(int)
    housing_test['Has' + var] = (housing_train[var] > 0).astype(int)
    
#use boxcox transform for those in need of trasnforming
for var in var_cont:
    if var not in ['GrLivArea','LotFrontage','LotArea',
                                 'TotalBsmtSF','1stFlrSF',
                                 '2ndFlrSF','SalePrice', 'TotalSF']:
        housing_train[var] = boxcox1p(housing_train[var], .15)
        housing_test[var] = boxcox1p(housing_test[var], .15)
        
#Label Encoding on the categorical variables
for var in var_cat:
    le = LabelEncoder()
    le.fit(list(housing_train[var].values))
    housing_train[var] = le.transform(list(housing_train[var].values))
    le.fit(list(housing_test[var].values))
    housing_test[var] = le.transform(list(housing_test[var].values))

#get dummy variables
housing_train = pd.get_dummies(housing_train)
#Cross validation function

def rmse_cross_validation(model, n_folds = 5):
    """
    takes the model and calcualtes the rmse using k folds cross validation
    
    Args:
        model (Estimator): the model used for prediciton
        n_folds (int): Number of folds to use in K Folds procedure
        
    Returns:
        rmse (float): The root mean squared error of the model
"""
    k_folds = KFold(n_folds, shuffle = True, random_state = 52918)
    rmse = np.sqrt(-cross_val_score(model, housing_train.values,housing_train.SalePrice.values,
                                    scoring = "neg_mean_squared_error", cv = k_folds))
    return rmse

#make some models
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha = .0005,l1_ratio = .9,random_state = 52918))
lasso = make_pipeline(RobustScaler(), Lasso(alpha = .00022, random_state = 42))
RF = RandomForestRegressor(n_estimators = 50, max_depth = 30)
KRidge = make_pipeline(RobustScaler(), KernelRidge())
BRidge = make_pipeline(RobustScaler(), BayesianRidge())
S_V_R = make_pipeline(RobustScaler(), SVR())

#train the models
train = housing_train.drop(['SalePrice'], axis = 1).values
train_y = housing_train.SalePrice.values
test = housing_test.values
def rmse(model):
    """
    Finds the rmse of the model on the train dataset
    
    Args:
        model (estimator): The model to be tested
        
    Returns:
        rmse (float): The RMSE of the model on the train dataset
    """
    model.fit(train, train_y)
    return np.sqrt(mean_squared_error(train_y, model.predict(train)))

def averaged_prediction(models):
    """ 
    Takes a set of models, evaluates their effectiveness using
    the rmse_cross_validation function, then fits the 'good'
    models and returns the average prediction from them
        
    Args:
        models (iterable of estimators): the models to be tested 
        
    Returns:
        prediction (numpy array): the averaged prediciton from the 
        good models. 
    """
    good_models = []
    for model in models:
        #test effectiveness of the model
        if rmse_cross_validation(model).mean() < .05:
            good_models.append(model)
            
    predictions = []
    for model in good_models:
        #train the good models
        model.fit(train, train_y)
        predictions.append(np.expm1(model.predict(train)))
        
    return np.mean(predictions, axis = 0)
        
prediction = averaged_prediction((ENet, lasso, KRidge, BRidge))

#create submission file
sub = pd.DataFrame()
sub['ID'] = test_ID
sub['SalePrice'] = prediction
sub.to_csv('test_submission.csv', index = False)
        
    





