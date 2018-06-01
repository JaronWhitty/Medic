import pytest 
import pandas as pd
from housing_predictor import housing_predictor as hp


@pytest.fixture
def set_up_feature():
    test = 'housing_predictor/data/test.csv'
    train = 'housing_predictor/data/train.csv'
    return test, train
    
def test_feature_engineering(set_up_feature):
    test, train = set_up_feature
    final_train, final_test, train_ID = hp.feature_engineering(train, test)
    assert not pd.read_csv(test).empty
    assert type(pd.read_csv(test)) == pd.DataFrame
    housing_test = pd.read_csv(test)
    housing_train = pd.read_csv(train)
    assert not housing_test.empty
    assert not housing_train.empty
    #with pytest.raises(TypeError) as excinfo:
        #hp.feature_engineering(5,'test.csv')
    #assert excinfo.value.args[0] == "inputs must be strings!"
    assert type(housing_train) == pd.core.frame.DataFrame
    assert type(housing_test) == pd.core.frame.DataFrame
    assert housing_train.shape == (1460, 81)
    assert housing_test.shape == (1459, 80)
  
@pytest.fixture
def set_up_prediction():
    train, test, train_ID = hp.feature_engineering('housing_predictor/data/train.csv',
                                                                     'housing_predictor/data/test.csv')
    return train, test, train_ID
    
def test_prediction_maker(set_up_prediction):
    train, test, test_ID = set_up_prediction
    hp.prediction_maker('testing.csv', train, test, test_ID)
    assert not pd.read_csv('testing.csv').empty
    
        

