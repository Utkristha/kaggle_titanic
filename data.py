import numpy as np
import pandas as pd

def data_loading():
    train_data = pd.read_csv("datasets/train.csv")
    test_data = pd.read_csv("datasets/test.csv")

    train_data['Sex'] = train_data['Sex'].map({'female': 1, 'male': 0})
    train_data.drop('SibSp',inplace=True,axis=1)
    train_data.drop('Cabin',inplace=True,axis=1)
    train_data.drop('Embarked',inplace=True,axis=1)
    train_data.drop('Age',inplace=True,axis=1)
    train_data.drop('Ticket',inplace=True,axis=1)
    train_data.drop('Name',inplace=True,axis=1)

    train_data_numpy = train_data.to_numpy()
    train_data_transformed = train_data_numpy.T
    train_data_label = train_data_transformed[1,:]
    train_data_label = train_data_label.reshape((1,891))
    train_data_transformed = np.delete(train_data_transformed,(1),axis=0)
    train_data_passenger_id = train_data_transformed[0,:]
    train_data_passenger_id = train_data_passenger_id.reshape((1,891))
    train_data_transformed = np.delete(train_data_transformed,(0),axis = 0)

    normalizing_vector = np.array([3,1,6,512.329200])
    result = train_data_transformed / normalizing_vector[:, np.newaxis]

    test_data['Sex'] = test_data['Sex'].map({'female': 1, 'male': 0})
    test_data.drop('SibSp',inplace=True,axis=1)
    test_data.drop('Cabin',inplace=True,axis=1)
    test_data.drop('Embarked',inplace=True,axis=1)
    test_data.drop('Age',inplace=True,axis=1)
    test_data.drop('Ticket',inplace=True,axis=1)
    test_data.drop('Name',inplace=True,axis=1)
    test_data.loc[152, "Fare"] = 75
    test_data_numpy = test_data.to_numpy()
    test_data_transformed = test_data_numpy.T
    test_data_passenger_id = test_data_transformed[0,:]
    test_data_passenger_id = test_data_passenger_id.reshape((1,418))
    test_data_transformed = np.delete(test_data_transformed,(0),axis = 0)

    normalizing_test = np.array([3,1,9,512.329200])
    test_data_normalized = test_data_transformed/normalizing_test[:,np.newaxis]

    return result,train_data_label,test_data_normalized
