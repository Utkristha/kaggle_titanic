import pandas as pd
import numpy as np
def in_csv(np_arr_predicted):
    data_read = pd.read_csv("result.csv")
    data_read.drop('Pclass',inplace=True,axis=1)
    data_read.drop('Name',inplace=True,axis=1)
    data_read.drop('Sex',inplace=True,axis=1)
    data_read.drop('Age',inplace=True,axis=1)
    data_read.drop('SibSp',inplace=True,axis=1)
    data_read.drop('Parch',inplace=True,axis=1)
    data_read.drop('Ticket',inplace=True,axis=1)
    data_read.drop('Fare',inplace=True,axis=1)
    data_read.drop('Cabin',inplace=True,axis=1)
    data_read.drop('Embarked',inplace=True,axis=1)

    data_read.to_csv("result.csv")
    