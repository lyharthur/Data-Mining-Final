import csv
import numpy as np
import itertools

from sklearn.ensemble import RandomForestClassifier ,RandomForestRegressor

from sklearn.metrics import mean_squared_error

file = ['data_28C2DDDD41EB.csv','data_28C2DDDD457E.csv','data_28C2DDDD4404.csv', 'data_28C2DDDD4534.csv']

for i in range(len(file)):
    print('Sensor name : ',file[i])
    data = []
    target = []
    with open(file[i], newline='') as csvfile:
        header1 = next(csvfile)
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            #print(row)
            #row = list(map(int, row))
            data.append(row[1:-1])#all - s_d0
            target.append(row[-1])#target(s_d0)


    #print(data,target)
    X_train = data[:-63 :]
    X_test = data[-63: :]
    y_train = target[:-63 :]
    y_test = target[-63: :]



    n_samples = len(X_train)
    n_test = len(X_test)
    print('Train num : ', n_samples)
    print('Test num : ',n_test)

    RFc = RandomForestRegressor(n_estimators=10, criterion='mse', max_depth=10, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.4)

    prediction_RFc = RFc.fit(X_train, y_train).predict(X_test)
    y_pred = prediction_RFc.tolist()

    y_test = [ float(x) for x in y_test ]
    y_pred = [ float(x) for x in y_pred ]


    #print(y_test,y_pred)
    print("MSE : " , mean_squared_error(y_test, y_pred))
    print(" ")


    #print(type(prediction_RFc.tolist()))
    diff = list(map(lambda x: float(x[0]) - float(x[1]), zip(y_test, prediction_RFc)))

    #diff = y_test - prediction_RFc.tolist()
    #print("y - y_hat" ,diff)

