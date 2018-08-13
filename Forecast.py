# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 23:59:31 2018

@author: vivek
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from pyramid.arima import auto_arima

data = pd.read_csv("C:\\Users\\vivek\\Downloads\\Forecast.csv")
data.index = pd.to_datetime(data['Month'], format = "%m/%d/%Y")
result = seasonal_decompose(data['Milk'], model='multiplicative')
result.plot()
data2 = data.drop(columns = ["Month"])
stepwise_model = auto_arima(data2, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
stepwise_model.fit(data2)
future_forecast = stepwise_model.predict(n_periods=24)
future_months = np.arange('2015-04', '2017-04', np.timedelta64(1, 'M'), dtype='datetime64')