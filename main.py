# Import python packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import requests
import time

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error

from helperlib import parameter_type

# Page Congifuration
st.set_page_config(
    page_title='HYPER-VISOR', 
    page_icon='🕐',
    layout='wide'
)

st.markdown("<h2 style='text-align: center; color: rgb(0, 0, 0);'> HYPER - VISOR </h2>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload data:")
if uploaded_file is not None and 'my_data' not in st.session_state:
    df = pd.read_csv(uploaded_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')
    df = df.set_index(['date'])
    df = df[['data', 'f_1']]
    st.dataframe(df,height=150,use_container_width=True)
    st.session_state['my_data'] = df

param = 0
param_input = st.text_input("Enter hyperparameter: (example - n_estimators 1 10 1)")
if st.button('Run Simulation', use_container_width=True):
    param = param_input.split(' ')
    start = parameter_type[param[0]](param[1])
    stop = parameter_type[param[0]](param[2])
    step = parameter_type[param[0]](param[3])
    
    result = pd.DataFrame()
    
    for item in np.arange(start, stop, step):
        tmp_result = pd.DataFrame()
        placeholder = st.empty()
        placeholder_metric = st.empty()
        
        df = st.session_state['my_data']
        model = XGBRegressor(**{param[0]:item})
        
        train_x = df.drop(['data'], axis=1)
        train_y = df.data
        model.fit(train_x, train_y)
        
        pred = model.predict(train_x)
        df['pred'] = pred
        
        display_data = df[['data', 'pred']]
        placeholder.line_chart(display_data)
        mape = mean_absolute_percentage_error (train_y, pred)
        test_mape = mape * 100
        accuracy = 100 - test_mape
        placeholder_metric.success("Accuracy : " + str(round(accuracy,2)) + " %")
        tmp_result[param[0]] = [item]
        tmp_result['accuracy'] = accuracy
        result = result.append(tmp_result)

        time.sleep(1)
        if item < float(param[2])-1:
            placeholder.empty()
            placeholder_metric.empty()


    st.dataframe(result.set_index(param[0]),height=200,use_container_width=True)
