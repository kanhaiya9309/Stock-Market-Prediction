import numpy as np
import pandas as pd
pip install matplotlib 
import matplotlib.pyplot as plt

import pandas_datareader as data
#from keras.model import load_model
import stramlit as st


#pip install yfinance

from pandas_datareader import data as pdr

import yfinance as yf
yf.pdr_override() # <== that's all it takes :-)

st.title("Stock Trend Predition")

user_input=st.text_input("Enter Stock Ticker","AAPL")

# download dataframe
df = pdr.get_data_yahoo("user_input", start="2023-01-01", end="2023-05-20")


#Describing Data

st.subheader('Date from 2022-2023')
st.write(df.describe())

