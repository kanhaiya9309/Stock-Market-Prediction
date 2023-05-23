
import numpy as np
import pandas as pd
pip install matplotlib
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st


#only two command work 

#pip install yfinance
import yfinance as yf
from pandas_datareader import data as pdr

yf.pdr_override() # <== that's all it takes :-)
start='2015-01-01'
end='2023-05-20'

st.title("Stock Trend Predition")


user_input=st.text_input("Enter Stock Ticker","TSLA")

# download dataframe
df = pdr.get_data_yahoo(user_input, start, end)


#Describing Data

st.subheader('Date from 2015-2023')
st.write(df.describe())



#visalizations 

st.subheader("Closing Price vs Time chart")
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader("Closing Price vs Time chart with 100MA")
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)



st.subheader("Closing Price vs Time chart with 100MA & 200MA")
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)



#Spliting Data into Training and Testing 

data_training =pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

#print(data_training.shape)
#print(data_testing.shape)


#scalling the Data 

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))


data_traning_array=scaler.fit_transform(data_training)
#data_traning_array

#Splitting Daat into X_train and Y_train

x_train = []
y_train = []

for i in range(100, data_traning_array.shape[0]):
    x_train.append(data_traning_array[i-100:i])
    y_train.append(data_traning_array[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)


#load my model 

model=load_model('keras_model.h5')

#Testing Part 

past_100_days=data_training.tail(100)

final_df=past_100_days.append(data_testing,ignore_index=True)


#scaler=MinMaxScaler(feature_range=(0,1))
input_data=scaler.fit_transform(final_df)
#input_data

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test = np.array(x_test)
y_test = np.array(y_test)

#Making prediction 

y_predicted=model.predict(x_test)

scaler=scaler.scale_

scale_factor=1/scaler[0]
y_predicted=y_predicted * scale_factor
y_test=y_test * scale_factor

#final Graph
st.subheader('Prediction vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted ,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

