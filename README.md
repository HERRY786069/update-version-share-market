# update-version-share-market
nazim4321raza@gmail.com
import numpy as np
import pandas as pd
import matplotlip.pyplot as plt
import pandas_datareader as data 
from keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = '2019-12-31'  
 
st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = data.DataReader('AAPL','yahoo',start,end)

#Descriping the data
st.subheader('Data from 2010 - 2019')
st.write(df.describe())

keep this in the folder and that folder what you have 
saved everything in  search bar that folder open the
cammand prompt and write in that streamlit run app.py 

#Visualzations
st.subheader('Closing price vs Time chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.close)
st.pyplot(fig)

st.subheader('Closing price vs Time chart with 100MA')
ma100 = df.close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.close)
st.pyplot(fig)


st.subheader('Closing price vs Time chart with 100MA & 200MA')
ma100 = df.close.rolling(100).mean()
ma200 = df.close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.close, 'b')
st.pyplot(fig)

data_training = pd.DataFrame(df['close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['close'][int(len(df)*0.70): int(len(df))])

print(data_training.shape)
print(data_testing.shape)


from sklearn.preprocessing import MinMaxscaler
scaler = MinMaxscaler(feature_range=(0,1))

data_training_array = scaler.fit_tramsformer(data_training)



x_train, y_train = np.array(x_train), np.array(y_train)


model = load_model('Keras_model.h5')

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_train = []
y_train = []

for i in range(100, input_data.shape[0]):
x_test.append(input_data.shape[i-100: i])
y_test.append(input_data[i, 0])



x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

scaler = scaler.scale_


scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original price')
plt.plot(y_predicted, 'r', label = 'predicted price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

