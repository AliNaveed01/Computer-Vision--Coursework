#Importing Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
%matplotlib inline

# For reading stock data from yahoo
from pandas_datareader.data import DataReader
import yfinance as yf
import numpy as np

# For time stamps
from datetime import datetime

### Create the LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM



# Loading Data
# Defining the tickers or indices
tickers = ['BTC-USD', 'ETH-USD','ADA-USD']

# Intializing the datetime as per today
end = datetime.now()
# Getting records of 2 year
start = datetime(end.year - 2, end.month, end.day)

# Extraction of stock data
df_btc= yf.download(tickers[0], start, end)
df_eth= yf.download(tickers[1], start, end)


df_ada = yf.download(tickers[2], start, end)


# df_btc

df_btc.to_csv('btc.csv')
df_ada.to_csv('ada.csv')
df_eth.to_csv('eth.csv')

print(df_btc.info())

# Display summary statistics
print(df_btc.describe())
print()
# Check for missing values
print(df_btc.isnull().sum())


df_btc.describe()


fig, axes = plt.subplots(1,2,figsize=(15, 10))
df_btc['Close'].plot(ax=axes[0])
axes[0].set_title("BTC-USD")
axes[0].set_ylabel('Close')
df_eth['Close'].plot(ax=axes[1])
axes[1].set_title("ETH-USD")
axes[1].set_ylabel('Close')


fig, axes = plt.subplots(1,2,figsize=(15,10))
df_ada['Close'].plot(ax=axes[0])
axes[0].set_title("ADA-USD")
axes[0].set_ylabel('Close')
df_eth['Close'].plot(ax=axes[1])
axes[1].set_title("ETH-USD")
axes[1].set_ylabel('Close')

plt.show()

# Using MinMaxScaler to scale the Close Attribute

scaler=MinMaxScaler(feature_range=(0,1))
btc=scaler.fit_transform(np.array(df_btc['Close']).reshape(-1,1))

##splitting dataset into train and test split by 80% 
training_size=int(len(btc)*0.8)


test_size=len(btc)-training_size
train_data,test_data=btc[0:training_size,:],btc[training_size:len(btc),:1]



 
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
 dataX, dataY = [], []
 for i in range(len(dataset)-time_step-1):
  a = dataset[i:(i+time_step), 0]   
  dataX.append(a)
  dataY.append(dataset[i + time_step, 0])
 return np.array(dataX), np.array(dataY)


# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 50
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# reshape input to be [samples, time steps, features] which is required for LSTM
'''
LSTM needs 3D shape therefore it needs to be change
'''
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)





# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (X_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
# Train the model
model.fit(X_train, y_train,validation_data=(X_test,ytest),epochs=500,batch_size=20,verbose=1)

### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


### Plotting 
# shift train predictions for plotting
look_back=50
trainPredictPlot = numpy.empty_like(btc)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(btc)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(btc)-1, :] = test_predict


# plot baseline and predictions
plt.plot(scaler.inverse_transform(btc))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.title('BTC-USD Model')
plt.xlabel('Days', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.legend(['Train', 'Val', 'Predictions'], loc='upper right')
plt.show()



x_input=test_data[30:].reshape(1,-1)

temp_input=list(x_input)
temp_input=temp_input[0].tolist()
print(x_input)



 import array

lst_output=[]
n_steps=44
i=0
while(i<30):
    
    if(len(temp_input)>44):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1

print(lst_output)



df3=btc.tolist()
df3.extend(lst_output)
plt.plot(df3[1200:])
df3=scaler.inverse_transform(df3).tolist()
btc_1=scaler.inverse_transform(btc).tolist()

#Demonstrating the forecast values using a plot
plt.title('BTC-USD Model')
plt.xlabel('Days', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(df3)
plt.plot(btc_1,color = "b")
plt.legend(['Current', 'Forecast'], loc='upper right')