import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import oandapy
import configparser
import pytz
import datetime
from datetime import datetime, timedelta
# Kerasの使用するコンポーネントをインポートしましょう
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM, Embedding
from keras.layers import Dropout
from keras import metrics

# 日付の調整する
# 文字列 -> datetime
def iso_jp(iso):
    date = None
    try:
        date = datetime.strptime(iso, '%Y-%m-%dT%H:%M:%S.%fZ')
        date = pytz.utc.localize(date).astimezone(pytz.timezone("Asia/Tokyo"))
    except ValueError:
        try:
            date = datetime.strptime(iso, '%Y-%m-%dT%H:%M:%S.%f%z')
            date = dt.astimezone(pytz.timezone("Asia/Tokyo"))
        except ValueError:
            pass
    return date
 
# datetime -> 表示用文字列
def date_string(date):
    if date is None:
        return ''
    return date.strftime('%Y/%m/%d %H:%M:%S')


ACCOUNT_ID = "1401673"
ACCESS_TOKEN = "88cb1d105fa6e388042747e4bfea5943-a4d0a01845fd3924f3aaf6ab10fce462"
 
oanda = oandapy.API(environment="practice", access_token=ACCESS_TOKEN)
 

#response = oanda.get_history(instrument="GBP_JPY", granularity="M5")

#res = pd.DataFrame(response["candles"])
 
# 最初の5行を表示させる
#res.head()
 
#res['time'] = res['time'].apply(lambda x: iso_jp(x))
#res['time'] = res['time'].apply(lambda x: date_string(x))
 
#print(res.head())
 
#df = res[['time', 'openAsk', 'closeAsk', 'highAsk', 'lowAsk', 'volume']]
#df.columns = ['time', 'open', 'close', 'high', 'low', 'volume']
 

for i in range(3):
    if i == 0:
        res_hist = oanda.get_history(instrument="GBP_JPY", granularity="M5", count=5000)
    else:
        res_hist = oanda.get_history(instrument="GBP_JPY", granularity="M5",end=endtime, count=5000)
    res = res_hist.get("candles")
    endtime = res[0]['time']
    if i == 0 : res1 = res
    else :
        for j in range(len(res1)):
            res.append(res1[j])
        res1 = res
    print('res ok', i+1, 'and', 'time =', res1[0]['time'])
 
#データフレームに変換しておく
res2= pd.DataFrame(res1)

 
#取得件数を数えて出力
#print(len(df))
res2["time"] = res2["time"].apply(lambda x: iso_jp(x))
res2['time'] = res2['time'].apply(lambda x: date_string(x))

#print(res2)

df = res2[['time','closeAsk']]
df.columns = ['time','close']



#print(df[479:480])
print(df[14799:14800])


split_date = '2019/07/19 20:40:00'
train, test = df[df['time'] < split_date], df[df['time']>=split_date]
del train['time']
del test['time']



# windowを設定
window_len = 12

batch_size = 1

# LSTMへの入力用に処理（訓練）
train_lstm_in = []
for i in range(len(train) - window_len):
    temp = train[i:(i + window_len)].copy()
    for col in train:      
        temp.loc[:, col] = temp[col] / temp[col].iloc[0] - 1
    train_lstm_in.append(temp)
 
lstm_train_out = (train['close'][window_len:].values / train['close'][:-window_len].values)-1

"""
lstm_train_out_bi = []
for j in range(len(lstm_train_out)):
    if(lstm_train_out[j] >= 0):
        lstm_train_out_bi.append(1)
    else:
        lstm_train_out_bi.append(0)
"""


# LSTMへの入力用に処理（テスト）
test_lstm_in = []
for i in range(len(test) - window_len):
    temp = test[i:(i + window_len)].copy()
    for col in test:
        temp.loc[:, col] = temp[col] / temp[col].iloc[0] - 1
    test_lstm_in.append(temp)
lstm_test_out = (test['close'][window_len:].values / test['close'][:-window_len].values)-1
 



# PandasのデータフレームからNumpy配列へ変換しましょう
train_lstm_in = [np.array(train_lstm_input) for train_lstm_input in train_lstm_in]

train_lstm_in = np.array(train_lstm_in)
#lstm_train_out_bi = np.array(lstm_train_out_bi)

test_lstm_in = [np.array(test_lstm_input) for test_lstm_input in test_lstm_in]
test_lstm_in = np.array(test_lstm_in)


# LSTMのモデルを設定
def build_model(inputs, output_size, neurons, activ_func="linear",
                dropout=0.25, loss="mae", optimizer="adam"):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(inputs[1].shape)))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer)
    return model


# ランダムシードの設定
np.random.seed(202)

#print(train_lstm_in)
# 初期モデルの構築
yen_model = build_model(train_lstm_in, output_size=1, neurons = 20)

print(train_lstm_in.shape)
yen_model.summary()

#print(test_lstm_in)
print(lstm_test_out.shape)




# データを流してフィッティングさせましょう
yen_history = yen_model.fit(train_lstm_in, lstm_train_out, 
                            epochs=3, batch_size=1, verbose=1, shuffle=True)


#予測精度をみる１~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
predict_test = yen_model.predict(test_lstm_in)
predict_test = predict_test.flatten() 

predict_test_bi = []

for i  in range(len(predict_test)):
    if(predict_test[i] >=0):
        predict_test_bi.append(1)
    else:
        predict_test_bi.append(0)
predict_test_bi = np.array(predict_test_bi)
print(predict_test_bi)

print(predict_test_bi.shape)

test_out_bi = []

for i  in range(len(lstm_test_out)):
    if(lstm_test_out[i] >=0):
        test_out_bi.append(1)
    else:
        test_out_bi.append(0)
test_out_bi = np.array(test_out_bi)

print(test_out_bi)
print(test_out_bi.shape)
accuracy_count = 0

for i in range(len(test_out_bi)):
    if(predict_test_bi[i] == test_out_bi[i]):
        accuracy_count += 1

accuracy = accuracy_count / len(test_out_bi) * 100

print("accuracy is",accuracy, "%")



fig, ax1 = plt.subplots(1,1)
ax1.plot(df[df['time']>= split_date]['time'][window_len:],
         test['close'][window_len:], label='Actual', color='blue')
ax1.plot(df[df['time']>= split_date]['time'][window_len:],
         (predict_test +1) * test['close'].values[:-window_len], 
         label='Predicted', color='red')
ax1.grid(True)
plt.show()


predict_train = yen_model.predict(train_lstm_in)
predict_train = predict_train.flatten() 

predict_train_bi = []

for i  in range(len(predict_train)):
    if(predict_train[i] >=0):
        predict_train_bi.append(1)
    else:
        predict_train_bi.append(0)
predict_train_bi = np.array(predict_train_bi)
print(predict_train_bi)

print(predict_train_bi.shape)

train_out_bi = []

for i  in range(len(lstm_train_out)):
    if(lstm_train_out[i] >=0):
        train_out_bi.append(1)
    else:
        train_out_bi.append(0)
train_out_bi = np.array(train_out_bi)

print(train_out_bi)
print(train_out_bi.shape)
accuracy_count = 0

for i in range(len(train_out_bi)):
    if(predict_train_bi[i] == train_out_bi[i]):
        accuracy_count += 1

accuracy = accuracy_count / len(train_out_bi) * 100

print("accuracy is",accuracy, "%")


fig, ax1 = plt.subplots(1,1)
ax1.plot(df[df['time']< split_date]['time'][window_len:],
         train['close'][window_len:], label='Actual', color='blue')
ax1.plot(df[df['time']< split_date]['time'][window_len:],
         (predict_train+1) * train['close'].values[:-window_len], 
         label='Predicted', color='red')
plt.show()


predict_test = yen_model.predict(test_lstm_in)
predict_test = predict_test.flatten() 

predict_test_four = []
threshold = 0.0001
print(threshold)

#print(predict_test)

for i  in range(len(predict_test)):
    if(predict_test[i] >= threshold):
        predict_test_four.append(0)
    elif(0<= predict_test[i] < threshold):
        predict_test_four.append(1)
    elif(-threshold <= predict_test[i] < 0):
        predict_test_four.append(2)
    elif(predict_test[i] < -threshold):
        predict_test_four.append(3)
predict_test_four = np.array(predict_test_four)
print(predict_test_four)

print(predict_test_four.shape)

test_out_four = []

#print(lstm_test_out)

for i  in range(len(lstm_test_out)):
    if(lstm_test_out[i] >= threshold):
        test_out_four.append(0)
    elif(0<= lstm_test_out[i] < threshold):
        test_out_four.append(1)
    elif(-threshold <= lstm_test_out[i] < 0):
        test_out_four.append(2)
    elif(lstm_test_out[i] < -threshold):
        test_out_four.append(3)
test_out_four = np.array(test_out_four)

print(test_out_four)
print(test_out_four.shape)
accuracy_count = 0

for i in range(len(test_out_four)):
    if(predict_test_four[i] == test_out_four[i]):
        accuracy_count += 1

accuracy = accuracy_count / len(test_out_four) * 100

print("accuracy is",accuracy, "%")


fig, ax1 = plt.subplots(1,1)
ax1.plot(df[df['time']>= split_date]['time'][window_len:],
         test['close'][window_len:], label='Actual', color='blue')
ax1.plot(df[df['time']>= split_date]['time'][window_len:],
         (predict_test +1) * test['close'].values[:-window_len], 
         label='Predicted', color='red')
ax1.grid(True)
plt.show()





predict_train = yen_model.predict(train_lstm_in)
predict_train = predict_train.flatten() 

predict_train_four = []

for i  in range(len(predict_train)):
    if(predict_train[i] >= threshold):
        predict_train_four.append(0)
    elif(0<= predict_train[i] < threshold):
        predict_train_four.append(1)
    elif(-threshold <= predict_train[i] < 0):
        predict_train_four.append(2)
    elif(predict_train[i] < -threshold):
        predict_train_four.append(3)
predict_train_four = np.array(predict_train_four)
print(predict_train)
print(predict_train_four)

print(predict_train_four.shape)

train_out_four = []

for i  in range(len(lstm_train_out)):
    if(lstm_train_out[i] >= threshold):
        train_out_four.append(0)
    elif(0<= lstm_train_out[i] < threshold):
        train_out_four.append(1)
    elif(-threshold <= lstm_train_out[i] < 0):
        train_out_four.append(2)
    elif(lstm_train_out[i] < -threshold):
        train_out_four.append(3)
train_out_four = np.array(train_out_four)

print(lstm_train_out)
print(train_out_four)
print(train_out_four.shape)
accuracy_count = 0

for i in range(len(train_out_four)):
    if(predict_train_four[i] == train_out_four[i]):
        accuracy_count += 1

accuracy = accuracy_count / len(train_out_four) * 100

print("accuracy is",accuracy, "%")


fig, ax1 = plt.subplots(1,1)
ax1.plot(df[df['time']< split_date]['time'][window_len:],
         train['close'][window_len:], label='Actual', color='blue')
ax1.plot(df[df['time']< split_date]['time'][window_len:],
         (predict_train+1) * train['close'].values[:-window_len], 
         label='Predicted', color='red')
plt.show()

"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~









#予測精度を見る２^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Lot = 0.1
Tsuka = 100000
yen = Lot * Tsuka 
#pal = pfofit and loss

predict_test = yen_model.predict(test_lstm_in)
predict_test = predict_test.flatten()
Totalpal = 0 
print(len(predict_test))

for i in range(len(predict_test)-window_len):
    print(i)
    print(Totalpal)
    if(predict_test[i] < 0):
        pal = (test['close'][:-window_len].values[i] - test['close'][:-window_len].values[i + window_len]) * yen
        Totalpal += pal
    elif(predict_test[i] >= 0):
        pal = (test['close'][:-window_len].values[i + window_len] - test['close'][:-window_len].values[i]) * yen
        Totalpal += pal
    
print("利益は",Totalpal,"円です。")








#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



"""
#１時間先を予測する----------------------------------------------------------------

#print(np.transpose(yen_model.predict(test_lstm_in))+1)

predict_seed = (np.transpose(yen_model.predict(test_lstm_in[-13:-1])+1) * test['close'].values[-(window_len + 13):-(1 + window_len)])[0]

#print(predict_seed)
predict_seed = np.reshape(predict_seed,(12,1))

predict_seed = np.array(predict_seed)
predict_len = 12

predict_price_list = [] #予想された変動価格
#predict_price_list.append(predict_seed)
#predict_price_list = np.array(predict_price_list)
predict = np.array([predict_seed])
#print(predict)
count = 0
for i in range(predict_len):
    count += 1
    #print(count)
    temp = predict.copy()
    #for col in predict:
    temp = temp/ temp[0][0] - 1
    #print(temp)
    predict_price = yen_model.predict(temp)
    #print('predict_price',predict_price)
    predict_price =  (predict_price+1) * predict[0][0]
    #print('predict_price',predict_price)
    predict_price_list = np.append(predict_price_list,predict_price)
    #print('predict_price_list',predict_price_list)
    predict = np.append(predict,predict_price)
    #print('predict',predict)
    predict = np.delete(predict, 0) 
    #print('predict',predict)
    predict = np.array([np.reshape(predict,(12,1))])
   # print('predict',predict)
    

print(predict_price_list)
"""
#-------------------------------------------------------------------------
"""


 # MAEをプロット
fig, ax1 = plt.subplots(1,1)

 
ax1.plot(yen_history.epoch, yen_history.history['loss'])
ax1.set_title('TrainingError')
 
if yen_model.loss == 'mae':
    ax1.set_ylabel('Mean Absolute Error (MAE)',fontsize=12)
else:
    ax1.set_ylabel('Model Loss',fontsize=12)
ax1.set_xlabel('# Epochs',fontsize=12)
plt.show()
 

#print((np.transpose(yen_model.predict(test_lstm_in))+1))
#print(test['close'].values[:-window_len])

# 訓練データから予測をして正解レートと予測したレートをプロット
fig, ax1 = plt.subplots(1,1)
ax1.plot(df[df['time']< split_date]['time'][window_len:],
         train['close'][window_len:], label='Actual', color='blue')
ax1.plot(df[df['time']< split_date]['time'][window_len:],
         ((np.transpose(yen_model.predict(train_lstm_in))+1) * train['close'].values[:-window_len])[0], 
         label='Predicted', color='red')
plt.show()


# テストデータを使って予測＆プロット
fig, ax1 = plt.subplots(1,1)
ax1.plot(df[df['time']>= split_date]['time'][window_len:],
         test['close'][window_len:], label='Actual', color='blue')
ax1.plot(df[df['time']>= split_date]['time'][window_len:],
         ((np.transpose(yen_model.predict(test_lstm_in))+1) * test['close'].values[:-window_len])[0], 
         label='Predicted', color='red')
ax1.grid(True)
plt.show()

"""
