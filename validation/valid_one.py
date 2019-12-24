#予測精度をみる１~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print(test_lstm_in.shape)

predict_test = yen_model.predict(test_lstm_in)
print(predict_test.shape)

print("predict_test_close",predict_test[:,0])

#predict_test = predict_test.flatten() 
predict_test = predict_test[:,0]

predict_test_bi = []


for i  in range(len(predict_test)):
    if(predict_test[i] - test_lstm_in[i][-1][0]  >=0):
        predict_test_bi.append(1)
    elif(predict_test[i] - test_lstm_in[i][-1][0] < 0):
        predict_test_bi.append(0)
predict_test_bi = np.array(predict_test_bi)
print("predict_test_bi",predict_test_bi)

print("predict_test_bi.shape",predict_test_bi.shape)


test_out_bi = []

for i  in range(len(lstm_test_out)):
    if(lstm_test_out[i][0] - test_lstm_in[i][-1][0] >=0):
        test_out_bi.append(1)
    elif(lstm_test_out[i][0] - test_lstm_in[i][-1][0] < 0):
        test_out_bi.append(0)
test_out_bi = np.array(test_out_bi)

print("test_out_bi",test_out_bi)
print("test_out_bi.shape",test_out_bi.shape)
accuracy_count = 0

for i in range(len(test_out_bi)):
    if(predict_test_bi[i] == test_out_bi[i]):
        accuracy_count += 1

accuracy = accuracy_count / len(test_out_bi) * 100

test_bi_up_count = 0
test_bi_down_count = 0

for i in test_out_bi:
    if(i == 1):
        test_bi_up_count += 1
    elif(i == 0):
        test_bi_down_count += 1
    
print("テストデータの上げのデータ数",test_bi_up_count,"個")
print("テストデータの下げのデータ数",test_bi_down_count,"個")




print("binarry_testdata_accuracy is",accuracy, "%")



predict_train = yen_model.predict(train_lstm_in)
predict_train = predict_train[:,0]

#predict_train = predict_train.flatten() 

predict_train_bi = []   

for i  in range(len(predict_train)):
    if(predict_train[i] - train_lstm_in[i][-1][0] >=0):
        predict_train_bi.append(1)
    elif(predict_train[i] - train_lstm_in[i][-1][0] < 0):
        predict_train_bi.append(0)
predict_train_bi = np.array(predict_train_bi)
print("predict_train_bi",predict_train_bi)

print("predict_train_bi.shape",predict_train_bi.shape)

train_out_bi = []

for i  in range(len(lstm_train_out)):
    if(lstm_train_out[i][0] - train_lstm_in[i][-1][0] >=0):
        train_out_bi.append(1)
    elif(lstm_train_out[i][0] - train_lstm_in[i][-1][0] < 0):
        train_out_bi.append(0)
train_out_bi = np.array(train_out_bi)

print("train_out_bi",train_out_bi)
print("train_out_bi.shape",train_out_bi.shape)
accuracy_count = 0

for i in range(len(train_out_bi)):
    if(predict_train_bi[i] == train_out_bi[i]):
        accuracy_count += 1

accuracy = accuracy_count / len(train_out_bi) * 100


train_bi_up_count = 0
train_bi_down_count = 0

for i in train_out_bi:
    if(i == 1):
        train_bi_up_count += 1
    elif(i == 0):
        train_bi_down_count += 1
    
print("訓練データの上げのデータ数",train_bi_up_count,"個")
print("訓練データの下げのデータ数",train_bi_down_count,"個")

print("binarry_traindata_accuracy is",accuracy, "%")


predict_train_bi_up_count = 0
predict_train_bi_down_count = 0


for i in predict_train_bi:
    if(i == 1):
        predict_train_bi_up_count += 1
    elif(i == 0):
        predict_train_bi_down_count += 1

"""
predict_test_four = []
threshold = 0.0005 #閾値
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
print("predict_test_four",predict_test_four)

print("predict_test_four.shape",predict_test_four.shape)

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

print("test_out_four",test_out_four)
print("test_out_four.shape",test_out_four.shape)
accuracy_count = 0

for i in range(len(test_out_four)):
    if(predict_test_four[i] == test_out_four[i]):
        accuracy_count += 1

accuracy = accuracy_count / len(test_out_four) * 100


test_0_count = 0
test_1_count = 0
test_2_count = 0
test_3_count = 0



for i in test_out_four:
    if(i == 0):
        test_0_count += 1
    elif(i == 1):
        test_1_count += 1
    elif(i == 2):
        test_2_count += 1
    elif(i == 3):
        test_3_count += 1
    
print("テストデータの0のデータ数",test_0_count,"個")
print("テストデータの1のデータ数",test_1_count,"個")
print("テストデータの2のデータ数",test_2_count,"個")
print("テストデータの3のデータ数",test_3_count,"個")


print("four_testdata_accuracy is",accuracy, "%")


fig, ax1 = plt.subplots(1,1)
ax1.plot(df[df['time']>= split_date]['time'][window_len:],
         test['close'][window_len:], label='Actual', color='blue')
ax1.plot(df[df['time']>= split_date]['time'][window_len:],
         (predict_test +1) * test['close'].values[:-window_len], 
         label='Predicted', color='red')
ax1.grid(True)
plt.show()


#predict_train = yen_model.predict(train_lstm_in)
#predict_train = predict_train.flatten() 

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
print("predict_train",predict_train)
print("predict_train_four",predict_train_four)

print("predict_train_four.shape",predict_train_four.shape)

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

print("lstm_train_out",lstm_train_out)
print("train_out_four",train_out_four)
print("train_out_four.shape",train_out_four.shape)
accuracy_count = 0

for i in range(len(train_out_four)):
    if(predict_train_four[i] == train_out_four[i]):
        accuracy_count += 1

accuracy = accuracy_count / len(train_out_four) * 100

train_0_count = 0
train_1_count = 0
train_2_count = 0
train_3_count = 0



for i in train_out_four:
    if(i == 0):
        train_0_count += 1
    elif(i == 1):
        train_1_count += 1
    elif(i == 2):
        train_2_count += 1
    elif(i == 3):
        train_3_count += 1
    
print("訓練データの0のデータ数",train_0_count,"個")
print("訓練データの1のデータ数",train_1_count,"個")
print("訓練データの2のデータ数",train_2_count,"個")
print("訓練データの3のデータ数",train_3_count,"個")

print("four_traindata_accuracy is",accuracy, "%")

fig, ax1 = plt.subplots(1,1)
ax1.plot(df[df['time']< split_date]['time'][window_len:],
         train['close'][window_len:], label='Actual', color='blue')
ax1.plot(df[df['time']< split_date]['time'][window_len:],
         (predict_train+1) * train['close'].values[:-window_len], 
         label='Predicted', color='red')
plt.show()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~