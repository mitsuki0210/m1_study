#予測精度を見る２^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#Lot = 0.1
Tsuka = 10000
total_money = 1000000
max_lot = total_money / ((current_late * Tsuka) / 25)
#pal = pfofit and loss
ppp = Tsuka * 0.01 * max_lot



predict_test = yen_model.predict(test_lstm_in)
predict_test = predict_test.flatten()
Totalpal = 0 
print(len(predict_test))

for i in range(len(predict_test)-window_len):
    print(i)
    print(total_money)
    max_lot = total_money / ((test['close'][:-window_len].values[i] * Tsuka) / 25)
    ppp = Tsuka * 0.01 * max_lot
    if(predict_test[i] < 0):
        pal = (test['close'][:-window_len].values[i] - test['close'][:-window_len].values[i + window_len]) * ppp
        total_money += pal
    elif(predict_test[i] >= 0):
        pal = (test['close'][:-window_len].values[i + window_len] - test['close'][:-window_len].values[i]) * ppp
        total_money += pal
    
print("利益は",total_money,"円です。")
