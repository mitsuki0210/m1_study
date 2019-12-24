#予測精度を見る２^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


#Lot = 0.1
Tsuka = 10000
total_money = 1000000
first_total_money = 1000000
win_count = 0
lose_count = 0
#max_lot = total_money / ((current_late * Tsuka) / 25)
#pal = pfofit and loss
#ppp = Tsuka * 0.01 * max_lot


predict_test = yen_model.predict(test_lstm_in)
predict_test = predict_test.flatten()
Totalpal = 0 
max_win = 0
max_lose = 0
trade_count_list = []
total_money_list = []



for i in range(len(predict_test)-window_len):
    #print(i)
    #print(total_money)
    trade_count_list.append(i)
    max_lot = total_money / ((test['close'][:-window_len].values[i] * Tsuka) / 25)
    ppp = Tsuka * 0.01 * max_lot
    if(predict_test[i] < 0):
        pal = (test['close'][:-window_len].values[i] - test['close'][:-window_len].values[i + window_len]) * ppp
        if(pal >= 0):
            if(pal > max_win):
                max_win = pal
        elif(pal < 0):
            if(pal < max_lose):
                max_lose = pal
        total_money += pal
    elif(predict_test[i] >= 0):
        pal = (test['close'][:-window_len].values[i + window_len] - test['close'][:-window_len].values[i]) * ppp
        if(pal >= 0):
            if(pal > max_win):
                max_win = pal
        elif(pal < 0):
            if(pal < max_lose):
                max_lose = pal
        total_money += pal
    if(pal > 0):
        win_count += 1
    elif(pal < 0):
        lose_count += 1
    total_money_list.append(total_money)
        

up_count = 0
down_count = 0
no_change_count = 0

for i in predict_test:
    if(i > 0):
        up_count += 1
    elif(i < 0):
        down_count += 1
    elif(i == 0):
        no_change_count += 1
    

up_ratio = up_count / len(predict_test)
down_ratio = down_count / len(predict_test)
no_change_ratio = no_change_count / len(predict_test)



print("上げの内訳",up_ratio)
print("下げの内訳",down_ratio)
print("変化なしの内訳",no_change_ratio)
    


print("利益は",total_money,"円です。")
print("勝った回数は",win_count,"回です。")
print("負けた回数は",lose_count,"回です。")

print("勝率は",(win_count / (len(predict_test)-window_len)),"％です。")
print("負け率は",(lose_count / (len(predict_test)-window_len)),"％です。")

print("最大勝ちは",max_win,"円")
print("最大負けは",max_lose,"円")


print(len(trade_count_list))
print(len(total_money_list))

total_money_per = []

for val in total_money_list:
    per = val / first_total_money * 100
    total_money_per.append(per)

fig, ax1 = plt.subplots(1,1)
ax1.plot(trade_count_list, total_money_per)
#fig_money.set_title('Line Plot of Sine and Cosine Between -2\pi and 2\pi')
ax1.set_xlabel("Number of trades")
ax1.set_ylabel("Total assets increase rate (%)")
plt.show()

