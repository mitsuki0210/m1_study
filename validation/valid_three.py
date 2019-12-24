#１時間先を予測する----------------------------------------------------------------

#print(np.transpose(yen_model.predict(test_lstm_in))+1)
def one_hour_predict():

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
    return predict_price_list

#-------------------------------------------------------------------------