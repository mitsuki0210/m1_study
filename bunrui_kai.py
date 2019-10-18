import numpy as np
import re
from data_append import make_data
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM, Embedding, SimpleRNN
from keras.layers import Dropout
from keras import metrics
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras import metrics
from keras.models import model_from_json



csv_data = "BBC_news.csv"


data = make_data(csv_data)

print("データ数",len(data))


train_data = data[:170]
test_data = data[170:]
max_len = 2000

print("trainデータ数",len(train_data))

print("testデータ数",len(test_data))



# テキストから不要な要素を取り除く
def convert_sentence_to_words(sentence):
    stopwords = ["i", "a", "an", "the", "and", "or", "if", "is", "are", "am", "it", "this", "that", "of", "from", "in", "on"]
    sentence = sentence.lower() # 小文字化
    sentence = sentence.replace("\n", "") # 改行削除
    sentence = re.sub(re.compile("[!-\/:-@[-`{-~]"), " ", sentence) # 記号をスペースに置き換える
    sentence = sentence.split(" ") # スペースで区切る
    sentence_words = []
    for word in sentence:
        if (re.compile(r"^.*[0-9]+.*$").fullmatch(word) is not None):
            continue
        if word in stopwords:
            continue
        sentence_words.append(word)
    return sentence_words

# 単語をIDに変換する単語辞書を生成する
def create_word_id_dict(sentence_list):
    word_to_id = {}
    for sentence in sentence_list:
        sentence_words = convert_sentence_to_words(sentence)
        for word in sentence_words:
            if word not in word_to_id:
                word_to_id[word] = len(word_to_id)
    return word_to_id

# 文章をID列に変換する
def convert_sentences_to_ids(sentence_list, word_to_id):
    sentence_id_vec = []
    for sentence in sentence_list:
        sentence_words = convert_sentence_to_words(sentence)
        sentence_ids = []
        for word in sentence_words:
            if word in word_to_id:
                sentence_ids.append(word_to_id[word])
            else:
                sentence_ids.append(0)
        sentence_id_vec.append(sentence_ids)
    return sentence_id_vec

# 文章はそれぞれ長さが違うので、前パディングして固定長の系列にする
# 前パディングなのは、RNNの構造上、後ろにあるデータのほうが反映されやすいため
def padding_sentence(sentence_id_vec):
    max_sentence_size = 0
    for sentence_vec in sentence_id_vec:
        if max_sentence_size < len(sentence_vec):
            max_sentence_size = len(sentence_vec)
    for sentence_ids in sentence_id_vec:
        while len(sentence_ids) < max_sentence_size:
            sentence_ids.insert(0,0)
    return sentence_id_vec



train_sentence_list = [s[0] for s in train_data]

#print(train_sentence_list)


word_to_id = create_word_id_dict(train_sentence_list)


sentence_id_vec = convert_sentences_to_ids(train_sentence_list,word_to_id)
sentence_id_vec = pad_sequences(sentence_id_vec, maxlen=max_len)
#sentence_id_vec = padding_sentence(sentence_id_vec)



# Numpy配列に変換する必要がある
x = sentence_id_vec
t = np.array([label[1] for label in train_data])




vocab_size = len(word_to_id)
embed_size = 100
hidden_size = 200
out_size = 4
lr = 1e-3
batch_size = 10
max_epochs = 50
data_size = len(train_data)
#max_iters = data_size // batch_size

#print("max_iters",max_iters)
print(vocab_size)


print(x)
print(x.shape)
#print(test_lstm_in)



t = np_utils.to_categorical(t, 4)
t = t.astype(np.int64)
print(t)
print(t.shape)



def build_model(inputs, output_size, neurons, activ_func="softmax", loss="mae", optimizer="adam"):
    model = Sequential()
    model.add(Embedding(vocab_size, embed_size, batch_input_shape = (batch_size,len(inputs[1]))))

    #model.add(SimpleRNN(hidden_size, return_sequences=False))
    model.add(LSTM((hidden_size), stateful=True))
    #model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer)
    return model


# ランダムシードの設定
#np.random.seed(202)



# 初期モデルの構築
yen_model = build_model(x, output_size=4, neurons = 20)

#print(train_lstm_in.shape)
yen_model.summary()






# データを流してフィッティングさせましょう
yen_history = yen_model.fit(x, t, epochs=50, batch_size=10, verbose=2, shuffle=True)


model_arc_json = yen_model.to_json()
open("news_bunrui.json", mode='w').write(model_arc_json)

# 学習済みの重みを保存
yen_model.save_weights('news_bunrui_weight.hdf5')


MODEL_ARC_PATH = 'news_bunrui.json'
WEIGHTS_PATH = 'news_bunrui_weight.hdf5'



model_arc_str = open(MODEL_ARC_PATH).read()
yen_model = model_from_json(model_arc_str)


# 重みの読み込み
yen_model.load_weights(WEIGHTS_PATH)





test_sentence_list = [s[0] for s in test_data]
test_sentence_id_vec = convert_sentences_to_ids(test_sentence_list, word_to_id)
#test_sentence_id_vec = padding_sentence(test_sentence_id_vec)
test_sentence_id_vec = pad_sequences(test_sentence_id_vec, maxlen=max_len)




"""
for test_sentence_id in test_sentence_id_vec:
    #print(type(test_sentense_id))
    #print(type(np.array(test_sentence_id)))
    query = np.array(test_sentence_id).reshape(1,max_len)
    print(query)
    result = yen_model.predict(query)
    print(result)
    print(result.argmax())
    test_pred_list.append(result.argmax())
"""


test_x = test_sentence_id_vec


pred = yen_model.predict(test_x,batch_size=10,verbose=2)
print(pred)
print(type(pred))






y_true = np.array([label[1] for label in test_data])
y_true = np_utils.to_categorical(y_true, 4)
y_true = y_true.astype(np.int64)



correct = 0
semi_correct = 0

for i in range(len(pred)):
    print(pred[i])
    pred_ = np.argmax(pred[i])
    print(pred_)
    tar_ = np.argmax(y_true[i])
    print(tar_)
    if pred_ == tar_ :
        correct += 1
    else :
        if pred_+tar_ == 1 or pred_+tar_ == 5 :
            semi_correct += 1

print("正答率:", 1.0 * correct / len(pred))


#print(accuracy_score(y_true, test_pred))
    

