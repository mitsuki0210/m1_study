import csv

list_ = []
count = 0
with open('BBC_news.csv', 'r', newline='', encoding='utf-8') as f:
    r = csv.reader(f)  # CSVファイルを読み込んでReaderオブジェクトを生成
    for l in r:
        count += 1
        print(count)
        list_.append(l)

        
