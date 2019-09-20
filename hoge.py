
#csvを開いて最初の行を取得するプログラム
csv_file = open("BBC_news.csv", "r", encoding="ms932", errors="", newline="" )
csv_read = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
for row in csv_read:
    print(csv_read.line_num)
    if csv_read.line_num  == 1:
        top_news_title = row[2]
        break
csv_file.close()
