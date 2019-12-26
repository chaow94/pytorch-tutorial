import os
import csv
import shutil

dict_ = {}

with open('train.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in list(spamreader)[1:]:
        print(row[0].split(",")[1])
        dirs = row[0].split(",")[1]
        file_name = row[0].split(",")[0]
        # print(', '.join(row))
        if not os.path.exists('train/' + dirs):
            os.makedirs('train/' + dirs)
        shutil.move('train/' + file_name, 'train/' + dirs + "/" + file_name)
        # print('train/' + file_name, 'train/' + dirs + "/" + file_name)

