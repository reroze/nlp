import os
import pandas as pd
import random


file_train = '../../data/train.csv'
file_test = '../../data/test_labled.csv'
stopword_file = '../../data/中文停用词表2.txt'

train_txt = '../../data/new_data/train.txt'
eval_txt = '../../data/new_data/eval.txt'
test_txt = '../../data/new_data/test.txt'


def clean(line):
    line = [x for x in line if (x >= u'\u4e00' and x<=u'\u9fa5')]
    return line

def text_guolv(data):#使用不分词的版本
    stopword_file_op = open(stopword_file, 'r', encoding='utf-8')
    stopword = stopword_file_op.readlines()
    stopword = [x.strip('\n') for x in stopword]
    #print(stopword)
    train_data = []
    text=[]
    for i in range(len(data)):
        text_single = []
        for j in range(len(data[i][0])):
            text_single.append(data[i][0][j])
        text_single = [x for x in text_single if x not in stopword]
        text_single = clean(text_single)
        text = ''.join(text_single)
        train_data.append([text, str(data[i][1])])
    #print(train_data[0])
    return train_data
    #a = 'abcdef'
    #print(a[5])


def file_write(file, data):
    file_op = open(file, 'w', encoding='utf-8')
    for i in range(len(data)):
        file_op.write(data[i][0])
        file_op.write('\t')
        file_op.write(data[i][1])
        file_op.write('\n')
    file_op.close()


if __name__ == '__main__':
    train_data = pd.read_csv(file_train)
    #print(train_data)
    #print(train_data.head())
    train_list = train_data.values.tolist()
    train_data=text_guolv(train_list)
    eval_data=[]
    split_train_data = []
    number_list = random.sample(range(45000), 5000)
    for i in range(len(train_data)):
        if(i in number_list):
            eval_data.append(train_data[i])
        else:
            split_train_data.append(train_data[i])
    test_data = pd.read_csv(file_test)
    test_list = test_data.values.tolist()
    test_data = text_guolv(test_list)

    file_write(train_txt, split_train_data)
    file_write(eval_txt, eval_data)
    file_write(test_txt, test_data)
    print('done')
    #print(len(eval_data))#5000
    #print(eval_data[0])
    #print(len(split_train_data))#40000
    #print(split_train_data[0])
    #print(len(test_data))#4500
    #print(test_data[0])


    #train_list = []
    #print(train_list[10000])
    #print(len(train_list))#45000 选5000作为dev验证集

