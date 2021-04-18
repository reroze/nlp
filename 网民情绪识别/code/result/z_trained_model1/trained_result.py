import os
import pandas as pd
if __name__ == '__main__':
    pd_all = pd.read_csv('./test_results.tsv',sep='\t', header=None)
    pd_list = pd_all.values.tolist()
    #print(pd_list)
    predict = []
    for x in pd_list:
        predict.append(x.index(max(x)))
    #print(predict)
    #print(len(predict))
    #print(predict)
    for i in range(len(predict)):
        predict[i]-=1
    print(predict)
    test_file = '../../data/test.txt'
    f = open(test_file, 'r', encoding='utf-8')
    true = []
    for line in f.readlines():
        line = line.replace('\n', '').split('\t')
        true.append(int(line[1]))
    f.close()
    print(true)
    accur=0
    for i in range(len(true)):
        if(true[i]==predict[i]):
            accur+=1
    accur/=len(true)
    print('accur_rate:', accur)

