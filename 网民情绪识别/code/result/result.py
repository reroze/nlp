import os
import pandas as pd

def class_test(pre_data, test_data):
    class1=0
    class2=0
    class3=0
    class1_pre=0
    class2_pre=0
    class3_pre=0

    for i in range(len(test_data)):
        if(test_data[i]==-1):
            class1+=1
            if(test_data[i]==pre_data[i]):
                class1_pre+=1
        if (test_data[i] == 0):
            class2 += 1
            if (test_data[i] == pre_data[i]):
                class2_pre += 1
        if (test_data[i] == 1):
            class3 += 1
            if (test_data[i] == pre_data[i]):
                class3_pre += 1
    class1_accr = class1_pre/class1
    class2_accr = class2_pre / class2
    class3_accr = class3_pre / class3
    print('class1_accur', class1_accr)
    print('class2_accur', class2_accr)
    print('class3_accur', class3_accr)

def predict_class_accur(predict_data, test_data):
    class1=0
    class2=0
    class3=0

    right_1=0
    right_2=0
    right_3=0
    for i in range(len(predict_data)):
        if(predict_data[i]==-1):
            class1+=1
        if(predict_data[i]==0):
            class2+=1
        if(predict_data[i]==1):
            class3+=1

        if(predict_data[i]==test_data[i]):
            if(predict_data[i]==-1):
                right_1+=1
            if(predict_data[i]==0):
                right_2+=1
            if(predict_data[i]==1):
                right_3+=1
    print('predict_class1_accur:', right_1/class1)
    print('predict_class2_accur:', right_2 / class2)
    print('predict_class3_accur:', right_3 / class3)

def class_sum(data):
    class1=0
    class2=0
    class3=0
    for i in range(len(data)):
        if(data[i]==-1):
            class1+=1
        if(data[i]==0):
            class2+=1
        if(data[i]==1):
            class3+=1
    print('class1:', class1)
    print('class2:', class2)
    print('class3:', class3)


if __name__ == '__main__':
    file = './cleaned_model/large/175/test_results.tsv'
    file_op = open(file, 'r', encoding='utf-8')
    prediction=[]
    for line in file_op.readlines():
       line = line.replace('\n', '').split('\t')
       prediction.append(line.index(max(line)))
    #print(len(prediction))
    #class_sum(prediction)

    for i in range(len(prediction)):
        prediction[i]-=1
    file_op.close()
    print(prediction)
    test_file = '../data/cleaned_data/less/test.txt'
    test_file_op = open(test_file, 'r', encoding='utf-8')
    test_data=[]
    for line in test_file_op.readlines():
        line = line.replace('\n', '').split('\t')
        test_data.append(int(line[1]))
    test_file_op.close()
    #print(test_data)
    accur = 0
    for i in range(len(test_data)):
        if(prediction[i]==test_data[i]):
            accur+=1
    print('accur', accur/len(test_data))
    class_test(prediction, test_data)
    predict_class_accur(prediction, test_data)
    class_sum(prediction)
    #class_sum(test_data)
