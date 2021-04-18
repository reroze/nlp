import random

def open_file(file):
    data = []
    file_op = open(file, 'r', encoding='utf-8')
    for line in file_op.readlines():
        line = line.replace('\n', '').split('\t')
        data.append(line)
    file_op.close()
    return data


def guolv(data, word):
    count = 0
    for i in range(len(data)):
        if (word in data[i][0]):
            count+=1
            #print(data[i][0])
            data[i][0]=data[i][0].replace(word, '')
            #print(data[i][0])
    print('word:', word, 'guolv:', count)

def write_data(data, file):
    file_op = open(file, 'w', encoding='utf-8')
    for line in data:
        file_op.write(line[0])
        file_op.write('\t')
        file_op.write(str(line[1]))
        file_op.write('\n')
    file_op.close()
    print('done')

def random_split(data, num):
    length = len(data)
    random_list = random.sample(range(length), num)
    data1=[]
    data2=[]
    for i in range(len(data)):
        if(i in random_list):
            data1.append(data[i])
        else:
            data2.append(data[i])
    return data1,data2



train_file = './new_data/new_cleaned_data/n_train.txt'
eval_file = './new_data/new_cleaned_data/n_eval.txt'
test_file = './new_data/new_cleaned_data/test.txt'

train_data = open_file(train_file)
eval_data = open_file(eval_file)
test_data = open_file(test_file)

'''a = 'abcd'
if 'cd' in a:
    print('hello')'''

guolv(train_data, "转发微博")#192
guolv(eval_data, "转发微博")#97
guolv(test_data, "转发微博")#30

guolv(train_data, "网页链接")#1228
guolv(eval_data, "网页链接")#651
guolv(test_data, "网页链接")#242

guolv(train_data, "展开全文")#5996
guolv(eval_data, "展开全文")#2927
guolv(test_data, "展开全文")#892

for i in range(len(test_data)):
    if('转发微博' in test_data[i][0]):
        print('wrong')

all_train_data = []

for data in train_data:
    all_train_data.append(data)

for data in eval_data:
    all_train_data.append(data)


less_train_file='./cleaned_data/less/train.txt'
less_eval_file='./cleaned_data/less/eval.txt'
less_test_file='./cleaned_data/less/test.txt'
write_data(train_data, less_train_file)
write_data(eval_data, less_eval_file)
write_data(test_data, less_test_file)



normal_train_file='./cleaned_data/normal/train.txt'
normal_eval_file='./cleaned_data/normal/eval.txt'
normal_test_file='./cleaned_data/normal/test.txt'

normal_train_data, normal_eval_data = random_split(all_train_data, 40000)
write_data(normal_train_data, normal_train_file)
write_data(normal_eval_data, normal_eval_file)
write_data(test_data, normal_test_file)


large_train_file='./cleaned_data/large/train.txt'
#large_eval_file='./cleaned_data/large/eval.txt'
large_test_file='./cleaned_data/large/test.txt'

large_train_data = all_train_data
write_data(large_train_data, large_train_file)
write_data(test_data, large_test_file)


for i in range(len(all_train_data)):
    if(all_train_data[i][1]!='0'):
        all_train_data[i][1]='1'
for i in range(len(test_data)):
    if(test_data[i][1]!='0'):
        test_data[i][1]='1'

_2_train_file='./cleaned_data/_2_/train.txt'
_2_eval_file='./cleaned_data/_2_/eval.txt'
_2_test_file='./cleaned_data/_2_/test.txt'

_2_train_data, _2_eval_data = random_split(all_train_data, 35000)
write_data(_2_train_data, _2_train_file)
write_data(_2_eval_data, _2_eval_file)
write_data(test_data, _2_test_file)






'''
guolv(train_data, "转发微博")
guolv(eval_data, "转发微博")
guolv(test_data, "转发微博")
'''
