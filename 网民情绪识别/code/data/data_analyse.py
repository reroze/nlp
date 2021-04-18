#1.可以适当加长最大长度
#2.可以适当减少训练的数据量，比如将eval设置为10000个



def read_date(file):
    file_op = open(file, 'r', encoding='utf-8')
    data = []
    index=0
    for line in file_op.readlines():
        line = line.replace('\n', '').split('\t')
        if(line[1]  not in ['-1', '0', '1']):
            line[1]=0
            print('wrong', index)
        else:
            line[1] = int(line[1])
        data.append(line)
        index+=1
    file_op.close()
    return data

def class_analyse(data):
    class1=0
    class2=0
    class3=0
    for x in data:
        if(x[1]==-1):
            class1+=1
        if(x[1]==0):
            class2+=1
        if(x[1]==1):
            class3+=1
    print('class1:', class1,'\tclass2:', class2, '\tclass3:', class3)

def get_maxsentence_length(data):
    max_length=0
    for x in data:
        if(len(x[0])>max_length):
            max_length=len(x[0])
    print('max_length', max_length)

def length_sum(data, num):
    count = 0
    for x in data:
        if(len(x[0])>num):
            count += 1
    print('count', count)

def length_sum_low(data, num):
    count = 0
    for x in data:
        if (len(x[0]) < num):
            count += 1
    print('count', count)


def random_test(data):
    num = int(len(data)/5)
    for i in range(5):
        class1 = 0
        class2 = 0
        class3 = 0
        for j in range(num*i, num*(i+1)):
            if(data[j][1]==-1):
                class1+=1
            if(data[j][1]==0):
                class2 += 1
            if(data[j][1]==1):
                class3+=1
        print('i:', i, 'class1:', class1, 'class2:', class2, 'class3', class3)




train_file = './new_data/train_ dataset/cleaned_train_data.txt'
#test_file = './zhongwen/z_test.txt'
#eval_file = './zhongwen/z_eval.txt'

'''
train_file_op = open(train_file, 'r', encoding='utf-8')
train_data = []
for line in train_file_op.readlines():
    line = line.replace('\n', '').split('\t')
    train_data.append(line)'''
train_data = read_date(train_file)
#eval_data = read_date(eval_file)
#test_data = read_date(test_file)

#print(train_data[0])
#print(len(train_data))#40000
print(len(train_data))#
#print(len(eval_data))#5000
#print(len(test_data))#4500

print('train')
#class_analyse(train_data)#class1: 13330 	class2: 13351 	class3: 13319 ++->40000
#class_analyse(eval_data)#class1: 1670 	class2: 1649 	class3: 1681 ++->5000
#class_analyse(test_data)#class1: 1500 	class2: 1500 	class3: 1500 ++->4500

class_analyse(train_data)#class1: 13330 	class2: 13351 	class3: 13319 ++->40000

#get_maxsentence_length(train_data)#126
#get_maxsentence_length(eval_data)#122
#get_maxsentence_length(test_data)#136

get_maxsentence_length(train_data)#126

#length_sum(train_data, 70)# 10631
#length_sum(eval_data, 70)# 1359
#length_sum(test_data, 70)# 1148

#length_sum(train_data, 100)#1981
#length_sum(eval_data, 100)#243
#length_sum(test_data, 100)#200

#length_sum_low(train_data, 10)#3674
#length_sum_low(train_data, 5)#1375
#length_sum_low(train_data, 1)#103
#length_sum_low(train_data, 4)#831
#length_sum_low(train_data, 3)#539
#length_sum_low(train_data, 2)#255
#length_sum_low(train_data, 1)#103
#length_sum_low(eval_data, 1)#8
#length_sum_low(test_data, 1)#10

#length_sum_low(test_data, 4)#<4不包括4 85

#random_test(train_data)
'''
i: 0 class1: 2620 class2: 2736 class3 2644
i: 1 class1: 2666 class2: 2630 class3 2704
i: 2 class1: 2595 class2: 2705 class3 2700
i: 3 class1: 2730 class2: 2634 class3 2636
i: 4 class1: 2719 class2: 2646 class3 2635
'''
#random_test(eval_data)
'''
i: 0 class1: 337 class2: 315 class3 348
i: 1 class1: 326 class2: 344 class3 330
i: 2 class1: 335 class2: 320 class3 345
i: 3 class1: 330 class2: 348 class3 322
i: 4 class1: 342 class2: 322 class3 336
'''



