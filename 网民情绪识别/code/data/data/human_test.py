import random
def open_file(file):
    data = []
    file_op = open(file, 'r', encoding='utf-8')
    for line in file_op.readlines():
        line = line.replace('\n', '').split('\t')
        data.append([line[0], int(line[1])])
    file_op.close()
    return data


file = 'test.txt'
test_data = open_file(file)

list = random.sample(range(len(test_data)), 100)
count = 0
index = 0
for i in list:
    print('index', index)
    print('line', test_data[i][0])
    x = input('predict:')
    x = int(x)
    if(x==test_data[i][1]):
        count+=1
        print('good')
    print('true', test_data[i][1])
    index+=1
print('right_count', count)
