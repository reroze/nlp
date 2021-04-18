def open_file(file):
    file_op = open(file, 'r', encoding='utf-8')
    data = []
    for line in file_op.readlines():
        line = line.replace('\n', '').split('\t')
        data.append([line[0], int(line[1])])
    file_op.close()
    return data




def open_tsv(tsv):
    tsv_op = open(tsv, 'r', encoding='utf-8')
    prediction = []
    for line in tsv_op.readlines():
        line = line.replace('\n', '').split('\t')
        prediction.append(line.index(max(line)))
    # print(len(prediction))
    # class_sum(prediction)

    for i in range(len(prediction)):
        prediction[i] -= 1
    tsv_op.close()
    return prediction

def check(prediction, test_data):
    list = []
    for i in range(len(prediction)):
        if(prediction[i]!=test_data[i][1]):
            list.append(i)
    return list

test_file = '../data/cleaned_data/less/test.txt'
predict_file = './cleaned_data2/less/117/test_results.tsv'

prediction = open_tsv(predict_file)
test_data = open_file(test_file)

wrong_list = check(prediction, test_data)
print(wrong_list)
print(len(wrong_list))#1686
for i in wrong_list:
    print(test_data[i][0], prediction[i], test_data[i][1])

