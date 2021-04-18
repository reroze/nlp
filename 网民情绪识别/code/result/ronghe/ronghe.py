def open_tsv(file):
    file_op = open(file, 'r', encoding='utf-8')
    data = []
    for line in file_op.readlines():
        line = line.replace('\n', '').split('\t')
        data.append(line.index(max(line))-1)
    file_op.close()
    return data

def good_neg_ronghe(accur_data, neg_data):
    count=0
    for i in range(len(accur_data)):
        if(accur_data[i]!=-1 and neg_data[i]==-1):
            accur_data[i]=-1


def accur_cal(prediction, test_data):
    accur=0
    for i in range(len(test_data)):
        if(prediction[i]==test_data[i]):
            accur+=1
    accur/=len(test_data)
    return accur




accur_file = 'good_accur/cleaned_data/test_results.tsv'
good_neg = 'good_-1/test_results.tsv'
test_file = '../../data/test.txt'

test_file_op = open(test_file, 'r', encoding='utf-8')
test_data=[]
for line in test_file_op.readlines():
    line = line.replace('\n', '').split('\t')
    test_data.append(int(line[1]))
test_file_op.close()

good_accur_data = open_tsv(accur_file)
good_neg_data = open_tsv(good_neg)

accur = accur_cal(good_accur_data, test_data)
print('accur', accur)

print(good_accur_data)
print(good_neg_data)
good_neg_ronghe(good_accur_data, good_neg_data)
accur = accur_cal(good_accur_data, test_data)
print('accur', accur)



