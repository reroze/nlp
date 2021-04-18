def open_file(file):
    file_op = open(file, 'r', encoding='utf-8')
    data = []
    for line in file_op.readlines():
        line = line.replace('\n', '').split('\t')
        data.append(int(line[1]))
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


def matrix_create(predict, test):
    matrix = [[0,0,0],
              [0,0,0],
              [0,0,0]]
    for i in range(len(prediction)):
        if(prediction[i]==-1):
            if(test[i]==-1):
                matrix[0][0]+=1
            if(test[i]==0):
                matrix[0][1]+=1
            if(test[i]==1):
                matrix[0][2]+=1
        if (prediction[i] == 0):
            if (test[i] == -1):
                matrix[1][0] += 1
            if (test[i] == 0):
                matrix[1][1] += 1
            if (test[i] == 1):
                matrix[1][2] += 1
        if (prediction[i] == 1):
            if (test[i] == -1):
                matrix[2][0] += 1
            if (test[i] == 0):
                matrix[2][1] += 1
            if (test[i] == 1):
                matrix[2][2] += 1
    return matrix

def open_paddle_txt(file):
    file_op = open(file, 'r', encoding='utf-8')
    data = []
    data = file_op.read().replace('\n', '').split('\t')
    data = data[:4500]
    for i in range(len(data)):
        data[i]=int(data[i])
    file_op.close()
    return data
'''
def precision_score(matrix):
    class1_score = 0
    class2_score = 0
    class3_score = 0

    class1_score = matrix[0][0]/(matrix[0][0]+matrix[0][1]+matrix[0][2])
    class2_score = matrix[1][1] / (matrix[1][0] + matrix[1][1] + matrix[1][2])
    class3_score = matrix[2][2] / (matrix[2][0] + matrix[2][1] + matrix[2][2])

    return class1_score, class2_score, class3_score

def recall_score(matrix):
    class1_score = 0
    class2_score = 0
    class3_score = 0

    class1_score = matrix[0][0]/(matrix[0][0]+matrix[1][0]+matrix[2][0])
    class2_score = matrix[1][1] / (matrix[0][1] + matrix[1][1] + matrix[2][1])
    class3_score = matrix[2][2] / (matrix[0][2] + matrix[1][2] + matrix[2][2])

    return class1_score, class2_score, class3_score
'''

def TP_class(matrix, class_index):
    return matrix[class_index][class_index]

def FP_class(matrix, class_index):
    fp = 0
    for i in range(len(matrix[class_index])):
        if(i!=class_index):
            fp+=matrix[class_index][i]
    return fp
def FN_class(matrix, class_index):
    fn=0
    for i in range(len(matrix)):
        if(i!=class_index):
            fn+=matrix[i][class_index]
    return fn

def F1_class(TP, FP, FN):
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = 2*precision*recall/(precision+recall)
    return f1

def write_score(score_file, matrix):
    score_file_op = open(score_file, 'w', encoding='utf-8')
    score_file_op.write(score_file)
    score_file_op.write('\n')
    score_file_op.write('matrix:\n')
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            score_file_op.write(str(matrix[i][j]))
            if(j!=len(matrix[i])-1):
                score_file_op.write('\t')
            else:
                score_file_op.write('\n')
    tp_class1 = TP_class(matrix, 0)
    tp_class2 = TP_class(matrix, 1)
    tp_class3 = TP_class(matrix, 2)

    fp_class1 = FP_class(matrix, 0)
    fp_class2 = FP_class(matrix, 1)
    fp_class3 = FP_class(matrix, 2)

    fn_class1 = FN_class(matrix, 0)
    fn_class2 = FN_class(matrix, 1)
    fn_class3 = FN_class(matrix, 2)

    precision_class1 = tp_class1 / (tp_class1 + fp_class1)
    precision_class2 = tp_class2 / (tp_class2 + fp_class2)
    precision_class3 = tp_class3 / (tp_class3 + fp_class3)

    recall_class1 = tp_class1 / (tp_class1 + fn_class1)
    recall_class2 = tp_class2 / (tp_class2 + fn_class2)
    recall_class3 = tp_class3 / (tp_class3 + fn_class3)

    all_precision = (tp_class1 + tp_class2 + tp_class3) / (tp_class1 + tp_class2 + tp_class3 + fp_class1 + fp_class2 + fp_class3)
    all_recall = (tp_class1 + tp_class2 + tp_class3) / (tp_class1 + tp_class2 + tp_class3 + fn_class1 + fn_class2 + fn_class3)
    all_F1 = 2*all_precision*all_recall/(all_precision+all_recall)

    macro_p = (precision_class1 + precision_class2 + precision_class3) / 3
    macro_r = (recall_class1 + recall_class2 + recall_class3) / 3
    macro_f1 = 2 * macro_p * macro_r / (macro_p + macro_r)

    score_file_op.write('precision_class1:\t')
    score_file_op.write(str(precision_class1))
    score_file_op.write('\n')
    score_file_op.write('precision_class2:\t')
    score_file_op.write(str(precision_class2))
    score_file_op.write('\n')
    score_file_op.write('precision_class3:\t')
    score_file_op.write(str(precision_class3))
    score_file_op.write('\n')

    score_file_op.write('recall_class1:\t')
    score_file_op.write(str(recall_class1))
    score_file_op.write('\n')

    score_file_op.write('recall_class2:\t')
    score_file_op.write(str(recall_class2))
    score_file_op.write('\n')

    score_file_op.write('recall_class3:\t')
    score_file_op.write(str(recall_class3))
    score_file_op.write('\n')

    score_file_op.write('micro_F1:\t')
    score_file_op.write(str(all_F1))
    score_file_op.write('\n')

    score_file_op.write('macro_F1:\t')
    score_file_op.write(str(macro_f1))
    score_file_op.write('\n')
    score_file_op.close()




predict_dir = './cleaned_model/large/175'
predict_file = '/'.join([predict_dir, 'test_results.tsv'])
#predict_file = '/'.join([predict_dir, 'result2.txt'])
score_file = '/'.join([predict_dir, 'score.txt'])
test_file = '../data/cleaned_data/less/test.txt'

prediction = open_tsv(predict_file)
#prediction = open_paddle_txt(predict_file)
test = open_file(test_file)
matrix=matrix_create(prediction, test)
print(matrix)

#precision_class1, precision_class2, precision_class3 = precision_score(matrix)
#print('precision_class1', precision_class1, 'precision_class2', precision_class2, 'precision_class3', precision_class3)

#recall_class1, recall_class2, recall_class3 = recall_score(matrix)
#print('recall_class1', recall_class1, 'recall_class2', recall_class2, 'recall_class3', recall_class3)

tp_class1 = TP_class(matrix, 0)
tp_class2 = TP_class(matrix, 1)
tp_class3 = TP_class(matrix, 2)
#print(tp_class3)
fp_class1 = FP_class(matrix, 0)
fp_class2 = FP_class(matrix, 1)
fp_class3 = FP_class(matrix, 2)
#print(fp_class3)
fn_class1 = FN_class(matrix, 0)
fn_class2 = FN_class(matrix, 1)
fn_class3 = FN_class(matrix, 2)
#print(fn_class3)
all_precision = (tp_class1+tp_class2+tp_class3)/(tp_class1+tp_class2+tp_class3+fp_class1+fp_class2+fp_class3)
#print(all_precision)
all_recall = (tp_class1+tp_class2+tp_class3)/(tp_class1+tp_class2+tp_class3+fn_class1+fn_class2+fn_class3)
#print(all_recall)

all_F1 = 2*all_precision*all_recall/(all_precision+all_recall)
print('all_F1', all_F1)


#f1_class1 = F1_class(tp_class1, fp_class1, fn_class1)
#f1_class2 = F1_class(tp_class2, fp_class2, fn_class2)
#f1_class3 = F1_class(tp_class3, fp_class3, fn_class3)
#print('f1_class1', f1_class1, 'f1_class2', f1_class2, 'f1_class3', f1_class3)
#F1 = (f1_class1+f1_class2+f1_class3)/3
#print('F1:', F1)

precision_class1 = tp_class1/(tp_class1+fp_class1)
precision_class2 = tp_class2/(tp_class2+fp_class2)
precision_class3 = tp_class3/(tp_class3+fp_class3)

recall_class1 = tp_class1/(tp_class1+fn_class1)
recall_class2 = tp_class2/(tp_class2+fn_class2)
recall_class3 = tp_class3/(tp_class3+fn_class3)

macro_p = (precision_class1+precision_class2+precision_class3)/3
macro_r = (recall_class1+recall_class2+recall_class3)/3
macro_f1 = 2*macro_p*macro_r/(macro_p+macro_r)
print('macro_f1', macro_f1)
write_score(score_file, matrix)





#print(prediction)
#print(test)