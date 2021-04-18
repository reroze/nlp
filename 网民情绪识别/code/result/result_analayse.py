file = './nb_model_acc/result2.txt'

file_op = open(file, 'r', encoding='utf-8')

result = file_op.read().split('\t')
file_op.close()
print(result)
result = result[:4500]
for i in range(len(result)):
    result[i]=int(result[i])
print(result)
print(len(result))

class1=0
class2=0
class3=0
for x in result:
    if(x==-1):
        class1+=1
    if(x==0):
        class2+=1
    if(x==1):
        class3+=1
print('class1:', class1, 'class2:', class2, 'class3:', class3)