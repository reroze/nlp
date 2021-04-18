
def clean(line):
    line = [x for x in line if (x >= u'\u4e00' and x<=u'\u9fa5')]
    return line


test_file = 'eval.txt'
zhongwen_test_file = 'zhongwen/z_eval.txt'



f = open(test_file, 'r', encoding='utf-8')
lines=[]
for line in f.readlines():
    line = line.replace('\n', '').split('\t')
    line[0]=clean(line[0])
    line[0]=''.join(line[0])
    lines.append(line)
f.close()
print(lines[2])
z_f = open(zhongwen_test_file, 'w', encoding='utf-8')
for l in lines:
    z_f.write(l[0])
    z_f.write('\t')
    z_f.write(str(l[1]))
    z_f.write('\n')
z_f.close()