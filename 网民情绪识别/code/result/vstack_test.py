import numpy as np

a = np.array(
    [[1,2],
     [2,3]]
)
print(a)

b = np.array(
    [[4,5],
    [5,6]]
)
print(b)

c = np.array(
    [[7,8],
    [8,9]]
)
print(c)

d = [a,b,c]

e = np.vstack(x for x in d)
print(e)
f = list(np.argmax(e, axis=1))
print(f)