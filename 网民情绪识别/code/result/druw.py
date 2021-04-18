import numpy as np
import matplotlib.pyplot as plt
'''

plt.plot(x1, y1, label='weight changes', linewidth=3, color='r', marker='o',
         markerfacecolor='blue', markersize=20)

'''
'''
for a, b in zip(x1, y1):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=20)

'''
x = [60, 90, 120, 150]
y1 = [0.603, 0.613, 0.612, 0.620]
plt.title('micro_f1-steps model:albert_tiny')
plt.plot(x, y1, label='weight changes', linewidth=2, color='b', marker='o', markerfacecolor='red', markersize=5)
plt.xlabel('steps')
plt.ylabel('micro_f1')
plt.ylim(0, 1)
plt.xlim(0, 200)

for a,b in zip(x, y1):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=6)

plt.show()
