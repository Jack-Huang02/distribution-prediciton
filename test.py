import math

x = [0.1, 12, 4, 56, 3]
y = [12.2, 3, 9.3, 39, 19]
c = [1.7, 4, 2, 24, 32.4]
for i in range(len(x)):
    c[i] = math.cos(-c[i])
