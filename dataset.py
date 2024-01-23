import csv
from matplotlib import pyplot as plt
import math

def getdata(width, velocity):
    filepath = 'data/total_mouse.csv'
    f = open(filepath)
    reader = csv.reader(f)
    header_row = next(reader)
    endpoint_x = []
    endpoint_y = []
    rectLeft = []
    rectTop = []
    targetWidth = []
    directionAngle = []
    for row in reader:
        if int(row[8]) < 1000 and int(row[3]) == velocity and int(row[2]) == width:
            endpoint_x.append(float((row[8])))
            endpoint_y.append(float(row[9]))
            rectLeft.append(float(row[6]))
            rectTop.append(float(row[7]))
            targetWidth.append(float(row[2]))
            directionAngle.append(float(row[4]))
    length = len(endpoint_x)
    for i in range(length):
        endpoint_x[i] = endpoint_x[i] - (rectLeft[i] + targetWidth[i] / 2)
        endpoint_y[i] = -(endpoint_y[i] - (rectTop[i] + targetWidth[i] / 2))
        directionAngle[i] = -directionAngle[i]
    X = []
    Y = []
    for i in range(length):
        X.append(endpoint_x[i] * math.cos(-directionAngle[i]) - endpoint_y[i] * math.sin(-directionAngle[i]))
        Y.append(endpoint_x[i] * math.cos(-directionAngle[i]) + endpoint_y[i] * math.sin(-directionAngle[i]))
    f.close()
    return X, Y

# width = 16 velocity = 64
X, Y = getdata(16, 64)
plt.scatter(X, Y, s=10)
plt.subplot(4, 4, 1)
plt.title("plot 1")

# width = 32 velocity = 64
X, Y = getdata(32, 64)
plt.scatter(X, Y, s=10)
plt.subplot(4, 4, 2)
plt.title("plot 2")

# width = 64 velocity = 64
X, Y = getdata(64, 64)
plt.scatter(X, Y, s=10)
plt.subplot(4, 4, 3)
plt.title("plot 3")

# width = 96 velocity = 64
X, Y = getdata(96, 64)
plt.scatter(X, Y, s=10)
plt.subplot(4, 4, 4)
plt.title("plot 4")

# width = 16 velocity = 128
X, Y = getdata(16, 128)
plt.scatter(X, Y, s=10)
plt.subplot(4, 4, 5)
plt.title("plot 5")

# width = 32 velocity = 128
X, Y = getdata(32, 128)
plt.scatter(X, Y, s=10)
plt.subplot(4, 4, 6)
plt.title("plot 6")

# width = 64 velocity = 128
X, Y = getdata(64, 128)
plt.scatter(X, Y, s=10)
plt.subplot(4, 4, 7)
plt.title("plot 7")

# width = 96 velocity = 128
X, Y = getdata(96, 128)
plt.scatter(X, Y, s=10)
plt.subplot(4, 4, 8)
plt.title("plot 8")

# width = 16 velocity = 192
X, Y = getdata(16, 192)
plt.scatter(X, Y, s=10)
plt.subplot(4, 4, 9)
plt.title("plot 9")

# width = 32 velocity = 192
X, Y = getdata(32, 192)
plt.scatter(X, Y, s=10)
plt.subplot(4, 4, 10)
plt.title("plot 10")

# width = 64 velocity = 192
X, Y = getdata(64, 192)
plt.scatter(X, Y, s=10)
plt.subplot(4, 4, 11)
plt.title("plot 11")

# width = 96 velocity = 192
X, Y = getdata(96, 192)
plt.scatter(X, Y, s=10)
plt.subplot(4, 4, 12)
plt.title("plot 12")

# width = 16 velocity = 256
X, Y = getdata(16, 256)
plt.scatter(X, Y, s=10)
plt.subplot(4, 4, 13)
plt.title("plot 13")

# width = 32 velocity = 256
X, Y = getdata(32, 256)
plt.scatter(X, Y, s=10)
plt.subplot(4, 4, 14)
plt.title("plot 14")

# width = 64 velocity = 256
X, Y = getdata(64, 256)
plt.scatter(X, Y, s=10)
plt.subplot(4, 4, 15)
plt.title("plot 15")

# width = 96 velocity = 256
X, Y = getdata(96, 256)
plt.scatter(X, Y, s=10)
plt.subplot(4, 4, 16)
plt.title("plot 16")

plt.suptitle("endpoint distribution")

print(X,Y)
plt.show()




#     for row in reader:
#         endpoint_x.append(int(row[8]))
#         endpoint_y.append(int(row[9]))
#     print(endpoint_x)
#     print(endpoint_y)
#
# fig=plt.figure(dpi=128,figsize=(10,6))
# plt.scatter(endpoint_x, endpoint_y)
#
# plt.title("distribution of endpoint",fontsize=24)
# plt.xlabel('endpoint_x',fontsize=24)
# fig.autofmt_xdate()
# plt.ylabel('endpoint_y',fontsize=24)

