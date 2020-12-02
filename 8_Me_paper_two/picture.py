import matplotlib.pyplot as plt
import matplotlib as mpl
import csv

'''读取csv文件'''


def readcsv(files):
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    for row in plots:
        y.append((row[1]))
        x.append((row[0]))
    return x, y


mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

plt.figure()
#################################################################
x1, y1 = readcsv("cifar_1.csv")
a1=2000
b1=0.103
plt.plot(x1, y1, color='r', linestyle='-',label='Gauss')
plt.scatter(a1, b1,s=250, color='r', marker='^')


x2, y2 = readcsv("cifar_2.csv")
a2=3100
b2=0.116
plt.plot(x2, y2, color='g', linestyle='-.',label='He')
plt.scatter(a2, b2, s=250,color='g', marker='^')

x3, y3 = readcsv("cifar_3.csv")

a3=1400
b3=0.08
plt.plot(x3, y3, color='b', linestyle=':',label='Our algorithm')
plt.scatter(a3, b3, s=250,color='b', marker='^')
# ##################################################################
# x1, y1 = readcsv("corel_1.csv")
# a1=180
# b1=0.109
# plt.plot(x1, y1, color='r', linestyle='-',label='Gauss')
# plt.scatter(a1, b1,s=250, color='r', marker='^')
#
#
# x2, y2 = readcsv("corel_2.csv")
# a2=210
# b2=0.1
# plt.plot(x2, y2, color='g', linestyle='-.',label='He')
# plt.scatter(a2, b2, s=250,color='g', marker='^')
#
# x3, y3 = readcsv("corel_3.csv")
# a3=150
# b3=0.070
# plt.plot(x3, y3, color='b', linestyle=':',label='Our algorithm')
# plt.scatter(a3, b3, s=250,color='b', marker='^')
##################################################################

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.ylim(0, 2)
plt.xlim(0, 4000)
# plt.ylim(0, 2)
# plt.xlim(0, 300)
plt.xlabel('Steps', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.legend(fontsize=16)
plt.show()