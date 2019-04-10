
import numpy as np
from numpy.linalg import norm, inv, det, matrix_power
from scipy.optimize import check_grad
from matplotlib.patches import Circle, Wedge, Polygon, PathPatch
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt


p1=[
    [1,1],
    [3,5],
    [4,2],
    [2,0]
]

p2=[
    [5,5],
    [6,7],
    [7,8],
    [8,7],
    [9,4]
]

dim=len(p1[0])

################################# заполняем матрицу M
M=np.zeros((dim,len(p1)+len(p2)))

for i in range(0,len(p1)):
    for j in range(0,dim):
        M[j][i]=p1[i][j]

for i in range(0,len(p2)):
    for j in range(0,dim):
        M[j][i+len(p1)]=-p2[i][j]
#######################################

#######################################
def f(x):
    return pow(norm(M.dot(x)),2)

# gradient
def fgrad(x):
    return 2*M.T.dot(M).dot(x)
########################################


def projectionPart(x):
    s = 0
    y=np.zeros((len(x),1))
    for i in range(0, len(x)):
        s = s + x[i][0]
    for i in range(0, len(x)):
        y[i][0] = x[i][0] + (1 - s) / len(x)
    ###################### нашли проекцию

    m = 0
    minInd = -1
    for i in range(0, len(x)):
        if m > y[i][0]:
            m = y[i][0]
            minInd = i
    ###################### нашли мин индекс
    if minInd == -1:
        return y
    else:
        y[minInd]=0
        yn = np.zeros((len(x)-1,1))
        yn[:minInd]=y[:minInd]
        yn[minInd:]=y[minInd+1:]
        res=projectionPart(yn)
        y[:minInd]=res[:minInd]
        y[minInd+1:]=res[minInd:]
        return y


def projection(x):
    y=np.zeros((len(x),1))
    y[:len(p1)]=projectionPart(x[:len(p1)])
    y[len(p1):] = projectionPart(x[len(p1):])
    return y


def printAnswer(x):
    first=np.zeros(dim)
    second=np.zeros(dim)
    for i in range(0,len(p1)):
        for j in range(0,dim):
            first[j]=first[j]+x[i][0]*p1[i][j]

    for i in range(len(p1),len(p1)+len(p2)):
        for j in range(0,dim):
            second[j]=second[j]+x[i][0]*p2[i-len(p1)][j]

    return (first,second)


if __name__ == '__main__':
    x0 = np.zeros((len(p1) + len(p2), 1))
    x0[0][0] = 1
    x0[len(p1) + 4][0] = 1
    eps = 0.0001
    x_prev=x0

    print(printAnswer(x_prev))
    for i in range(1,100):
        x_prev=projection(x_prev-0.1/i*fgrad(x_prev))

    print(printAnswer(x_prev))
    print(np.sqrt(f(x_prev)))

    xOpt,yOpt =printAnswer(x_prev)

#########################################
######################################### визуализация
    fig, ax = plt.subplots()
    patches = []

    patches.append(Polygon(p1, True))
    patches.append(Polygon(p2, True))
    patches.append(Polygon([[8,1],[9,2],[9,2]],True))
    plt.plot([xOpt[0], yOpt[0]], [xOpt[1], yOpt[1]])

    colors = 100*np.random.rand(len(patches))
    p = PatchCollection(patches, alpha=0.4)
    p.set_array(np.array(colors))
    ax.add_collection(p)
    axes = plt.gca()
    axes.set_xlim([0, 10])
    axes.set_ylim([0, 10])

    plt.axis('equal')
    plt.show()

