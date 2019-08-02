
from random import randint, shuffle
import matplotlib.pyplot as plt

def feed_tick(feed, fifo):
    total = 0
    for i in range(len(fifo)):
        total = total + fifo[i][-1]
        fifo[i].insert(0, fifo[i].pop())  # 向右移动一格
        fifo[i][0] = feed[i]  # 设置最左边值
    return total


_threshold = 2.5
threshold = _threshold
Recovery=True

def cell_tick(total):
    global threshold, _threshold,Recovery
    result = 0
    print(total,threshold)
    if total > threshold and Recovery == True:
        result = (total - threshold)**2
        threshold = threshold + (total - threshold)**0.5
        Recovery = False
    else:
        threshold = threshold - 0.2

    if threshold < _threshold:
        threshold = _threshold
        Recovery = True

    return result


def out_tick(feed, fifo):
    total = 0
    for i in range(len(fifo)):
        total = total + fifo[i][-1]
        fifo[i].insert(0, fifo[i].pop())  # 向右移动一格
        fifo[i][0] = feed  # 设置最左边值
    return total


if __name__ == '__main__':

    input = [[0],[0,0],[0,0,0],[0,0,0]]
    output = [[0,0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0]]
    threshold = 1.0

    data=[[1,1,1,1],[1,0,1,0],[0,1,0,0],[0,0,0,0],[0,1,0,0],[0,1,0,1],[0,1,0,1],[0,1,0,1]]

    co=100
    s=[]
    # for i in range(len(data)):
    for i in range(co):
    # while True:
        # total = feed_tick([randint(0,1) for _ in range(4)], input)
        total = feed_tick([1,1,1,1], input)
        result = cell_tick(total)
        o = out_tick(result,output)
        s.append(o)
        # print("%.1f"%o)

    plt.plot(range(co), s)
    plt.show()