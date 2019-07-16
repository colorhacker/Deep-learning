from mnist import MNIST
import datetime as dt
import numpy as np
import os

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

    #拆分一个矩阵分解为特征矩阵
def partition_matrix_for(data,c_w,c_h,s):
    result = np.empty(shape=[0,c_w*c_h], dtype=int)
    for x in range(0,data.shape[0]-c_w+1,s):
        for y in range(0,data.shape[1]-c_h+1,s):
            result = np.vstack((result, data[x:x+c_w,y:y+c_h].flatten()))
    return result

#删除相同的行
def delete_same_rows(data):
    new_array = [tuple(row) for row in data]
    uniques = np.unique(new_array, axis=0)
    print('delete ', data.shape[0] - uniques.shape[0],"same matrix")
    return uniques

#删除无效值的行
def delete_nan_rows(data):
    return np.delete(data, np.where(np.isnan(data))[0], axis=0)

#拆分一堆矩阵分解为特征
#matirx 特征数据
#p_width 拆分的宽度
#p_height 拆分的高度
#step 移动步数
def partition_matrix(matirx,p_width,p_height,step):
    mkdir("./temp/")
    width_height = int(matirx.shape[1]**0.5)
    if width_height**2 != matirx.shape[1]:
        raise RuntimeError('matirx width not equal height')
    if matirx.shape[0] < 1000:
        raise RuntimeError('matirx number need more than 1000')
    for i in range(0, int(matirx.shape[0]/1000), 1):
        result = np.empty(shape=[0, p_width * p_height])
        print(dt.datetime.now(), i)
        for j in range(0, 1000, 1):
            result = np.vstack((result, partition_matrix_for(matirx[i*1000+j].reshape(width_height, width_height), p_width, p_height, step)))
        np.save("./temp/"+ str(i), result)
    result = np.load("./temp/" + str(0)+".npy")
    for i in range(1, int(matirx.shape[0]/1000), 1):
        print("load ",i)
        result = np.vstack((result,np.load("./temp/" + str(i)+".npy")))
    np.save("./temp/feature_"+str(p_width)+"x"+str(p_height)+"x"+str(step),delete_same_rows(result.astype('uint8')))
    for i in range(int(matirx.shape[0]/1000)):
        print("delete ", i)
        os.remove("./temp/" + str(i)+".npy")

if __name__=='__main__':
    images, labels = MNIST('./python-mnist/data', mode='vanilla', return_type='numpy').load_training()
    partition_matrix(images,7,7,7) #按照7x7大小 每步移动7 28x28产生16个矩阵 一共16*60000