from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

def kmeans_process(n_class,data):
    kmeans = KMeans(n_clusters=n_class,init='random',algorithm='full',random_state=99).fit(data)
    return kmeans.cluster_centers_, kmeans.labels_


def sort_func(x,y):
    return np.linalg.norm(x - y)


def custum_sort_matrix(data, rule=False):
    target_data = np.zeros(data.shape[1])
    data_list = data.tolist()
    if rule == True:
        # target_data = np.array(sorted(data_list, key=lambda element: sort_func(target_data , np.array(element))))
        target_data = np.array(sorted(data_list, key=lambda element: sort_func(target_data , np.array(element))))[len(data_list)-1]
    value = np.array(sorted(data_list, key=lambda element: sort_func(target_data , np.array(element))))
    labels = np.array(sorted(range(len(data_list)), key=lambda element: sort_func(target_data , np.array(data_list[element]))))
    return value,labels


def kmeans_sort(n_class, data):
    k_center, k_label = kmeans_process(n_class, data)
    k_sort_center, k_sort_label = custum_sort_matrix(k_center,rule=True)
    list_data = {}
    for i in range(n_class):
        list_data[i]=[]
        for j in range(data.shape[0]):
            if i == k_label[j]:
                list_data[i].append(data[j])
        list_data[i], _ = custum_sort_matrix(np.array(list_data[i]),rule=True) #最小元素排序
    sort_data = [list_data[i] for i in k_sort_label]
    ret_data = np.empty(shape=[0, data.shape[1]])
    for i in range(len(sort_data)):
        ret_data = np.vstack((ret_data,sort_data[i]))
    return ret_data


if __name__=='__main__':
    src_data = np.random.randn(10000,2)
    # np.random.shuffle(src_data)
    res_data = kmeans_sort(10,src_data)

    plt.plot(np.arange(0, res_data.shape[0]), res_data, alpha=0.5)
    plt.show()
    # a,b=custum_sort_matrix(src_data, rule=True)
    # plt.plot(np.arange(0, res_data.shape[0]), a, alpha=0.5)
    # plt.show()


