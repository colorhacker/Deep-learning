from sklearn.cluster import KMeans
import numpy as np

def kmeans_process(n_class,data):
    kmeans = KMeans(n_clusters=n_class,init='random',algorithm='full',random_state=99).fit(data)
    return kmeans.cluster_centers_, kmeans.labels_

def custum_sort_matrix(data):
    data_list = data.tolist()
    #return np.array(sorted(data_list, key=lambda x:np.linalg.norm(np.zeros(data.shape[1]) - np.array(x))))
    #return np.array(sorted(range(len(data_list)), key=lambda x:np.linalg.norm(np.zeros(data.shape[1]) - np.array(data_list[x]))))
    value = np.array(sorted(data_list, key=lambda x:np.linalg.norm(np.zeros(data.shape[1]) - np.array(x))))
    labels = np.array(sorted(range(len(data_list)), key=lambda x:np.linalg.norm(np.zeros(data.shape[1]) - np.array(data_list[x]))))
    return value,labels


def kmeans_sort(n_class,data):
    k_center, k_label = kmeans_process(n_class, data)
    k_sort_center, k_sort_label = custum_sort_matrix(k_center)
    list_data = {}
    for i in range(n_class):
        list_data[i]=[]
        for j in range(data.shape[0]):
            if i == k_label[j]:
                list_data[i].append(data[j])
    # k_center.take(k_sort_label)
    sort_data = [list_data[i] for i in k_sort_label]
    array_data=np.empty(shape=[0, data.shape[1]])
    for i in range(len(sort_data)):
        array_data = np.vstack((array_data,sort_data[i]))
    return array_data


if __name__=='__main__':
    data_size = 100
    data_class = 10
    src_data = np.arange(data_size).reshape(50,2)
    res_data = kmeans_sort(10,src_data)
    print(res_data)



