from sklearn.model_selection import KFold
import torch
from torch_geometric.data import Data
import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt

#set seed for reproducibility
torch.manual_seed(35813)
np.random.seed(35813)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def create_better_simulated(N_Subjects, N_ROIs):
    """
        Simulated dataset distributions are inspired from real measurements
        so this function creates better dataset for demo.
        However, number of views are hardcoded.
    """
    features =  np.triu_indices(N_ROIs)[0].shape[0]
    view1 = np.random.normal(0.1,0.069, (N_Subjects, features))
    view1 = view1.clip(min = 0)
    view1 = np.array([antiVectorize(v, N_ROIs) for v in view1])
    
    view2 = np.random.normal(0.72,0.5, (N_Subjects, features))
    view2 = view2.clip(min = 0)
    view2 = np.array([antiVectorize(v, N_ROIs) for v in view2])
    
    view3 = np.random.normal(0.32,0.20, (N_Subjects, features))
    view3 = view3.clip(min = 0)
    view3 = np.array([antiVectorize(v, N_ROIs) for v in view3])
    
    view4 = np.random.normal(0.03,0.015, (N_Subjects, features))
    view4 = view4.clip(min = 0)
    view4 = np.array([antiVectorize(v, N_ROIs) for v in view4])
    
    return np.stack((view1, view2, view3, view4), axis = 3)

def simulate_dataset(N_Subjects, N_ROIs, N_views):
    """
        Creates random dataset
        Args:
            N_Subjects: number of subjects
            N_ROIs: number of region of interests
            N_views: number of views
        Return:
            dataset: random dataset with shape [N_Subjects, N_ROIs, N_ROIs, N_views]
    """
    features =  np.triu_indices(N_ROIs)[0].shape[0]
    views = []
    for _ in range(N_views):
        view = np.random.uniform(0.1,2, (N_Subjects, features))
        
        view = np.array([antiVectorize(v, N_ROIs) for v in view])
        views.append(view)
    return np.stack(views, axis = 3)



def get_std_and_mean(list_of_tensors):
    tensor = np.array(list_of_tensors)
    tensor_means = np.mean(tensor[:,:,:,:], axis=(0,1,2))
    tensor_std =   np.std(tensor[:,:,:,:], axis=(0,1,2))
    return tensor_std, tensor_means


def plot_graphs(losses_array, labels, name):
    x = range(0, len(losses_array[0]) * 10, 10)
    
    if(len(losses_array) == 1):
        plt.plot(x, losses_array[0])
        plt.savefig(name + ".png")
        plt.close()
        return
    
    for losses in losses_array:
        plt.plot(x, losses)
    plt.legend(labels, loc='upper right')
    plt.savefig(name + ".png")
    plt.close()

def generate_same_folds_for_matlab(data_path, n_folds):
    for i in range(n_folds):
            print("********* FOLD {} *********".format(i))
            train_data, test_data, _, _ = preprocess_data_array(data_path, number_of_folds=n_folds, current_fold_id=i)
            
            #Uncomment these three lines to generate exactly same train and test for netNorm
            mdict = {"train": np.array(train_data).swapaxes(3,0), "test": np.array(test_data).swapaxes(3,0)}
            save_path = "/data_{}_{}.mat".format(data_path.split("/")[-1], i)
            scipy.io.savemat("./netNorm/dataset" + save_path , mdict)

def clear_dir(dir_name):
    for file in os.listdir(dir_name):
        os.remove(os.path.join(dir_name, file))

def antiVectorize(vec, m):
    M = np.zeros((m,m))
    t = 0
    for i  in range(0,m - 1):
        for j in range(i+1, m):
            M[i,j] = vec[t]
            M[j,i] = vec[t]
            t = t + 1
    return M

def Vectorize(matrix):
    return matrix[np.triu_indices(matrix.shape[0], k = 1)]

def binary_correspondence(arr1, arr2):
    count = 0
    for a in arr1:
        if (a in arr2):
            count += 1
    return count

def read_all_dataset(root, read_indices = None, connection_mask = None):
    print("reading " + root)
    files = os.listdir(root)
    all_data = []
    #try:
    files = sorted(files, key=lambda f: int(f.split(".mat")[0].split("Sub")[1]))[:155]
    for i, file in enumerate(files):
        if read_indices == None:
            mvbn = scipy.io.loadmat(root + "/" + file)["views"]
            if connection_mask is not None:
                mvbn[connection_mask != 1] = 0
            all_data.append(mvbn)
        else:
            if (i in read_indices):
                mvbn = scipy.io.loadmat(root + "/" + file)["views"]
                if connection_mask is not None:
                    mvbn[connection_mask != 1] = 0
                all_data.append(mvbn)
    
    return [np.array(data) for data in all_data]

def preprocess_data_array(data_path, number_of_folds, current_fold_id):
    X = np.load(data_path)
    kf = KFold(n_splits=number_of_folds)
    split_indices = kf.split(range(X.shape[0]))
    train_indices, test_indices = [(list(train), list(test)) for train, test in split_indices][current_fold_id]
    #Split train and test
    X_train = X[train_indices]
    X_test = X[test_indices]
    train_channel_means = np.mean(X_train, axis=(0,1,2))
    train_channel_std =   np.std(X_train, axis=(0,1,2))
    return X_train, X_test, train_channel_means, train_channel_std

    
def cast_data(array_of_tensors, subject_type = None, flat_mask = None):
    """
        Casting for NNConv
    """
    N_ROI = array_of_tensors[0].shape[0]
    CHANNELS = array_of_tensors[0].shape[2]
    
    dataset = []
    for mat in array_of_tensors:
            #Allocate numpy arrays 
            edge_index = np.zeros((2, N_ROI * N_ROI))
            edge_attr = np.zeros((N_ROI * N_ROI,CHANNELS))
            x = np.zeros((N_ROI, 1))
            y = np.zeros((1,))
            
            counter = 0
            for i in range(N_ROI):
                for j in range(N_ROI):
                    edge_index[:, counter] = [i, j]
                    edge_attr[counter, :] = mat[i, j]
                    counter += 1
    
            #Fill node feature matrix (no features every node is 1)
            for i in range(N_ROI):
                x[i,0] = 1
                
            #Get graph labels
            y[0] = None
            
            if flat_mask is not None:
                edge_index_masked = []
                edge_attr_masked = []
                for i,val in enumerate(flat_mask):
                    if val == 1:
                        edge_index_masked.append(edge_index[:,i])
                        edge_attr_masked.append(edge_attr[i,:])
                edge_index = np.array(edge_index_masked).T
                edge_attr = edge_attr_masked
            
            
            edge_index = torch.tensor(edge_index, dtype = torch.long)
            edge_attr = torch.tensor(edge_attr, dtype = torch.float)
            x = torch.tensor(x, dtype = torch.float)
            y = torch.tensor(y, dtype = torch.float)
            con_mat = torch.tensor(mat, dtype=torch.float)
            data = Data(x = x, edge_index=edge_index, edge_attr=edge_attr, con_mat = con_mat,  y=y, label = subject_type)
            dataset.append(data)
    return dataset
