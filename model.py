import torch
import helper
import config
import random
import uuid
import os
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import NNConv
import time
import sys
from torch.nn import Sequential, Linear, ReLU

#set seed for reproducibility
torch.manual_seed(35813)
np.random.seed(35813)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#check if any gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
class MGN_NET(torch.nn.Module):
    def __init__(self, dataset):
        super(MGN_NET, self).__init__()
        
        model_params = config.PARAMS
        
        nn = Sequential(Linear(model_params["Linear1"]["in"], model_params["Linear1"]["out"]), ReLU())
        self.conv1 = NNConv(model_params["conv1"]["in"], model_params["conv1"]["out"], nn, aggr='mean')
        
        nn = Sequential(Linear(model_params["Linear2"]["in"], model_params["Linear2"]["out"]), ReLU())
        self.conv2 = NNConv(model_params["conv2"]["in"], model_params["conv2"]["out"], nn, aggr='mean')
        
        nn = Sequential(Linear(model_params["Linear3"]["in"], model_params["Linear3"]["out"]), ReLU())
        self.conv3 = NNConv(model_params["conv3"]["in"], model_params["conv3"]["out"], nn, aggr='mean')
        
        
    def forward(self, data):
        """
            Args:
                data (Object): data object consist of three parts x, edge_attr, and edge_index.
                                This object can be produced by using helper.cast_data function
                        x: Node features with shape [number_of_nodes, 1] (Simply set to vector of ones since we dont have any)
                        edge_attr: Edge features with shape [number_of_edges, number_of_views]
                        edge_index: Graph connectivities with shape [2, number_of_edges] (COO format) 
                        
        """
        x, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        
        x = F.relu(self.conv3(x, edge_index, edge_attr))
        
        repeated_out = x.repeat(35,1,1)
        repeated_t   =  torch.transpose(repeated_out, 0, 1)
        diff = torch.abs(repeated_out - repeated_t)
        cbt = torch.sum(diff, 2)
        
        return cbt
    
    @staticmethod
    def generate_subject_biased_cbts(model, train_data):
        """
            Generates all possible CBTs for a given training set.
            Args:
                model: trained DGN model
                train_data: list of data objects
        """
        model.eval()
        cbts = np.zeros((35,35, len(train_data)))
        train_data = [d.to(device) for d in train_data]
        for i, data in enumerate(train_data):
            cbt = model(data)
            cbts[:,:,i] = np.array(cbt.cpu().detach())
        
        return cbts
        
    @staticmethod
    def generate_cbt_median(model, train_data):
        """
            Generate optimized CBT for the training set (use post training refinement)
            Args:
                model: trained DGN model
                train_data: list of data objects
        """
        model.eval()
        cbts = []
        train_data = [d.to(device) for d in train_data]
        for data in train_data:
            cbt = model(data)
            cbts.append(np.array(cbt.cpu().detach()))
        final_cbt = torch.tensor(np.median(cbts, axis = 0), dtype = torch.float32).to(device)
        
        return final_cbt  
    
    @staticmethod
    def KL_error(cbt, target_data, six_views = False):
        """
            Calculate the KL_divergence between the CBT and test subjects (all views)
            Args:
                cbt: models output
                target_data: list of data objects
        """
        cbt_dist = cbt.sum(axis = 1)
        cbt_probs = cbt_dist / cbt_dist.sum()
        
        views = torch.cat([data.con_mat for data in target_data], axis = 2).permute((2,1,0))
        #View 1
        view1_mean = views[range(0,views.shape[0],6 if six_views else 4)].mean(axis = 0)
        view1_dist = view1_mean.sum(axis = 1)
        view1_prob = view1_dist / view1_dist.sum()
        kl_1 = ((cbt_probs * torch.log2(cbt_probs/view1_prob)).sum().abs()) +  ((view1_prob * torch.log2(view1_prob/cbt_probs)).sum().abs())
        
        #View 2
        view2_mean = views[range(1,views.shape[0],6 if six_views else 4)].mean(axis = 0)
        view2_dist = view2_mean.sum(axis = 1)
        view2_prob = view2_dist / view2_dist.sum()
        kl_2 = ((cbt_probs * torch.log2(cbt_probs/view2_prob)).sum().abs()) +  ((view2_prob * torch.log2(view2_prob/cbt_probs)).sum().abs())
        
        #View 3
        view3_mean = views[range(2,views.shape[0],6 if six_views else 4)].mean(axis = 0)
        view3_dist = view3_mean.sum(axis = 1)
        view3_prob = view3_dist / view3_dist.sum()
        kl_3 = ((cbt_probs * torch.log2(cbt_probs/view3_prob)).sum().abs()) +  ((view3_prob * torch.log2(view3_prob/cbt_probs)).sum().abs())
        
        #View 4
        view4_mean = views[range(3,views.shape[0],6 if six_views else 4)].mean(axis = 0) 
        view4_dist = view4_mean.sum(axis = 1)
        view4_prob = view4_dist / view4_dist.sum()
        kl_4 = ((cbt_probs * torch.log2(cbt_probs/view4_prob)).sum().abs()) + ((view4_prob * torch.log2(view4_prob/cbt_probs)).sum().abs())
        
        if six_views:
            #View 5
            view5_mean = views[range(4,views.shape[0],6 if six_views else 4)].mean(axis = 0)
            view5_dist = view5_mean.sum(axis = 1)
            view5_prob = view5_dist / view5_dist.sum()
            kl_5 = ((cbt_probs * torch.log2(cbt_probs/view5_prob)).sum().abs()) +  ((view5_prob * torch.log2(view5_prob/cbt_probs)).sum().abs())
            
            #View 6
            view6_mean = views[range(5,views.shape[0],6 if six_views else 4)].mean(axis = 0) 
            view6_dist = view6_mean.sum(axis = 1)
            view6_prob = view6_dist / view6_dist.sum()
            kl_6 = ((cbt_probs * torch.log2(cbt_probs/view6_prob)).sum().abs()) + ((view6_prob * torch.log2(view6_prob/cbt_probs)).sum().abs())
        else:
            kl_5, kl_6 = 0, 0 
        return kl_1, kl_2, kl_3, kl_4, kl_5, kl_6
        
                    
    
    @staticmethod
    def mean_frobenious_distance(generated_cbt, test_data):
        """
            Calculate the mean Frobenious distance between the CBT and test subjects (all views)
            Args:
                generated_cbt: trained DGN model
                test_data: list of data objects
        """
        frobenius_all = []
        for data in test_data:
            views = data.con_mat
            for index in range(views.shape[2]):
                diff = torch.abs(views[:,:,index] - generated_cbt)
                diff = diff*diff
                sum_of_all = diff.sum()
                d = torch.sqrt(sum_of_all)
                frobenius_all.append(d)
        return sum(frobenius_all) / len(frobenius_all)
    
    @staticmethod
    def train_model(n_max_epochs, data_path, early_stop, model_name, weighted_loss = True, random_sample_size = 10, n_folds = 5):
        """
            Trains a model for each cross validation fold and 
            saves all models along with CBTs to ./output/<model_name> 
            Args:
                n_max_epochs (int): number of training epochs (if early_stop == True this is maximum epoch limit)
                data_path (string): file path for the dataset 
                early_stop (bool): if set true, model will stop training when overfitting starts.
                model_name (string): name for saving the model
                weighted (bool): view normalization in centeredness loss
                random_sample_size (int): random subset size for SNL function
                n_folds (int): number of cross validation folds
            Return:
                models: trained models 
        """
        models = []
        n_attr = config.Nattr
        dataset = "simulated"
        
        save_path = config.MODEL_WEIGHT_BACKUP_PATH + "/" + model_name + "/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        if not os.path.isdir("temp"):
            os.makedirs("temp")
            
        model_id = str(uuid.uuid4())
        model_params = config.PARAMS
        
        with open(save_path + "model_params.txt", 'w') as f:
            print(model_params, file=f)

        for i in range(n_folds):
            print("********* FOLD {} *********".format(i))
            train_data, test_data, train_mean, train_std = helper.preprocess_data_array(data_path,
                                number_of_folds=n_folds, current_fold_id=i)
            
            
            test_casted = [d.to(device) for d in helper.cast_data(test_data)]
            if weighted_loss:
                loss_weightes = torch.tensor(np.array(list((1 / train_mean) / np.max(1 / train_mean))*len(train_data)), dtype = torch.float32)
            else:
                loss_weightes =  torch.tensor(np.ones((n_attr*len(train_data))), dtype = torch.float32)
                
            loss_weightes = loss_weightes.to(device)
            train_casted = [d.to(device) for d in helper.cast_data(train_data)]
            
            model = MGN_NET(dataset)
            model = model.to(device)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=model_params["learning_rate"], weight_decay= 0.00)
            targets = [torch.tensor(tensor, dtype = torch.float32).to(device) for tensor in train_data]
            
            test_errors_rep = []
            kl1_error_ave = []
            kl2_error_ave = []
            kl3_error_ave = []
            kl4_error_ave = []
            number_views = 4
                
            tick = time.time()
            for epoch in range(n_max_epochs):
                model.train()
                losses = []
                for data in train_casted:
                    #Compose Dissimilarity matrix from network outputs
                    cbt = model(data)
                    views_sampled = random.sample(targets, random_sample_size)
                    sampled_targets = torch.cat(views_sampled, axis = 2).permute((2,1,0))
                    expanded_cbt = cbt.expand((sampled_targets.shape[0],35,35))
                    
                    #rep loss
                    diff = torch.abs(expanded_cbt - sampled_targets) #Absolute difference
                    sum_of_all = torch.mul(diff, diff).sum(axis = (1,2)) #Sum of squares
                    l = torch.sqrt(sum_of_all)  #Square root of the sum
                    
                    #KL loss
                    cbt_dist = cbt.sum(axis = 1)
                    cbt_probs = cbt_dist / cbt_dist.sum()
                    
                    #View 1 target
                    target_mean1 = sampled_targets[range(0,random_sample_size * number_views, number_views)].mean(axis = 0)
                    target_dist1 = target_mean1.sum(axis = 1)
                    target_probs1 = target_dist1 / target_dist1.sum()
                    kl_loss_1 = ((cbt_probs * torch.log2(cbt_probs/target_probs1)).sum().abs()) + ((target_probs1* torch.log2(target_probs1/cbt_probs)).sum().abs())
                    
                    #View 2 target
                    target_mean2 = sampled_targets[range(1,random_sample_size * number_views, number_views)].mean(axis = 0)
                    target_dist2 = target_mean2.sum(axis = 1)
                    target_probs2 = target_dist2 / target_dist2.sum()
                    kl_loss_2 = ((cbt_probs * torch.log2(cbt_probs/target_probs2)).sum().abs()) + ((target_probs2 * torch.log2(target_probs2/cbt_probs)).sum().abs())
                    
                    #View 3 target
                    target_mean3 = sampled_targets[range(2,random_sample_size * number_views, number_views)].mean(axis = 0)
                    target_dist3 = target_mean3.sum(axis = 1)
                    target_probs3 = target_dist3 / target_dist3.sum()
                    kl_loss_3 = ((cbt_probs * torch.log2(cbt_probs/target_probs3)).sum().abs()) + ((target_probs3* torch.log2(target_probs3/cbt_probs)).sum().abs())
                    
                    #View 4 target
                    target_mean4 = sampled_targets[range(3,random_sample_size * number_views, number_views)].mean(axis = 0)
                    target_dist4 = target_mean4.sum(axis = 1)
                    target_probs4 = target_dist4 / target_dist4.sum()
                    kl_loss_4 = ((cbt_probs * torch.log2(cbt_probs/target_probs4)).sum().abs()) + ((target_probs4* torch.log2(target_probs4/cbt_probs)).sum().abs())
                    
                        
                        
                    kl_loss = (kl_loss_1 + kl_loss_2 + kl_loss_3 + kl_loss_4)  
                    rep_loss = (l * loss_weightes[:random_sample_size * n_attr]).mean()
                    losses.append(kl_loss * model_params["lambda_kl"] + rep_loss )
            
                optimizer.zero_grad()
                loss = torch.mean(torch.stack(losses))
                loss.backward()
                optimizer.step()
                
                if epoch % 10 == 0:
                     cbt = MGN_NET.generate_cbt_median(model, train_casted)
                     rep_loss = MGN_NET.mean_frobenious_distance(cbt, test_casted)
                     kl1, kl2, kl3, kl4, kl5, kl6 = MGN_NET.KL_error(cbt, test_casted, six_views= True if dataset == "nc_asd" else False)
                     tock = time.time()
                     time_elapsed = tock - tick
                     tick = tock
                     rep_loss = float(rep_loss)
                     test_errors_rep.append(rep_loss)
                     kl1_error_ave.append(float(kl1)), kl2_error_ave.append(float(kl2))
                     kl3_error_ave.append(float(kl3)), kl4_error_ave.append(float(kl4))
                     
                     print("Epoch: {}  |  {} Rep: {:.2f}  |  KL: {:.2f} | Time Elapsed: {:.2f}  |".format(epoch,
                           data_path.split("/")[-1].split(" ")[0], rep_loss, float(kl1+kl2+kl3+kl4) * model_params["lambda_kl"], time_elapsed))
                     try:
                         #Early stopping and restoring logic
                         if len(test_errors_rep) > 5 and early_stop:
                            torch.save(model.state_dict(), "./temp/weight_" + model_id + "_" + str(rep_loss)[:5]  + ".model")
                            last_5 = test_errors_rep[-5:]
                            if(all(last_5[i] < last_5[i + 1] for i in range(4))):
                                print("Early Stopping")
                                break
                     except:
                        print("ERROR occured")
                        break
                    
                        
            restore = "./temp/weight_" + model_id + "_" + str(min(test_errors_rep))[:5] + ".model"
            model.load_state_dict(torch.load(restore))
            torch.save(model.state_dict(), save_path + "fold" + str(i) + ".model")
            models.append(model)
            cbt = MGN_NET.generate_cbt_median(model, train_casted)
            rep_loss = MGN_NET.mean_frobenious_distance(cbt, test_casted)
            kl_loss = float(sum(MGN_NET.KL_error(cbt, test_casted)))
            cbt = cbt.cpu().numpy()
            np.save( save_path + "fold" + str(i) + "_cbt", cbt)
            all_cbts = MGN_NET.generate_subject_biased_cbts(model, train_casted)
            np.save(save_path + "fold" + str(i) + "_all_cbts", all_cbts)
            print("FINAL RESULTS  REP: {}  KL: {}".format(rep_loss, kl_loss))
            
        return models