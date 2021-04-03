from model import MGN_NET
import config

#Run MGN_Net
MGN_NET.train_model(n_max_epochs = 100, 
                        data_path = config.DATASET_PATH, 
                        early_stop=True,
                        model_name = "MGN_NET")


