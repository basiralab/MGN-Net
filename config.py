#Some important constants
RANDOM_SEED = 35813

#Some important paths
DATASET_PATH = "./simulated_dataset/example.npy"


MODEL_WEIGHT_BACKUP_PATH = "./output/model_weights"
DEEP_CBT_SAVE_PATH = "./output/deep_cbts"
TEMP_FOLDER = "./temp"


#Model Configuration
N_ROIs = 35
N_RANDOM_SAMPLES = 10

#Model hyperparams
Nattr = 4
CONV1 = 36
CONV2 = 24
CONV3 = 5
lambda_kl = 25

PARAMS = {
            "learning_rate" : 0.0006,
            "n_attr": Nattr,
            "lambda_kl": lambda_kl,
            "Linear1" : {"in": Nattr, "out": CONV1},
            "conv1": {"in" : 1, "out": CONV1},
            
            "Linear2" : {"in": Nattr, "out": CONV1 * CONV2},
            "conv2": {"in" : CONV1, "out": CONV2},
            
            "Linear3" : {"in": Nattr, "out": CONV2 * CONV3},
            "conv3": {"in" : CONV2, "out": CONV3}, 
        }