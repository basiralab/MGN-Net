#Some important constants
RANDOM_SEED = 35813


#Some important paths
AD_LH_DATASET_PATH = "./dataset/data_ad_lmci_L/AD LH"
AD_RH_DATASET_PATH = "./dataset/data_ad_lmci_R/AD RH"
LMCI_LH_DATASET_PATH = "./dataset/data_ad_lmci_L/LMCI LH"
LMCI_RH_DATASET_PATH = "./dataset/data_ad_lmci_R/LMCI RH"
NC_LH_DATASET_PATH = "./dataset/data_nc_asd_L/NC LH"
NC_RH_DATASET_PATH = "./dataset/data_nc_asd_R/NC RH"
ASD_LH_DATASET_PATH = "./dataset/data_nc_asd_L/ASD LH"
ASD_RH_DATASET_PATH = "./dataset/data_nc_asd_R/ASD RH"



MODEL_WEIGHT_BACKUP_PATH = "./output/model_weights"
DEEP_CBT_SAVE_PATH = "./output/deep_cbts"
TEMP_FOLDER = "./temp"


#Model Configuration
N_ROIs = 35
N_RANDOM_SAMPLES = 10


#AD_LMCI KL LOSS
KL_AD_LMCI_Nattr = 4
KL_AD_LMCI_CONV1 = 36
KL_AD_LMCI_CONV2 = 24
KL_AD_LMCI_CONV3 = 5
AD_LMCI_lambda_kl = 25

KL_AD_LMCI_PARAMS = {
            "learning_rate" : 0.0006,
            "n_attr": KL_AD_LMCI_Nattr,
            "lambda_kl": AD_LMCI_lambda_kl,
            "Linear1" : {"in": KL_AD_LMCI_Nattr, "out": KL_AD_LMCI_CONV1},
            "conv1": {"in" : 1, "out": KL_AD_LMCI_CONV1},
            
            "Linear2" : {"in": KL_AD_LMCI_Nattr, "out": KL_AD_LMCI_CONV1 * KL_AD_LMCI_CONV2},
            "conv2": {"in" : KL_AD_LMCI_CONV1, "out": KL_AD_LMCI_CONV2},
            
            "Linear3" : {"in": KL_AD_LMCI_Nattr, "out": KL_AD_LMCI_CONV2 * KL_AD_LMCI_CONV3},
            "conv3": {"in" : KL_AD_LMCI_CONV2, "out": KL_AD_LMCI_CONV3}, 
        }

#NC_ASD KL LOSS
KL_NC_ASD_Nattr = 6
KL_NC_ASD_CONV1 = 36
KL_NC_ASD_CONV2 = 24
KL_NC_ASD_CONV3 = 8
NC_ASD_lambda_kl = 10

KL_NC_ASD_PARAMS = {
            "learning_rate" : 0.0006,
            "n_attr": KL_NC_ASD_Nattr,
            "lambda_kl": NC_ASD_lambda_kl,
            "Linear1" : {"in": KL_NC_ASD_Nattr, "out": KL_NC_ASD_CONV1},
            "conv1": {"in" : 1, "out": KL_NC_ASD_CONV1},
            
            "Linear2" : {"in": KL_NC_ASD_Nattr, "out": KL_NC_ASD_CONV1 * KL_NC_ASD_CONV2},
            "conv2": {"in" : KL_NC_ASD_CONV1, "out": KL_NC_ASD_CONV2},
            
            "Linear3" : {"in": KL_NC_ASD_Nattr, "out": KL_NC_ASD_CONV2 * KL_NC_ASD_CONV3},
            "conv3": {"in" : KL_NC_ASD_CONV2, "out": KL_NC_ASD_CONV3}, 
        }
