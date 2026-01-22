import os  # Import os module, used for path joining


DATA_DIR_Win = {
	'IEMOCAPFour': '/path/to/your/dataset/IEMOCAPFour', # IEMOCAP 4-class directory
	'IEMOCAPSix': '/path/to/your/dataset/IEMOCAP',      # IEMOCAP 6-class directory (uses the same source data)
}

PATH_TO_FEATURES_Win = {
	'IEMOCAPFour': os.path.join(DATA_DIR_Win['IEMOCAPFour'], 'features'),
	'IEMOCAPSix': os.path.join(DATA_DIR_Win['IEMOCAPSix'], 'features'),
}

PATH_TO_LABEL_Win = {
	'IEMOCAPSix': os.path.join(DATA_DIR_Win['IEMOCAPSix'], 'IEMOCAP_features_raw_6way.pkl'),# IEMOCAP 6-class labels
	'IEMOCAPFour': os.path.join(DATA_DIR_Win['IEMOCAPFour'], 'IEMOCAP_features_raw_4way.pkl'),# IEMOCAP 4-class labels
}

PATH_MELD = '/path/to/your/dataset/MELD_features/MELD_features_raw1.pkl'

# Paths for model saving, data caching, and logging
SAVED_ROOT = os.path.join('/path/to/your/model/output')
RESULT_DIR = os.path.join(SAVED_ROOT, 'result')     # Directory to save model output results
LOG_DIR = os.path.join(SAVED_ROOT, 'log')         # Directory to save logs
MODEL_DIR = os.path.join(SAVED_ROOT, 'model')   # Directory to save model parameters