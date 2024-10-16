import os

DATA_FOLDER = 'data'
TRAIN_CSV = os.path.join(DATA_FOLDER, 'train.csv')
TRAIN_SPLIT_CSV = os.path.join(DATA_FOLDER, 'train_split.csv')
VAL_CSV = os.path.join(DATA_FOLDER, 'val.csv')
SAVED_ESTIMATOR = 'models/LDA.pickle'