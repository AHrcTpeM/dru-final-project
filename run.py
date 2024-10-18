# import pickle
# import json
# import pandas as pd
# from sklearn.svm import SVC
#
# from utils.dataloader import DataLoader
# from settings.constants import TRAIN_SPLIT_CSV, TRAIN_CSV
#
#
# with open('settings/specifications.json') as f:
#     specifications = json.load(f)
#
# raw_train = pd.read_csv(TRAIN_CSV)
# x_columns = specifications['description']['X']
# y_column = specifications['description']['y']
#
# X_raw = raw_train[x_columns]
#
# loader = DataLoader()
# loader.fit(X_raw)
# X = loader.load_data()
# y = raw_train.Survived
#
# model = SVC()
# model.fit(X, y)
# with open('models/SVC.pickle', 'wb')as f:
#     pickle.dump(model, f)
#
# import pickle
# import json
# import pandas as pd
# from sklearn.svm import SVC
#
# from utils.dataloader import DataLoader
# from settings. constants import VAL_CSV
#
#
# with open('settings/specifications.json') as f:
#     specifications = json.load(f)
#
# x_columns = specifications['description']['X']
# y_column = specifications['description']['y']
#
# raw_val = pd.read_csv(VAL_CSV)
# x_raw = raw_val[x_columns]
#
# loader = DataLoader()
# loader.fit(x_raw)
# X = loader.load_data()
# y = raw_val.Survived
#
# loaded_model = pickle.load(open('models/SVC.pickle', 'rb'))
# print(loaded_model.score(X, y))



import json
import requests
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from utils import DataLoader, Estimator, Dataset
from settings.constants import TRAIN_CSV, VAL_CSV

with open('settings/specifications.json') as f:
    specifications = json.load(f)

info = specifications['description']
x_columns, y_column, metrics = info['X'], info['y'], info['metrics']

train_set = pd.read_csv(TRAIN_CSV, header=0)
val_set = pd.read_csv(VAL_CSV, header=0)

train_x, train_y = train_set[x_columns], train_set[y_column]
val_x, val_y = val_set[x_columns], val_set[y_column]

loader = DataLoader()
loader.fit(val_x)
val_processed = loader.load_data()
print('data: ', val_processed[:10])

req_data = {'data': json.dumps(val_x.to_dict())}
response = requests.get('http://0.0.0.0:8000/predict', data=req_data)
api_predict = response.json()['prediction']
print('predict: ', api_predict[:10])
print('real_val_y: ', val_y.values[:10])

api_score = eval(metrics)(val_y, api_predict)
print('accuracy: ', api_score) #0.8603351955307262