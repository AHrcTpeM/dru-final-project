from utils import Predictor
from utils import DataLoader

from flask import Flask, request, jsonify, make_response

import pandas as pd
import json

import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from settings.constants import TRAIN_CSV, SAVED_ESTIMATOR

app = Flask(__name__)


@app.route('/predict', methods=['GET'])
def predict():
    received_keys = sorted(list(request.form.keys()))
    if len(received_keys) > 1 or 'data' not in received_keys:
        err = 'Wrong request keys'
        return make_response(jsonify(error=err), 400)

    data = json.loads(request.form.get(received_keys[0]))
    print(data)
    df = pd.DataFrame.from_dict(data)

    loader = DataLoader()
    loader.fit(df)
    processed_df = loader.load_data()

    predictor = Predictor()
    response_dict = {'prediction': predictor.predict(processed_df).tolist()}

    return make_response(jsonify(response_dict), 200)


@app.route('/fit', methods=['GET'])
def fit():
    with open('settings/specifications.json') as f:
        specifications = json.load(f)

    raw_train = pd.read_csv(TRAIN_CSV)
    x_columns = specifications['description']['X']
    y_column = specifications['description']['y']

    X_raw = raw_train[x_columns]

    loader = DataLoader()
    loader.fit(X_raw)
    X = loader.load_data()
    y = raw_train[y_column]

    model = LinearDiscriminantAnalysis()
    model.fit(X, y)
    with open(SAVED_ESTIMATOR, 'wb') as f:
        pickle.dump(model, f)

    return make_response(jsonify({"message": "OK"}), 200)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)