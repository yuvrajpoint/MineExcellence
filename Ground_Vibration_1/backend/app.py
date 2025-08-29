from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from utils import graph

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:5173"}})


model = joblib.load("/Users/yuvraj/Downloads/MineExcellence/ppv_rf_model.pkl")

features = [
    'Hole dia. [mm]', 'Hole depth [m]', 'No. of holes', 'Avg. Burden [m]', 'Avg.Spacing [m]', 'Avg. top stemming length [m]', 'Total charge [kg]', 'Max.charge delay [kg]', 'distance', 'Pit'
]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    d = {}
    for f in features:
        d[f] = data[f]

    print('This is a test line and it should be printed as it is a very useful wesite.')
    ''
    print(d)
    graph_data = graph(d, int(data['bno']))
    return jsonify(graph_data)
    # return {"message": "Success"}



if __name__ == '__main__':
    app.run(debug=True)
