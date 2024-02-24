# import joblib
import os
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

from util import *

# from tensorflow import keras
app = Flask(__name__)
cors = CORS(app)

map_dict = read_mapping_dicts()

models = load_models()


@app.route("/", methods=["POST", "GET"])
@cross_origin()
def index():
    data = request.json
    prrocessed_data = process_input(data, map_dict)
    prrocessed_data = map_concept_codes(prrocessed_data, map_dict)
    obser_pred_wins = determine_observ_predict_windows(prrocessed_data)  # Determine observation and prediction windows
    represantation_data = extract_representations(prrocessed_data, map_dict, obser_pred_wins)


    net = models.get(obser_pred_wins["obser_max"], None)
    if net is None:
        inference_data = {'preds': "No model available to predict for patients at this age."}
    else:
        inference_data = inference(represantation_data, net, obser_pred_wins["obser_max"])

    anthropometric_data = extract_anthropometric_data(prrocessed_data, obser_pred_wins["obser_max"])
    ehr_history = extract_ehr_history(prrocessed_data)

    response_dict = {**anthropometric_data, **inference_data, **ehr_history}
    #########################################################################
    return jsonify(response_dict)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 4000))
    app.run(debug=False, host='0.0.0.0', port=port, ssl_context=('./key/cert.pem', './key/key.pem'))
