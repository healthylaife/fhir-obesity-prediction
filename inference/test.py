import importlib

import pandas as pd
import torch

import mimic_model_sig_obs as model
import parameters
from inference.util import inference

importlib.reload(model)

importlib.reload(parameters)

if __name__ == '__main__':
    data = {}
    person_id = [820427166, 846275781, 848524638]
    enc_df = pd.read_csv('/home/hamed/PycharmProjects/ObesityPrediction/data/5/tt/enc_test.csv', header=0)
    enc_df = enc_df[enc_df['person_id'].isin(person_id)]
    data['enc'] = enc_df

    demo_df = pd.read_csv('/home/hamed/PycharmProjects/ObesityPrediction/data/5/tt/demo_test.csv', header=0)
    demo_df = demo_df[demo_df['person_id'].isin(person_id)]
    data['demo'] = demo_df

    dec_df = pd.read_csv('/home/hamed/PycharmProjects/ObesityPrediction/data/5/tt/dec_test.csv', header=0)
    dec_df = dec_df[dec_df['person_id'].isin(person_id)]
    data['dec'] = dec_df

    net = torch.load('saved_models/obsNew_4.tar')

    inference(data, net, 1)
