import warnings

import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)
import pickle
import math

from fhirclient.models.condition import Condition
from fhirclient.models.medicationrequest import MedicationRequest
from fhirclient.models.observation import Observation
from fhirclient.models.patient import Patient
from datetime import date
import pandas as pd
import importlib
import torch
import mimic_model_sig_obs as model

importlib.reload(model)

person_id = 820427166

data_folder = "./data/"
model_folder = "./saved_models/"


def calculate_wfl_stage(height, weight, map_dict, sex):
    ih = height.sort_values(by=['age_dict'])  # interpolate(height)  # interpolate height
    iw = weight.sort_values(by=['age_dict'])  # interpolate(weight)  # interpolate weight

    stage_list = []
    t_list = [6, 9, 12, 18, 24]
    for t in t_list:
        hv = ih[ih['age_dict'] <= t]['value']
        wv = iw[iw['age_dict'] <= t]['value']
        if len(hv) & len(wv):
            hv = hv.iloc[-1]
            wv = wv.iloc[-1]

            bmi_stage, stage_dict = calc_bmip({'height': hv, 'weight': wv, 'sex': sex, 'age': t}, map_dict)
            if t != 24:
                stage_list.append(stage_dict)
            else:
                stage_list.append(bmi_stage)
        else:
            stage_list.append('BMIp_Change_1')

    return pd.DataFrame({'age_dict': t_list, 'stage': stage_list})


def interpolate(df):
    idf = df.copy()
    idf.index = idf['age_dict']
    exist = set(idf['age_dict'].values.tolist())
    all = set(range(0, 25))
    missing = all - exist
    missing = list(missing)
    missing.sort()
    for m in missing:
        idf.loc[m] = None
    idf.interpolate(method='linear', limit_direction='forward')
    return idf


def process_input(data, map_dict):
    medication_list = []
    observation_list = []
    condition_list = []

    if data['medications']:
        for medication in data['medications']:
            medication_list.append(MedicationRequest(medication["resource"]))
    if data['observations']:
        for observation in data['observations']:
            observation_list.append(Observation(observation["resource"]))
    if data['conditions']:
        for condition in data['conditions']:
            condition_list.append(Condition(condition["resource"]))
    patient = Patient(data['patient'])
    ####################################################################################################################
    medication_df = pd.DataFrame(columns=['system', 'code', 'display', 'date'])
    for idx, medication in enumerate(medication_list):
        curr = pd.DataFrame({'system': medication.medicationCodeableConcept.coding[0].system,
                             'code': medication.medicationCodeableConcept.coding[0].code,
                             'display': medication.medicationCodeableConcept.coding[0].display,
                             'date': medication.authoredOn.date}, index=[idx])
        medication_df = pd.concat([medication_df, curr], ignore_index=True)
    ####################################################################################################################
    observation_df = pd.DataFrame(columns=['system', 'code', 'display', 'value', 'unit', 'date'])
    for idx, observation in enumerate(observation_list):
        curr = pd.DataFrame({'system': observation.code.coding[0].system,
                             'code': observation.code.coding[0].code,
                             'display': observation.code.coding[0].display,
                             'value': observation.valueQuantity.value if observation.valueQuantity else None,
                             'unit': observation.valueQuantity.unit if observation.valueQuantity else None,
                             'date': observation.effectiveDateTime.date}, index=[idx])
        observation_df = pd.concat([observation_df, curr], ignore_index=True)
    ####################################################################################################################
    condition_df = pd.DataFrame(columns=['system', 'code', 'display', 'date'])
    for idx, condition in enumerate(condition_list):
        curr = pd.DataFrame({'system': condition.code.coding[0].system,
                             'code': condition.code.coding[0].code,
                             'display': condition.code.coding[0].display,
                             'date': condition.onsetDateTime.date}, index=[idx])
        condition_df = pd.concat([condition_df, curr], ignore_index=True)

    ####################################################################################################################
    ####################################################################################################################
    #################  split BMI% form other selected observatiions #############
    bmi_df = observation_df[observation_df['code'] == '39156-5']
    observation_df = observation_df[observation_df['code'] != '39156-5']
    ###################  calculate WFL for 6,9,12,18,24  ########################
    dob = pd.to_datetime(patient.birthDate.date, utc=True)

    height = observation_df[observation_df['code'] == '8302-2'][['date', 'value']]
    weight = observation_df[observation_df['code'] == '29463-7'][['date', 'value']]

    height = add_age(height, dob)
    weight = add_age(weight, dob)

    bmi_stage_df = calculate_wfl_stage(height, weight, map_dict, patient.gender.capitalize())
    ################# Calculate BMI based on Height&Weight ######################

    bmi = pd.merge(height, weight, on='date', how='inner')
    bmi['value'] = (bmi['value_y'] * 10000) / (bmi['value_x'] ** 2)

    bmi['code'] = '39156-5'
    bmi['display'] = 'Body mass index (BMI) [Ratio]'
    bmi['system'] = 'http://loinc.org'
    bmi['unit'] = 'kg/m2'
    bmi = bmi[['system', 'code', 'display', 'value', 'unit', 'date']]

    bmi_df = pd.concat([bmi, bmi_df], ignore_index=True)
    bmi_df.drop_duplicates(subset=['value', 'code', 'date'], inplace=True)

    # TODO: FAMILY HISTORY
    ########################## Calculate Age ################################
    dob = pd.to_datetime(patient.birthDate.date, utc=True)
    medication_df = add_age(medication_df, dob)
    observation_df = add_age(observation_df, dob)
    condition_df = add_age(condition_df, dob)
    bmi_df = add_age(bmi_df, dob)

    return {'medications': medication_df, 'observations': observation_df, 'conditions': condition_df,
            'patient': patient, 'family_history': None, 'bmi': bmi_df, 'bmi_stage': bmi_stage_df}


def add_age(df, dob):
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df['age'] = (df['date'] - dob).dt.days / 30.4
    df['age_dict'] = df['age'].apply(lambda x: math.ceil(x) if x <= 24 else math.ceil(x / 12) + 22)
    # df = df[df['age_dict'] <= 32] # TODO
    df = df.sort_values(by=['age'])
    return df


def map_concept_codes(prrocessed_data, map_dict):
    loinc2concept = map_dict["loinc2concept"]
    snomed2desc = map_dict["snomed2desc"]
    rxcode2concept = map_dict["rxcode2concept"]
    atc_map = map_dict["atc_map"]
    feat_vocab = map_dict["feat_vocab"]
    meas_q = map_dict["meas_quartiles"]

    obs_df = prrocessed_data['observations']
    cond_df = prrocessed_data['conditions']
    med_df = prrocessed_data['medications']
    bmi_stage_df = prrocessed_data['bmi_stage']
    bmi_df = prrocessed_data['bmi']
    ############################### Map Medication Codes ################################
    # 'system', 'code', 'display', 'date', 'age', 'age_dict'
    med_df['concept_id'] = med_df['code'].apply(lambda x: rxcode2concept.get(str(x).strip(), -111))
    med_df['atc_3_code'] = med_df['concept_id'].apply(lambda x: atc_map.get(str(x).strip(), -222))
    med_df['feat_dict'] = med_df['atc_3_code'].apply(lambda x: feat_vocab.get(x, -333))
    med_df = med_df[med_df['feat_dict'] != -333]
    ############################### Map Observation Codes ################################
    # 'system', 'code', 'display', 'value', 'unit', 'date', 'age', 'age_dict'
    obs_df['concept_id'] = obs_df['code'].apply(lambda x: loinc2concept.get(str(x).strip(), -666))
    obs_df = obs_df[obs_df['concept_id'] != -666]
    obs_df.loc[:, 'quartile'] = obs_df.apply(lambda x: calc_q(x.concept_id, x.value, meas_q), axis=1)
    obs_df.loc[:, 'range_code'] = obs_df['concept_id'].astype(str) + "_" + obs_df['quartile'].astype(str)
    obs_df.loc[:, 'feat_dict'] = obs_df['range_code'].map(feat_vocab)
    obs_df = obs_df.drop(['range_code'], axis=1)
    ############################### Map Conditions Codes ################################
    # 'system', 'code', 'display', 'date', 'age', 'age_dict'
    cond_df['feat'] = cond_df['code'].apply(lambda x: snomed2desc.get(str(x).strip(), -444))
    cond_df['feat_dict'] = cond_df['feat'].apply(lambda x: feat_vocab.get(str(x).strip(), -555))
    cond_df = cond_df[cond_df['feat_dict'] != -555]
    ############################### Map BMIp Codes ################################
    bmi_df['feat_dict'] = 144
    bmi_stage_df['feat_dict'] = bmi_stage_df['stage'].apply(lambda x: feat_vocab.get(str(x).strip(), 244))

    prrocessed_data['observations'] = obs_df
    prrocessed_data['conditions'] = cond_df
    prrocessed_data['medications'] = med_df
    prrocessed_data['bmi_stage'] = bmi_stage_df
    prrocessed_data['bmi'] = bmi_df
    return prrocessed_data


def extract_representations(processed_data, map_dict, obser_pred_wins):
    demo = extract_demo_data(processed_data, map_dict)
    enc = extract_enc_data(processed_data)
    dec = extract_dec_data(processed_data, map_dict, obser_pred_wins)
    if len(dec):  # check if there is any record in decorder representation dataframe
        demo, enc, dec = duplicate(demo, enc, dec)

    return {"demo": demo, "enc": enc, "dec": dec}


def duplicate(demo, enc, dec):
    demo2 = demo.copy()
    demo2['person_id'] = demo2['person_id'] + 1000
    demo = pd.concat([demo, demo2], ignore_index=True)

    enc2 = enc.copy()
    enc2['person_id'] = enc2['person_id'] + 1000
    enc = pd.concat([enc, enc2], ignore_index=True)

    dec2 = dec.copy()
    dec2['person_id'] = dec2['person_id'] + 1000
    dec = pd.concat([dec, dec2], ignore_index=True)

    return demo, enc, dec


def extract_demo_data(data, map_dict):
    demoVocab = map_dict["demo_vocab"]

    ### Mehak used race/eth vice versa wrongly
    # Eth_label_list = ['NI', 'White', 'Black', 'Some Other Race', 'Asian']
    # Race_label_list = ['Non-Hispanic', 'Hispanic', 'NI']
    # Sex_list = ['Female', 'Male']
    # payer_y_list = ['Medicaid/sCHIP', 'Private/Commercial', 'NI']
    # HL7_race_list =     ["American Indian or Alaska Native", "Asian", "Black or African American", "Native Hawaiian or Other Pacific Islander", "White", "Other Race"]
    eth = data["patient"].extension[1].extension[0].valueCoding.display
    race = data["patient"].extension[0].extension[0].valueCoding.display
    sex = data["patient"].gender.capitalize()
    payer = 'Medicaid/sCHIP'  # TODO
    coi = 'COI_2'  # TODO

    if eth == "Hispanic or Latino":
        eth = 'Hispanic'
    else:
        eth = 'Non-Hispanic'

    race_dict = {"American Indian or Alaska Native": 'Some Other Race',
                 "Asian": 'Asian',
                 "Black or African American": 'Black',
                 "Native Hawaiian or Other Pacific Islander": 'Some Other Race',
                 "White": 'White',
                 "Other Race": 'Some Other Race'}
    race = race_dict.get(race, 'NI')

    patient_dict = {}
    patient_dict["person_id"] = person_id
    patient_dict["Eth_dict"] = demoVocab.get(race, -1111)  ### race/eth are swapped in the Mehaks vocab
    patient_dict["Race_dict"] = demoVocab.get(eth, -1111)
    patient_dict["Sex_dict"] = demoVocab.get(sex, -1111)
    patient_dict["Payer_dict"] = demoVocab.get(payer, -1111)
    patient_dict["COI_dict"] = demoVocab.get(coi, -1111)

    df = pd.DataFrame(patient_dict, index=[0])

    return df


def extract_enc_data(processed_data):
    # person_id	Age	value	feat_dict	age_dict

    cols_name = ["person_id", "Age", "value", "feat_dict", "age_dict"]
    df = pd.DataFrame(columns=cols_name)

    cond = processed_data['conditions']
    for index, row in cond.iterrows():
        new_row = [person_id, row['age'], row['code'], row['feat_dict'], row['age_dict']]
        df = pd.concat([df, pd.DataFrame([new_row], columns=cols_name)], ignore_index=True)

    med = processed_data['medications']
    for index, row in med.iterrows():
        new_row = [person_id, row['age'], row['code'], row['feat_dict'], row['age_dict']]
        df = pd.concat([df, pd.DataFrame([new_row], columns=cols_name)], ignore_index=True)

    obs = processed_data['observations']
    for index, row in obs.iterrows():
        new_row = [person_id, row['age'], row['code'], row['feat_dict'], row['age_dict']]
        df = pd.concat([df, pd.DataFrame([new_row], columns=cols_name)], ignore_index=True)

    bmi = processed_data['bmi_stage']
    for index, row in bmi.iterrows():
        new_row = [person_id, row['age_dict'], row['stage'], row['feat_dict'], row['age_dict']]
        df = pd.concat([df, pd.DataFrame([new_row], columns=cols_name)], ignore_index=True)

    # TODO: FAMILY HISTORY
    # fam = processed_data['family_history']
    # for index, row in fam.iterrows():
    #     # df = df.append({"person_id": person_id, "Age": row['age'], "value": 1, "feat_dict": row['feat_dict'], "age_dict": row['age_dict']}, ignore_index=True)
    #     new_row = [person_id, row['age'], row['code'], row['feat_dict'], row['age_dict']]
    #     df = pd.concat([df, pd.DataFrame([new_row], columns=cols_name)], ignore_index=True)

    ### filter based on age
    df = df[df['age_dict'] <= 24]
    ### filter unknowns
    df = df[df['feat_dict'] > 0]
    ### sort based on age
    df = df.sort_values(by=['age_dict'])
    df['feat_dict'] = df['feat_dict'].astype(int)
    df['age_dict'] = df['age_dict'].astype(int)
    return df


def extract_dec_data(data, map_dict, obser_pred_wins):
    if obser_pred_wins['obser_max'] >= 3:
        obser_max_dict_value = obser_pred_wins['obser_max'] + 22
    else:
        return pd.DataFrame()
    d = map_dict['feat_vocab']
    dec_features = map_dict['dec_features']
    med, obs, cond = data['medications'], data['observations'], data['conditions']

    # decoder representation
    dr = pd.DataFrame(columns=dec_features['name'].values.tolist())

    # time_windows -> tw
    for tw in range(25, obser_max_dict_value + 1):
        med_t = med[med['age_dict'] == tw]
        obs_t = obs[obs['age_dict'] == tw]
        cond_t = cond[cond['age_dict'] == tw]

        dr = pd.concat([dr, pd.DataFrame({'person_id': [person_id], 'age_dict': [tw]})], ignore_index=True)
        for i, f in dec_features.iterrows():
            ############## medication ################
            if f['type'] == 'med':
                if med_t[med_t['atc_3_code'] == f['name']].shape[0] > 0:
                    dr.loc[dr.age_dict == tw, f['name']] = d[f['name']]
                else:
                    dr.loc[dr.age_dict == tw, f['name']] = d[f['name']]  # d['0.0']
            ############## condition ################
            elif f['type'] == 'cond':
                if cond_t[cond_t['display'] == f['name']].shape[0] > 0:
                    dr.loc[dr.age_dict == tw, f['name']] = d[f['name']]
                else:
                    dr.loc[dr.age_dict == tw, f['name']] = d[f['name']]  # d['0.0']

            ############## observation ################
            elif f['type'] == 'meas':
                curr_obs = obs_t[obs_t['concept_id'] == f['name']]
                if curr_obs.shape[0] > 0:
                    dr.loc[dr.age_dict == tw, f['name']] = curr_obs.head(1)['feat_dict'].values[0]
                else:
                    dr.loc[dr.age_dict == tw, f['name']] = d['0.0']
            ############# TODO: FAMILY HISTORY #########
            elif f['type'] == 'fh':
                dr.loc[dr.age_dict == tw, f['name']] = d[f['name']]

    return dr


def read_mapping_dicts():
    print("Reading mapping dictionaries...")

    wflb = pd.read_csv(data_folder + "wfl_b.csv", dtype={'Length': float, 'P5': float, 'P85': float, 'P95': float})
    wflg = pd.read_csv(data_folder + "wfl_g.csv", dtype={'Length': float, 'P5': float, 'P85': float, 'P95': float})
    bmip = pd.read_csv(data_folder + "bmip.csv", dtype={'Agemos': float, 'P5': float, 'P85': float, 'P95': float})

    wflb = wflb[['Length', 'P5', 'P85', 'P95']]
    wflg = wflg[['Length', 'P5', 'P85', 'P95']]
    bmip = bmip[['Agemos', 'Sex', 'P5', 'P85', 'P95']]
    bmip['Sex'] = bmip['Sex'].map({1: 'Female', 2: 'Male'})
    wflb['Sex'] = 'Male'
    wflg['Sex'] = 'Female'
    wfl = pd.concat([wflb, wflg])

    dec_features = pd.read_csv(data_folder + 'dec_features.csv', header=0)
    meas_quartiles = pd.read_csv(data_folder + 'meas_q_intervals.csv', header=0)
    meas_quartiles['col_name'] = meas_quartiles['col_name'].astype(str).apply(lambda x: x[:-2])
    meas_quartiles = meas_quartiles.set_index('col_name')

    with open(data_folder + 'loinc2concept', 'rb') as f:
        loinc2concept = pickle.load(f)
    with open(data_folder + 'snomed2desc', 'rb') as f:
        snomed2desc = pickle.load(f)
    with open(data_folder + 'rxcode2conceptid', 'rb') as f:
        rxcode2concept = pickle.load(f)
    with open(data_folder + 'atc_map', 'rb') as f:
        atc_map = pickle.load(f)
    with open(data_folder + 'featVocab', 'rb') as f:
        feat_vocab = pickle.load(f)
    with open(data_folder + 'demoVocab', 'rb') as f:
        demoVocab = pickle.load(f)

    return {"meas_quartiles": meas_quartiles, "loinc2concept": loinc2concept, "snomed2desc": snomed2desc,
            "rxcode2concept": rxcode2concept, "atc_map": atc_map, "feat_vocab": feat_vocab,
            "dec_features": dec_features, 'bmip': bmip, 'wfl': wfl, 'demo_vocab': demoVocab}


def calc_bmip(data, map_dict):
    # >0 <=5 'Underweight'
    # >=5 <85 'Normal'
    # >=85 & <95 'Overweight'
    # >=95 'Obesity'
    bmip = map_dict['bmip']
    wfl = map_dict['wfl']

    sex = data['sex']
    age = data['age']
    if age <= 24:
        row = wfl[(wfl['Length'] == int(data['height'])) & (wfl['Sex'] == sex)].iloc[0]
        if row['P5'] > data['weight']:
            return 'Underweight', "BMIp_Change_0"
        elif row['P5'] <= data['weight'] < row['P85']:
            return 'Normal', "BMIp_Change_1"
        elif row['P85'] <= data['weight'] < row['P95']:
            return 'Overweight', "BMIp_Change_2"
        elif row['P95'] <= data['weight']:
            return 'Obesity', "BMIp_Change_3"
        else:
            return 'Not Available'
    else:
        age = round(age, 0) + 0.5
        row = bmip[(bmip['Agemos'] == age) & (bmip['Sex'] == sex)].iloc[0]
        if row['P5'] > data['bmi']:
            return 'Underweight', "BMIp_Change_0"
        elif row['P5'] <= data['bmi'] < row['P85']:
            return 'Normal', "BMIp_Change_1"
        elif row['P85'] <= data['bmi'] < row['P95']:
            return 'Overweight', "BMIp_Change_2"
        elif row['P95'] <= data['bmi']:
            return 'Obesity', "BMIp_Change_3"
        else:
            return 'Not Available'


def calc_q(concept_id, value, meas_q):
    if concept_id in meas_q.index:
        qs = meas_q.loc[concept_id]
        if value < qs['q1']:
            return "0"
        elif value < qs['q2']:
            return "1"
        elif value < qs['q3']:
            return "2"
        elif value < qs['q4']:
            return "3"
        else:
            return "4"
    else:
        return "-999"


def determine_observ_predict_windows(prrocessed_data):
    age = round((date.today() - prrocessed_data['patient'].birthDate.date).days / 365)

    if 6 >= age >= 2:
        obser_max = age
    else:
        obser_max = -10
    return {"obser_max": obser_max}


def load_models():
    models = {}
    models[2] = torch.load(model_folder + 'obsNew_0.tar', map_location='cpu')
    models[3] = torch.load(model_folder + 'obsNew_1.tar', map_location='cpu')
    models[4] = torch.load(model_folder + 'obsNew_2.tar', map_location='cpu')
    models[5] = torch.load(model_folder + 'obsNew_3.tar', map_location='cpu')
    models[6] = torch.load(model_folder + 'obsNew_4.tar', map_location='cpu')

    return models


def extract_ehr_history(prrocessed_data):
    m = prrocessed_data['medications']
    m['Type'] = 'Medication'
    o = prrocessed_data['observations']
    o['Type'] = 'Observation'
    c = prrocessed_data['conditions']
    c['Type'] = 'Condition'
    b = prrocessed_data['bmi']
    b['Type'] = 'BMI'

    out_df = pd.concat([m, o, c], ignore_index=True)  # , b
    out_df.sort_values(by=['age'], inplace=True)
    out_df = out_df[out_df['age'] >= 0]
    out_df = out_df[out_df['feat_dict'] > 0]
    out_df['age'] = out_df['age'].astype(int)
    out_df.rename(columns={'display': 'Name', 'age': 'Age (months)', 'code': 'Code'}, inplace=True)

    out_df = out_df[['Age (months)', 'Type', 'Name', 'value', 'unit']]  # , 'Code', 'feat_dict'

    return {"moc_data": out_df.to_html(na_rep="", justify='left', index=False)}


def extract_anthropometric_data(data, smax):
    observation_df = data['bmi']

    observation_df['age'] = observation_df['age'].round(0)
    observation_df['value'] = observation_df['value'].round(1)

    observation_df.sort_values(by=['age'], inplace=True)

    bmi = observation_df[observation_df['code'] == '39156-5'][['age', 'value']]
    bmi['show'] = False
    for i in range(0, smax * 12, 6):
        if i not in bmi['age'].values:
            bmi = pd.concat([bmi, pd.DataFrame([{'age': i, 'value': None, 'show': True}])], ignore_index=True)
        else:
            bmi.loc[bmi['age'] == i, 'show'] = True
    bmi = bmi.sort_values(by=['age'])
    bmi.interpolate(method='linear', limit_area='inside', inplace=True)
    bmi = bmi[bmi['show'] == True]
    bmi['value'].replace({None: 'No Data'}, inplace=True)
    bmi = bmi.replace({np.nan: None})

    return {"bmi_x": bmi["age"].to_list() + [smax * 12], "bmi_y": bmi["value"].to_list() + [None], 'smax': smax * 12}


########################################################################################################################

def encXY(data):
    enc = data['enc']
    enc_len = pd.DataFrame(enc[enc['value'] != '0'].groupby('person_id').size().reset_index(name='counts'))
    demo = data['demo']

    enc_feat = enc['feat_dict'].values
    enc_eth = demo['Eth_dict'].values
    enc_race = demo['Race_dict'].values
    enc_sex = demo['Sex_dict'].values
    enc_payer = demo['Payer_dict'].values
    enc_coi = demo['COI_dict'].values
    enc_len = enc_len['counts'].values

    enc_age = enc['age_dict'].values

    ids_len = len(pd.unique(enc['person_id']))
    # Reshape to 3-D
    enc_feat = torch.tensor(enc_feat)
    enc_feat = torch.reshape(enc_feat, (ids_len, -1))
    enc_feat = enc_feat.type(torch.LongTensor)

    enc_len = torch.tensor(enc_len)
    enc_len = enc_len.type(torch.LongTensor)

    enc_age = torch.tensor(enc_age)
    enc_age = torch.reshape(enc_age, (ids_len, -1))
    enc_age = enc_age.type(torch.LongTensor)

    enc_eth = torch.tensor(enc_eth)
    enc_eth = enc_eth.unsqueeze(1)
    enc_race = torch.tensor(enc_race)
    enc_race = enc_race.unsqueeze(1)
    enc_sex = torch.tensor(enc_sex)
    enc_sex = enc_sex.unsqueeze(1)
    enc_payer = torch.tensor(enc_payer)
    enc_payer = enc_payer.unsqueeze(1)
    enc_coi = torch.tensor(enc_coi)
    enc_coi = enc_coi.unsqueeze(1)

    enc_demo = torch.cat((enc_eth, enc_race), 1)
    enc_demo = torch.cat((enc_demo, enc_sex), 1)
    enc_demo = torch.cat((enc_demo, enc_payer), 1)
    enc_demo = torch.cat((enc_demo, enc_coi), 1)
    enc_demo = enc_demo.type(torch.LongTensor)

    return enc_feat, enc_len, enc_age, enc_demo


def decXY(data):
    dec = data['dec']

    dec = dec.fillna(0)
    dec = dec.apply(pd.to_numeric)
    del dec['age_dict']

    dec_feat = dec.iloc[:, 2:].values

    ids_len = len(pd.unique(dec['person_id']))

    mask = np.ones((dec_feat.shape[0],))
    # mask = mask['value'].values

    # Reshape to 3-D
    dec_feat = torch.tensor(dec_feat)
    dec_feat = torch.reshape(dec_feat, (
        ids_len, int(dec_feat.shape[0] / ids_len), dec_feat.shape[1]))  # Hamed Change 8 to dec_feat.shape[0]

    mask = torch.tensor(mask)
    mask = torch.reshape(mask, (ids_len, -1))

    return dec_feat, mask


def inference(data, net, obsrv_max, obs=1):
    net.eval()

    enc_feat, enc_len, enc_age, enc_demo = encXY(data)
    dec_feat, mask = decXY(data)

    pred_mask = np.ones((mask.shape[0], mask.shape[1]))  # Hamed Change zeros to ones

    if obs > 0:
        pred_mask[:, 0:obs] = mask[:, 0:obs]  # mask right
    pred_mask = torch.tensor(pred_mask)
    pred_mask = pred_mask.type(torch.DoubleTensor)
    # dec_feat

    pred_mask_feat = pred_mask.unsqueeze(2)
    pred_mask_feat = pred_mask_feat.repeat(1, 1, dec_feat.shape[2])
    pred_mask_feat = pred_mask_feat.type(torch.DoubleTensor)

    dec_feat_pred = dec_feat * pred_mask_feat
    if obs > 0:
        obs_idx = pred_mask[:, obs - 1]  # take last entry before prediction window
        obs_idx = torch.nonzero(obs_idx > 0)
        obs_idx = obs_idx.squeeze()
        dec_feat_pred = dec_feat_pred[obs_idx]
        enc_feat, enc_len, enc_age, enc_demo = enc_feat[obs_idx], enc_len[obs_idx], enc_age[obs_idx], enc_demo[obs_idx]

    output, prob, disc_input, logits = net(False, False, enc_feat, enc_len, enc_age, enc_demo, dec_feat_pred)
    ########################################################################
    output_class_list = []
    output_prob_list = []
    output_time_list = []
    for i in range(0, len(prob) - 1):  # time
        if output[i].data.cpu().numpy()[0] == 0:  ## 0 is for the first patient
            output_class_list.append('No')
        else:
            output_class_list.append('Yes')
        output_prob_list.append(round(prob[i][:, 1].data.cpu().numpy()[0], 2))
        output_time_list.append(obsrv_max + i + 1)
    ########################################################################
    # for i in range(0, len(prob) - 1):  # time
    #     print(prob[i][:, 1].data.cpu().numpy())[0]  # prob of class 1 (obesity) ---- [0] is for the first patient
    ########################################################################
    preds = pd.DataFrame({'Age (years)': output_time_list, 'Probability of Obesity': output_prob_list}).to_html(
        index=False, justify='left')  # , 'Obesity': output_class_list
    return {'preds': preds}
