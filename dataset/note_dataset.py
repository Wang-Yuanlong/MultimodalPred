import os
import torch
import gensim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

def date_diff_hrs(t1, t0):
    # Inputs:
    #   t1 -> Final timestamp in a patient hospital stay
    #   t0 -> Initial timestamp in a patient hospital stay

    # Outputs:
    #   delta_t -> Patient stay structure bounded by allowed timestamps

    try:
        delta_t = (t1-t0).total_seconds()/3600 # Result in hrs
    except:
        delta_t = np.nan
    
    return delta_t

class Note_dataset(Dataset):
    def __init__(self, split='train', data_path = 'data_clean', regenerate=True, task='mortality'):
        super(Note_dataset, self).__init__()
        self.data_path = data_path

        key_ids = pd.read_csv(os.path.join(data_path, 'haim_mimiciv_key_ids.csv')).astype('int64')
        demographic = pd.read_csv(os.path.join(data_path, 'core.csv'), dtype={"subject_id":'int64', "hadm_id":'int64'})
        demographic['admittime'] = pd.to_datetime(demographic['admittime'])
        demographic['dischtime'] = pd.to_datetime(demographic['dischtime'])

        if regenerate:
            all_admissions = key_ids[['subject_id', 'hadm_id']].drop_duplicates().reset_index(drop=True)

            data = pd.read_csv('data/mimic-note/radiology.csv').dropna()
            data[['subject_id', 'hadm_id']] = data[['subject_id', 'hadm_id']].astype('int64')
            data['charttime'] = pd.to_datetime(data['charttime'])
            data['storetime'] = pd.to_datetime(data['storetime'])
            data = all_admissions.merge(data, how='inner', on=['subject_id', 'hadm_id'])
            all_admissions = data[['subject_id', 'hadm_id']].drop_duplicates().reset_index(drop=True)
            demographic = all_admissions.merge(demographic, how='inner', on=['subject_id', 'hadm_id'])
            all_adm_disch = demographic[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'discharge_location']].drop_duplicates().reset_index(drop=True)
            all_admit_disch_time = all_adm_disch[['subject_id', 'hadm_id', 'admittime', 'dischtime']]

            data = data.merge(all_admit_disch_time, on=['subject_id', 'hadm_id'], how='left')
            data['delta_time'] = data.apply(lambda x: date_diff_hrs(x['charttime'], x['admittime']) if not x.empty else None, axis=1)
            
            data = data[(data['delta_time'] <= 48) & (data['delta_time'] > 0) & (data['charttime'] < data['dischtime'])]

            data['text'] = data.apply(lambda x: x['text'].replace('\n', ' '), axis=1)
            data['text'] = data.apply(lambda x: gensim.utils.simple_preprocess(x['text']), axis=1)
            data = data[['subject_id', 'hadm_id', 'delta_time', 'text']]

            all_admissions = data[['subject_id', 'hadm_id']].drop_duplicates().reset_index(drop=True)
            demographic = all_admissions.merge(demographic, how='inner', on=['subject_id', 'hadm_id'])
            all_adm_disch = demographic[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'discharge_location']].drop_duplicates().reset_index(drop=True)

            data.to_csv(os.path.join(data_path, 'processed/radiology_notes.csv'), index=False)
            all_adm_disch.to_csv(os.path.join(data_path, 'processed/all_adm_disch_notes.csv'), index=False)
        else:
            all_adm_disch = pd.read_csv(os.path.join(data_path, 'processed/all_adm_disch_notes.csv'), dtype={"subject_id":'int64', "hadm_id":'int64'})
            data = pd.read_csv(os.path.join(data_path, 'processed/radiology_notes.csv'), dtype={"subject_id":'int64', "hadm_id":'int64'})
            data['text'] = data.apply(lambda x: eval(x['text']), axis=1)
        
        if task == 'mortality':
            all_adm_disch['label'] = all_adm_disch.apply(lambda x: (1 if x['discharge_location'] == 'DIED' else 0) if not x.empty else None, axis=1)
        elif task == 'longstay':
            all_adm_disch['admittime'] = pd.to_datetime(all_adm_disch['admittime'])
            all_adm_disch['dischtime'] = pd.to_datetime(all_adm_disch['dischtime'])
            all_adm_disch['time_stay'] = all_adm_disch.apply(lambda x: date_diff_hrs(x['dischtime'], x['admittime']) if not x.empty else None, axis=1)
            all_adm_disch['label'] = all_adm_disch.apply(lambda x: (1 if x['time_stay'] > 7 * 24 else 0) if not x.empty else None, axis=1).astype('int64')
        elif task == 'readmission':
            def readmission_transform(x:pd.DataFrame):
                out = x.sort_values('admittime')
                out['nexttime'] = out['admittime'].shift(-1)
                out['label'] = out.apply(lambda x: 0 if pd.isna(x['nexttime']) else (1 if date_diff_hrs(x['nexttime'], x['dischtime']) <= 30 * 24 else 0), axis=1).astype('int64')
                return out['label']
            all_adm_disch['admittime'] = pd.to_datetime(all_adm_disch['admittime'])
            all_adm_disch['dischtime'] = pd.to_datetime(all_adm_disch['dischtime'])
            all_adm_disch['label'] = all_adm_disch.groupby(['subject_id'], as_index=False, group_keys=False).apply(readmission_transform)
            
        self.label = all_adm_disch.pop('label')
        self.all_notes = data
        self.all_adm_disch = all_adm_disch

        if split == 'all':
            self.data_table, self.label = self.all_adm_disch, self.label.to_numpy()
        else:
            X_train, X_test, Y_train, Y_test = train_test_split(np.arange(len(self.label)).reshape(-1,1), self.label.to_numpy(), test_size=0.15, random_state=0)
            X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.15, random_state=0)
            if split == 'train':
                self.data_table, self.label = self.all_adm_disch.iloc[X_train.reshape(-1)], Y_train
            elif split == 'test':
                self.data_table, self.label = self.all_adm_disch.iloc[X_test.reshape(-1)], Y_test
            if split == 'val':
                self.data_table, self.label = self.all_adm_disch.iloc[X_val.reshape(-1)], Y_val
        self.data_table.reset_index(drop=True)

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        sample = self.data_table.iloc[index]
        subject_id = sample.subject_id
        hadm_id = sample.hadm_id
        target = self.label[index]

        notes = self.all_notes[(self.all_notes['subject_id'] == subject_id) &\
                               (self.all_notes['hadm_id'] == hadm_id)]

        notes = list(notes['text'])
        return notes, target
    
    def query(self, subject_id, hadm_id):
        notes = self.all_notes[(self.all_notes['subject_id'] == subject_id) &\
                               (self.all_notes['hadm_id'] == hadm_id)]

        notes = list(notes['text'])
        return notes
    
    def get_samplelist(self):
        return self.data_table
    
    def get_collate(self):
        def collate_fn(samples):
            return samples
        return collate_fn

if __name__ == '__main__':
    x = Note_dataset(regenerate=True, split='all', task='longstay')
    from tqdm import tqdm
    for i in tqdm(range(len(x))):
        a = x[i]
        assert (a[0] != []) and (type(a[0]) == type([]))
    pass


        
        