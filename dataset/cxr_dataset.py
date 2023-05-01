import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import os

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

class CXR_dataset(Dataset):
    def __init__(self, data_path = 'data_clean',
                       img_path='./data/mimic-cxr-jpg/',
                       split='train',
                       regenerate=False,
                       img_shape=[224, 224],
                       transform=None,
                       task='mortality'):
        super(CXR_dataset, self).__init__()
        self.data_path = data_path
        self.img_path = img_path
        self.img_shape = img_shape
        self.transform = transform
        self.key_ids = pd.read_csv(os.path.join(data_path, 'haim_mimiciv_key_ids.csv')).astype('int64')
        self.key_ids = self.key_ids[['subject_id', 'hadm_id']].drop_duplicates()
        
        self.viewpoints = ['antero-posterior']
                            # , 'left lateral', 'postero-anterior', 'lateral']
        label_name = ["Atelectasis", 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
                    "Fracture", 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 
                    'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
        if regenerate:
            self.cxr = pd.read_csv(os.path.join(data_path, 'cxr_escaped.csv')).drop_duplicates()
            self.cxr['study_id'] = self.cxr['study_id'].astype('int64')
            self.core = pd.read_csv(os.path.join(data_path, 'core.csv'))
            self.core[['subject_id', 'hadm_id']] = self.core[['subject_id', 'hadm_id']].astype('int64')

            patient_timebound = self.core[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'discharge_location']].drop_duplicates()
            patient_timebound = self.key_ids[['subject_id', 'hadm_id']].merge(patient_timebound, on=['subject_id', 'hadm_id'], how='left')

            self.cxr['cxrtime'] = pd.to_datetime(self.cxr['cxrtime']) 
            patient_timebound['admittime'] = pd.to_datetime(patient_timebound['admittime']) 
            patient_timebound['dischtime'] = pd.to_datetime(patient_timebound['dischtime']) 

            self.cxr = self.cxr.merge(patient_timebound, how='left')
            self.cxr = self.key_ids[['subject_id', 'hadm_id']].merge(self.cxr, on=['subject_id', 'hadm_id'], how='left')
            self.cxr = self.cxr[(self.cxr['cxrtime'] > self.cxr['admittime']) & (self.cxr['cxrtime'] < self.cxr['dischtime'])]
            self.cxr['time_before_img'] = self.cxr.apply(lambda x: date_diff_hrs(x['cxrtime'], x['admittime']) if not x.empty else None, axis=1)
            self.cxr = self.cxr[self.cxr['time_before_img'] < 48]
            self.cxr = self.cxr[self.cxr['ViewCodeSequence_CodeMeaning'].isin(self.viewpoints)]
            
            # last_cxr_time = self.cxr.groupby(['subject_id', 'hadm_id'], as_index=False).agg({'cxrtime':'max'})
            # last_cxr_study = last_cxr_time.merge(self.cxr[['subject_id', 'hadm_id', 'study_id', 'cxrtime']], how='left')[['subject_id', 'hadm_id', 'study_id']].drop_duplicates()
            # self.cxr = last_cxr_study.merge(self.cxr, how='left').drop_duplicates() 
            
            self.imglabel = self.cxr[label_name]

            # self.cxr['time_after_img'] = self.cxr.apply(lambda x: date_diff_hrs(x['dischtime'], x['cxrtime']) if not x.empty else None, axis=1)
            # death_sample =  self.cxr[(self.cxr['time_after_img'] < 48) & (self.cxr['discharge_location'] == 'DIED')]
            
            # death_sample = self.cxr[(self.cxr['time_before_img'] < 48) & (self.cxr['discharge_location'] == 'DIED')]
            # death_sample['label'] = 1
            # self.cxr = self.cxr.merge(death_sample, how='left')
            # self.cxr['label'] = self.cxr['label'].fillna(0).astype('int64')
            self.patient_admission = self.cxr[['subject_id', 'hadm_id']].drop_duplicates()
            self.patient_admission = self.patient_admission.merge(patient_timebound, how='left')
            self.patient_admission['time_stay'] = self.patient_admission.apply(lambda x: date_diff_hrs(x['dischtime'], x['admittime']) if not x.empty else None, axis=1)
            self.patient_admission['label_longstay'] = self.patient_admission.apply(lambda x: (1 if x['time_stay'] > 7 * 24 else 0) if not x.empty else None, axis=1).astype('int64')
            self.patient_admission['label_mortality'] = self.patient_admission.apply(lambda x: (1 if x['discharge_location'] == 'DIED' else 0) if not x.empty else None, axis=1).astype('int64')
            def readmission_transform(x:pd.DataFrame):
                out = x.sort_values('admittime')
                out['nexttime'] = out['admittime'].shift(-1)
                out['label'] = out.apply(lambda x: 0 if pd.isna(x['nexttime']) else (1 if date_diff_hrs(x['nexttime'], x['dischtime']) <= 30 * 24 else 0), axis=1).astype('int64')
                return out['label']
            self.patient_admission['label_readmission'] = self.patient_admission.groupby(['subject_id'], as_index=False, group_keys=False).apply(readmission_transform)


            self.cxr.reset_index(drop=True)
            self.cxr.to_csv(os.path.join(data_path, 'cxr_labeled.csv'), index=False)
            self.patient_admission.to_csv(os.path.join(data_path, 'pat_admit_labeled.csv'), index=False)
        else:
            self.cxr = pd.read_csv(os.path.join(data_path, 'cxr_labeled.csv')).drop_duplicates()
            self.patient_admission = pd.read_csv(os.path.join(data_path, 'pat_admit_labeled.csv'))
            self.imglabel = self.cxr[label_name]

        self.cxr.reset_index(drop=True)
        if task == 'mortality':
            self.patient_admission['label'] = self.patient_admission.pop('label_mortality')
        elif task == 'longstay':
            self.patient_admission['label'] = self.patient_admission.pop('label_longstay')
        elif task == 'readmission':
            self.patient_admission['label'] = self.patient_admission.pop('label_readmission')
        self.label = self.patient_admission['label']
        # self.cxr = self.cxr[['subject_id', 'dicom_id', 'study_id', 'split', 'Img_Filename', 'Img_Folder', 'cxrtime']]
        if split == 'all':
            self.data_table, self.label = self.patient_admission, self.label.to_numpy()
        else:
            X_train, X_test, Y_train, Y_test = train_test_split(np.arange(len(self.label)).reshape(-1,1), self.label.to_numpy(), test_size=0.15, random_state=0)
            X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.15, random_state=0)
            if split == 'train':
                self.data_table, self.label = self.patient_admission.iloc[X_train.reshape(-1)], Y_train
            elif split == 'test':
                self.data_table, self.label = self.patient_admission.iloc[X_test.reshape(-1)], Y_test
            if split == 'val':
                self.data_table, self.label = self.patient_admission.iloc[X_val.reshape(-1)], Y_val
        
        self.data_table.reset_index()


    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        x = self.data_table.iloc[[index]]
        subject_id = x['subject_id'].iloc[0]
        hadm_id = x['hadm_id'].iloc[0]
        img_meta = self.cxr[(self.cxr['subject_id'] == subject_id) & (self.cxr['hadm_id'] == hadm_id)]
        img_list = []
        img_position = []
        img_time = []
        for row in img_meta.itertuples():
            # img_folder = x['Img_Folder'].iloc[0]
            # img_file = x['Img_Filename'].iloc[0]
            img_folder = row.Img_Folder
            img_file = row.Img_Filename
            img_path = os.path.join(self.img_path, img_folder[1:], img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.img_shape[0], self.img_shape[1]))
            if self.transform != None:
                img = self.transform(img)
            img_list.append(img)
            img_position.append(row.ViewCodeSequence_CodeMeaning)
            img_time.append(row.time_before_img)
        return img_list, self.label[index], img_position, img_time

    def query(self, subject_id, hadm_id):
        img_meta = self.cxr[(self.cxr['subject_id'] == subject_id) & (self.cxr['hadm_id'] == hadm_id)]
        img_list = []
        img_position = []
        img_time = []
        for row in img_meta.itertuples():
            # img_folder = x['Img_Folder'].iloc[0]
            # img_file = x['Img_Filename'].iloc[0]
            img_folder = row.Img_Folder
            img_file = row.Img_Filename
            img_path = os.path.join(self.img_path, img_folder[1:], img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.img_shape[0], self.img_shape[1]))
            if self.transform != None:
                img = self.transform(img)
            img_list.append(img)
            img_position.append(row.ViewCodeSequence_CodeMeaning)
            img_time.append(row.time_before_img)
        return img_list, img_position, img_time

    def get_samplelist(self):
        return self.data_table


    def get_collate(self):
        def collate_fn(samples):
            img_list, labels, img_positions, img_times = map(list, zip(*map(list, samples)))
            return img_list, torch.LongTensor(labels), img_positions, img_times
        return collate_fn
        


if __name__ == '__main__':
    a = CXR_dataset(regenerate=False, split='all', task='readmission')
    a[1]