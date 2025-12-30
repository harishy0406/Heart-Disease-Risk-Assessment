import pickle
import pandas as pd

MODEL_PATH = 'models/heart_disease_model.pkl'
METADATA_PATH = 'models/model_metadata.pkl'

with open(MODEL_PATH,'rb') as f:
    model = pickle.load(f)

with open(METADATA_PATH,'rb') as f:
    metadata = pickle.load(f)

feature_columns = metadata['feature_columns']

sample = {col: 0 for col in feature_columns}
# set some sensible values if columns exist
if 'age' in sample:
    sample['age'] = 55
if 'sex' in sample:
    sample['sex'] = 1
# try to set values from earlier test
mapping = {
    'age':55,'sex':1,'cp':0,'trestbps':140,'chol':230,'fbs':0,'restecg':0,'thalach':150,'exang':0,'oldpeak':1.0,'slope':2
}
for k,v in mapping.items():
    if k in sample:
        sample[k]=v

input_df = pd.DataFrame([sample], columns=feature_columns)

print('feature_columns:', feature_columns)
print('input_df:')
print(input_df)

pred = model.predict(input_df)
proba = model.predict_proba(input_df)
print('predict:', pred)
print('model.classes_:', getattr(model, 'classes_', 'NO CLASSES'))
print('predict_proba:', proba)
