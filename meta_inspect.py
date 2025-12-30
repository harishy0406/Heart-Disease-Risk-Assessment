import pickle
with open('models/model_metadata.pkl','rb') as f:
    metadata = pickle.load(f)
print('metadata keys:', list(metadata.keys()))
print('target_encoder:', metadata.get('target_encoder'))
te = metadata.get('target_encoder')
if te is not None:
    try:
        print('classes:', te.classes_)
    except Exception as e:
        print('target_encoder has no classes_ or error:', e)
