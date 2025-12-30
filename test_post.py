import urllib.request, json

data = {
    "Age": "55",
    "Sex": "M",
    "ChestPainType": "TA",
    "RestingBP": "140",
    "Cholesterol": "230",
    "FastingBS": "0",
    "RestingECG": "Normal",
    "MaxHR": "150",
    "ExerciseAngina": "N",
    "Oldpeak": "1.0",
    "ST_Slope": "Up"
}

req = urllib.request.Request('http://127.0.0.1:5000/predict', data=json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'})
try:
    resp = urllib.request.urlopen(req, timeout=10)
    print(resp.read().decode())
except Exception as e:
    import sys
    print('ERROR:', e, file=sys.stderr)
    raise
