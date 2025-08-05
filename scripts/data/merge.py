import joblib
from tqdm import tqdm

pkl_files = [
    "./accad.pkl",
    "./bmlhandball.pkl",
    "./bmlmovi.pkl",
    "./bmlrub.pkl",
    "./cmu.pkl",
    "./dancedb.pkl",
    "./dfaust.pkl",
    "./ekut.pkl",
    "./eyes_japan.pkl",
    "./hdm05.pkl",
    "./humaneva.pkl",
    "./human4d.pkl",
    "./kit.pkl",
    "./mosh.pkl",
    "./poseprior.pkl",
    "./sfu.pkl",
    "./soma.pkl",
    "./ssm.pkl",
    "./tcdhands.pkl",
    "./totalcapture.pkl",
    "./transitions.pkl", 
]

amass = {}

for file_path in tqdm(pkl_files):
    try:
        data = joblib.load(file_path)
        amass.update(data)
        print(f"Successfully loaded {file_path}")
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")

joblib.dump(amass, "./amass.pkl")
print(f"Total number of samples: {len(amass)}")