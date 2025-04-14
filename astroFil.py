import os
import json
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import tempfile
from lighthouseweb3 import Lighthouse
from python-dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("LIGHTHOUSE_API_KEY")
if not API_KEY:
    raise ValueError("LIGHTHOUSE_API_KEY is missing!")
lighthouse = Lighthouse(token=API_KEY)

def create_sample_dataset():
    """Download and prepare NASA Exoplanet dataset"""
    print("\n")
    print(f"┌────────────────────────────────────────────────────────────────┐")
    print(f"│     Process : 🌌 Creating NASA Exoplanet dataset subset...  │" )
    print(f"└────────────────────────────────────────────────────────────────┘")
    print("\n")
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+top+500+pl_orbper,pl_rade,pl_bmasse,st_teff+from+pscomppars&format=csv"
    df = pd.read_csv(url)
    df = df.dropna()
    df['planet_type'] = 1  # Label all as 'confirmed'
    return df

def upload_to_lighthouse(file_path):
    """Upload file to Lighthouse Storage"""
    print("\n")
    print(f"┌────────────────────────────────────────────────────────────────┐")
    print(f"│  Process : Uploading astro dataset to Lighthouse Storage...   │")
    print(f"└────────────────────────────────────────────────────────────────┘")
    print("\n")
    try:
        result = lighthouse.upload(source=file_path)
        if 'data' in result and 'Hash' in result['data']:
            cid = result['data']['Hash']
            print(f"✅ Uploaded {file_path} | CID: {cid}")
            return cid
        else:
            print("❌ Upload failed:", result)
            return None
    except Exception as e:
        print(f"🔥 Exception during upload: {e}")
        return None
    
def download_from_lighthouse(cid, output_path):
    """Download a file from Lighthouse Storage using its CID"""
    print("\n")
    print(f"┌────────────────────────────────────────────────────────────────┐")
    print(f"│          Process : Fetching from Lighthouse Storage            │")
    print(f"└────────────────────────────────────────────────────────────────┘")
    print("\n")
    gateway_url = f"https://gateway.lighthouse.storage/ipfs/{cid}"
    print("Fetching from: ", gateway_url)
    response = requests.get(gateway_url)
    
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Download successful! Saved to {output_path}")
        return True
    else:
        print(f"Download failed with status code: {response.status_code}")
        return False

def train_model(dataset_path):
    """Train a simple ML model on the dataset"""
    df = pd.read_csv(dataset_path)
    X = df[['pl_orbper', 'pl_rade', 'pl_bmasse', 'st_teff']]
    y = df['planet_type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Model Accuracy: {acc:.4f}")
    
    return model, acc, list(X.columns), X_test

def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Step 1: Dataset
        df = create_sample_dataset()
        dataset_path = os.path.join(temp_dir, "exoplanet.csv")
        df.to_csv(dataset_path, index=False)
        
        dataset_cid = upload_to_lighthouse(dataset_path)
        if not dataset_cid:
            return
        
        # Step 2: Model Training
        model, acc, features, X_test = train_model(dataset_path)
        model_path = os.path.join(temp_dir, "exoplanet_model.joblib")
        joblib.dump(model, model_path)
        model_cid = upload_to_lighthouse(model_path)

        # Step 3: Metadata
        metadata = {
            "title": "Exoplanet Classifier",
            "model_cid": model_cid,
            "dataset_cid": dataset_cid,
            "trained_on": "NASA Exoplanet Archive",
            "accuracy": acc
        }
        metadata_path = os.path.join(temp_dir, "model_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        metadata_cid = upload_to_lighthouse(metadata_path)
        print(f"🌐 Metadata CID: {metadata_cid}")

        # Step 4: Inference
        print("\n🔭 Performing Inference")
        sample = X_test.iloc[0:1]
        pred = model.predict(sample)
        print(f"Sample input: {sample.to_dict(orient='records')[0]}")
        print(f"Prediction: {'Confirmed Exoplanet' if pred[0] == 1 else 'Not Confirmed'}")

if __name__ == "__main__":
    main()
