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
import feedparser
import re
from transformers import pipeline
from dotenv import load_dotenv
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
    
    df_synthetic = df.sample(frac=0.3, random_state=42).copy()
    df_synthetic[['pl_orbper', 'pl_rade', 'pl_bmasse', 'st_teff']] *= np.random.uniform(1.1, 1.5)
    df_synthetic['planet_type'] = 0

    df['planet_type'] = 1
    df = pd.concat([df, df_synthetic], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
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
            print("Upload failed:", result)
            return None
    except Exception as e:
        print(f" Exception during upload: {e}")
        return None

def upload_model_to_lighthouse(file_path):
    """Upload a file to Lighthouse Storage and return the CID"""
    print("\n")
    print(f"┌────────────────────────────────────────────────────────────────┐")
    print(f"│    Process : Uploading trained model to Lighthouse Storage     │")
    print(f"└────────────────────────────────────────────────────────────────┘")
    print("\n")
    # Upload the file to Lighthouse Storage
    result = lighthouse.upload(source=file_path)
    
    if 'data' in result and 'Hash' in result['data']:
        cid = result['data']['Hash']
        print(f"Upload successful! CID: {cid}")
        return cid
    else:
        print("Upload failed:", result)
        return None
  

def upload_model_info_to_lighthouse(file_path):
    """Upload a file to Lighthouse Storage and return the CID"""
    print("\n")
    print(f"┌────────────────────────────────────────────────────────────────┐")
    print(f"│      Process : Uploading model info to Lighthouse Storage      │")
    print(f"└────────────────────────────────────────────────────────────────┘")
    print("\n")
    result = lighthouse.upload(source=file_path)
    
    if 'data' in result and 'Hash' in result['data']:
        cid = result['data']['Hash']
        print(f"Upload successful! CID: {cid}")
        return cid
    else:
        print("Upload failed:", result)
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
    print("\n")
    print(f"┌────────────────────────────────────────────────────────────────┐")
    print(f"│                    Process : Training Model                    │")
    print(f"└────────────────────────────────────────────────────────────────┘")
    print("\n")
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

# Loading a lightweight NER pipeline
extractor = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

def fetch_arxiv_astro_papers(max_results=5):
    url = f"http://export.arxiv.org/api/query?search_query=cat:astro-ph&sortBy=submittedDate&max_results={max_results}"
    parsed = feedparser.parse(url)
    papers = []
    for entry in parsed.entries:
        abstract = re.sub(r'\s+', ' ', entry.summary.strip())
        papers.append({
            "title": entry.title,
            "summary": abstract,
            "published": entry.published,
            "authors": [a.name for a in entry.authors],
            "link": entry.link
        })
    return papers

def extract_keywords(text):
    entities = extractor(text)
    return list(set(e["word"] for e in entities if e["entity_group"] in ["MISC", "ORG", "PER", "LOC"]))

def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        print("\n📡 Fetching fresh astro-ph scientific papers from arXiv...")
        papers = fetch_arxiv_astro_papers(3)

        for i, paper in enumerate(papers):
            print(f"\n Title: {paper['title']}")
            print(f" Authors: {', '.join(paper['authors'])}")
            print(f" Published: {paper['published']}")
            print(f" Link: {paper['link']}")
            keywords = extract_keywords(paper["summary"])
            print(f" Extracted Keywords: {keywords}")
        
            df = create_sample_dataset()
            dataset_filename = f"data_{i}.csv"
            dataset_path = os.path.join(temp_dir, f"exoplanet_{i}.csv")
            df.to_csv(dataset_filename, index=False)
            print("\n=== Dataset created successfully ===")

            dataset_cid = upload_to_lighthouse(dataset_filename)
            os.remove(dataset_filename)
            if not dataset_cid:
                print("Failed to upload dataset. Exiting.")
                return
        
            print("\n=== Dataset uploaded successfully ===")
            print(f"Dataset CID: {dataset_cid}")
            print(f"Access at: https://gateway.lighthouse.storage/ipfs/{dataset_cid}")

            downloaded_dataset_path = os.path.join(temp_dir, f"downloaded_exoplanet_{i}.csv")
            if not download_from_lighthouse(dataset_cid, downloaded_dataset_path):
                print("Failed to download dataset. Exiting.")
                return

            model, acc, features, X_test = train_model(downloaded_dataset_path)
            model_filename = f"model_{i}.joblib"
            model_path = os.path.join(temp_dir, f"model_{i}.joblib")
            joblib.dump(model, model_filename)

            model_cid = upload_model_to_lighthouse(model_filename)
            os.remove(model_filename)
            if not model_cid:
                print("Failed to upload model. Exiting.")
                continue

            metadata = {
               "title": "Exoplanet Classifier",
               "model_cid": model_cid,
               "dataset_cid": dataset_cid,
               "accuracy": acc,
               "trained_on": "NASA Exoplanet Archive/ArXiv",
               "paper_title": paper["title"],
               "paper_link": paper["link"],
               "paper_summary": paper["summary"],
               "keywords": keywords,
               "authors": paper["authors"],
               "published": paper["published"]
            }

            metadata_path = os.path.join(temp_dir, f"model_metadata_{i}.json")
            metadata_filename = f"model_metadata_{i}.json"
            with open(metadata_filename, 'w') as f:
               json.dump(metadata, f, indent=2)

            metadata_cid = upload_model_info_to_lighthouse(metadata_filename)
            os.remove(metadata_filename)
            if not metadata_cid:
              print("Failed to upload metadata. Exiting.")
              return

            print(f"\n🌐 Model CID: {model_cid}")
            print(f"🌐 Metadata CID: {metadata_cid}")
            print("\n")
            print(f"┌────────────────────────────────────────────────────────────────┐")
            print(f"│                    Process : Performing Inference..                    │")
            print(f"└────────────────────────────────────────────────────────────────┘")
            print("\n")

            downloaded_model_path = os.path.join(temp_dir, f"downloaded_model_{i}.joblib")
            if download_from_lighthouse(model_cid, downloaded_model_path):
                loaded_model = joblib.load(downloaded_model_path)

                print("\n 🟡Performing AI-Powered Query")
                sample = X_test.iloc[0:1]
                prediction = loaded_model.predict(sample)
                proba = loaded_model.predict_proba(sample)
                
                print(f"\nSample input: {sample.to_dict(orient='records')[0]}")
                print(f"✅Prediction: {'Confirmed Exoplanet' if prediction[0] == 1 else 'Not Confirmed'}")
                if proba.shape[1] == 2:
                  print(f"Prediction Probability → Confirmed: {proba[0][1]:.4f} | Not Confirmed: {proba[0][0]:.4f}")
                else:
                  print(f"⚠️ Model only predicted one class. Probability: {proba[0][0]:.4f}")


if __name__ == "__main__":
    main()
