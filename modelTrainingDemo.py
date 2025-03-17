"""
Complete ML Workflow with Filecoin & Lighthouse Storage

This script demonstrates a full machine learning workflow using Filecoin via Lighthouse Storage:
1. Upload a custom dataset to Filecoin via Lighthouse
2. Fetch the dataset and train a model on it
3. Save and upload the trained model to Filecoin
4. Query the trained model by loading it and making predictions

Prerequisites:
- Python 3.8+
- lighthouse-web3 package: pip install lighthouse-web3
- scikit-learn, pandas, numpy, joblib packages
"""

import os
import json
import time
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import tempfile
from lighthouseweb3 import Lighthouse

# Initialize Lighthouse client
API_KEY = "LIGHTHOUSE_API_KEY"  # Replace with your actual API key
lighthouse = Lighthouse(token=API_KEY)

def create_sample_dataset():
    """Create a sample diabetes dataset for demonstration"""
    print("\n")
    print(f"┌────────────────────────────────────────────────────────────────┐")
    print(f"│     Process : Creating sample diabetes prediction dataset...   │")
    print(f"└────────────────────────────────────────────────────────────────┘")
    print("\n")
    # Generate synthetic data based on the diabetes dataset structure
    n_samples = 1000
    
    # Features: age, bmi, blood pressure, glucose, etc.
    features = np.random.rand(n_samples, 8) 
    features[:, 0] *= 50  # age: 0-50
    features[:, 1] *= 40  # bmi: 0-40
    features[:, 2] *= 120  # blood pressure: 0-120
    features[:, 3] *= 200  # glucose: 0-200
    
    # Target: diabetes or not (0 or 1)
    target = np.random.randint(0, 2, size=n_samples)
    
    # Create a DataFrame
    feature_names = ['age', 'bmi', 'blood_pressure', 'glucose', 
                     'insulin', 'skin_thickness', 'dpf', 'pregnancies']
    
    df = pd.DataFrame(features, columns=feature_names)
    df['diabetes'] = target
    
    return df

def upload_to_lighthouse(file_path):
    """Upload a file to Lighthouse Storage and return the CID"""
    print("\n")
    print(f"┌────────────────────────────────────────────────────────────────┐")
    print(f"│  Process : Uploading sample dataset to Lighthouse Storage...   │")
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
    # Upload the file to Lighthouse Storage
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
    """Train a machine learning model on the dataset"""
    print("\n")
    print(f"┌────────────────────────────────────────────────────────────────┐")
    print(f"│                    Process : Training Model                    │")
    print(f"└────────────────────────────────────────────────────────────────┘")
    print("\n")
    # Load the dataset
    df = pd.read_csv(dataset_path)
    
    # Split features and target
    X = df.drop('diabetes', axis=1)
    y = df['diabetes']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained with accuracy: {accuracy:.4f}")
    
    # Create model info
    model_info = {
        "model_type": "RandomForestClassifier",
        "features": list(X.columns),
        "accuracy": float(accuracy),
        "training_samples": len(X_train),
        "test_samples": len(X_test)
    }
    
    return model, model_info

def main():
    # Step 1: Create and upload a dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create the dataset
        dataset = create_sample_dataset()
        dataset_path = os.path.join(temp_dir, "diabetes_dataset.csv")
        dataset.to_csv(dataset_path, index=False)
        print("\n=== Dataset created successfully ===")
        # Upload dataset to Lighthouse Storage
        dataset_cid = upload_to_lighthouse(dataset_path)
        if not dataset_cid:
            print("Failed to upload dataset. Exiting.")
            return
        
        print("\n=== Dataset uploaded successfully ===")
        print(f"Dataset CID: {dataset_cid}")
        print(f"Access at: https://gateway.lighthouse.storage/ipfs/{dataset_cid}")
        
        # Step 2: Download the dataset and train a model
        downloaded_dataset_path = os.path.join(temp_dir, "downloaded_dataset.csv")
        if download_from_lighthouse(dataset_cid, downloaded_dataset_path):
            # Train the model
            model, model_info = train_model(downloaded_dataset_path)
            
            # Step 3: Save and upload the trained model
            model_path = os.path.join(temp_dir, "diabetes_model.joblib")
            model_info_path = os.path.join(temp_dir, "model_info.json")
            
            # Save the model and its info
            joblib.dump(model, model_path)
            with open(model_info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            # Upload the model and info to Lighthouse Storage
            model_cid = upload_model_to_lighthouse(model_path)
            info_cid = upload_model_info_to_lighthouse(model_info_path)
            
            if not model_cid or not info_cid:
                print("Failed to upload model. Exiting.")
                return
            
            print(f"Model CID: {model_cid}")
            print(f"Model Info CID: {info_cid}")
            print(f"Access model at: https://gateway.lighthouse.storage/ipfs/{model_cid}")
            print(f"Access model info at: https://gateway.lighthouse.storage/ipfs/{info_cid}")
            
            # Step 4: Download and query the trained model
            downloaded_model_path = os.path.join(temp_dir, "downloaded_model.joblib")
            if download_from_lighthouse(model_cid, downloaded_model_path):
                # Load the model
                loaded_model = joblib.load(downloaded_model_path)
                
                # Generate some sample data for prediction
                print("\n")
                print(f"┌────────────────────────────────────────────────────────────────┐")
                print(f"│                    Process : Querying Model                    │")
                print(f"└────────────────────────────────────────────────────────────────┘")
                print("\n")
                sample_data = np.array([[45, 26.5, 80, 140, 200, 35, 0.5, 0]])  # Age, BMI, BP, glucose, etc.
                
                # Make predictions
                sample_df = pd.DataFrame([sample_data[0]], columns=model_info["features"])
                prediction = loaded_model.predict(sample_df)
                
                print(f"Sample input: {sample_df.iloc[0].to_dict()}")
                print(f"Prediction: {'Diabetic' if prediction[0] == 1 else 'Not Diabetic'}")
                
                # Get prediction probabilities
                proba = loaded_model.predict_proba(sample_df)
                print(f"Probability: Not Diabetic: {proba[0][0]:.4f}, Diabetic: {proba[0][1]:.4f}")

if __name__ == "__main__":
    main()