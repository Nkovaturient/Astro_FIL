## 🌌 AstroFIL MVP: Exoplanet Classifier with Filecoin/Lightouse 🌌

- The AstroFIL MVP is a Python-based project that demonstrates an end-to-end workflow for building a machine learning model to classify exoplanets using data from the NASA Exoplanet Archive. It integrates decentralized storage via Lighthouse (a Filecoin-based storage solution) to store datasets, trained models, and metadata. The project showcases data retrieval, preprocessing, model training, decentralized storage, and inference, all in a streamlined pipeline.

## 🌌 Project Overview

- The AstroFIL MVP performs the following tasks:
- Fetches exoplanet data from the NASA Exoplanet Archive.
- Preprocesses the data into a clean, usable format.
- Trains a Random Forest Classifier to label exoplanets as "confirmed."
- Uploads the dataset, trained model, and metadata to Lighthouse Storage.
- Performs inference on sample data to demonstrate the model's functionality.
- Stores metadata with Content Identifiers (CIDs) for decentralized access.

The project is designed for scalability and reproducibility, leveraging decentralized storage to ensure data and model persistence.

## 🎯 Features

- Data Source: Pulls realtime exoplanet data (orbital period, radius, mass, stellar temperature) from [NASA's Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+top+500+pl_orbper,pl_rade,pl_bmasse,st_teff+from+pscomppars&format=csv)
- Machine Learning: Trains a Random Forest Classifier with scikit-learn.
- Decentralized Storage: Uses Lighthouse to store files on the Filecoin network, returning CIDs for retrieval.
- Error Handling: Includes robust checks for upload/download failures.
- Temporary File Management: Uses Python's tempfile to handle temporary files cleanly.
- Inference: Demonstrates model predictions on test data.

## 📋 Prerequisites
- Before running the project, ensure you have:

- Python 3.8+ installed.
- A Lighthouse API key (sign up at Lighthouse Storage).

## Dependencies
- Install the required Python packages listed in `requirements.txt`:
```pandas
numpy
scikit-learn
joblib
requests
lighthouseweb3
```

- Run the following command to install dependencies:
`pip install -r requirements.txt`

## 🚀 Setup Instructions

1. Clone the Repository:
    ```
    git clone <repository-url>
    cd astrofil-mvp
     ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Set Up Lighthouse API Key:

- Obtain an API key from [Lighthouse Storage](https://docs.lighthouse.storage/lighthouse-1/how-to/create-an-api-key).
- Update the API_KEY variable in the script:API_KEY = "your-lighthouse-api-key"

5. Run the Script:
`python astrofil.py`


## 🛠️ Code Structure
- The project consists of the following key functions:

   - `create_sample_dataset()` - Downloads a subset of exoplanet data from NASA and labels it as "confirmed."
   - `upload_to_lighthouse()` - Uploads a file to Lighthouse Storage and returns its CID.
   - `download_from_lighthouse()` - Downloads a file from Lighthouse using its CID.
   - `train_model()` - Trains a Random Forest Classifier on the dataset and evaluates accuracy.
   - `main()` - Orchestrates the workflow: dataset creation, training, storage, and inference.


## Key Libraries

- pandas: Data manipulation and CSV handling.
- numpy: Numerical operations.
- scikit-learn: Machine learning model training and evaluation.
- joblib: Model serialization.
- requests: HTTP requests for NASA data and Lighthouse downloads.
- lighthouseweb3: Interaction with Lighthouse Storage.


<!-- ## 📈 Workflow Diagram
- Below is a diagrammatic representation of the AstroFIL MVP workflow:
- 🌌 AstroFIL MVP Workflow

   
┌──────────────────────────────────────────────────────────────┐
│ 1. Fetch Data                                                │
│   ┌───────────────────────────────┐                          │
│   │ NASA Exoplanet Archive       │                          │
│   │ URL: .../pscomppars (CSV)    │                          │
│   └──────────────┬────────────────┘                          │
│                  │ GET Request (requests)                    │
└──────────────────┴───────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────────────┐
│ 2. Preprocess Data                                           │
│   - Read CSV (pandas)                                        │
│   - Drop NaN values                                          │
│   - Add 'planet_type' = 1                                    │
│   - Save as exoplanet.csv (tempfile)                         │
└──────────────────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────────────┐
│ 3. Upload Dataset to Lighthouse                              │
│   ┌───────────────────────────────┐                          │
│   │ Lighthouse Storage            │                          │
│   │ API: lighthouseweb3           │                          │
│   └──────────────┬────────────────┘                          │
│                  │ Upload exoplanet.csv                      │
│                  │ Return dataset_cid                        │
└──────────────────┴───────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────────────┐
│ 4. Train ML Model                                            │
│   - Load exoplanet.csv (pandas)                              │
│   - Features: pl_orbper, pl_rade, pl_bmasse, st_teff         │
│   - Target: planet_type                                      │
│   - Split data (train_test_split)                            │
│   - Train RandomForestClassifier (scikit-learn)              │
│   - Save model as exoplanet_model.joblib (joblib)            │
└──────────────────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────────────┐
│ 5. Upload Model to Lighthouse                                │
│   ┌───────────────────────────────┐                          │
│   │ Lighthouse Storage            │                          │
│   │ API: lighthouseweb3           │                          │
│   └──────────────┬────────────────┘                          │
│                  │ Upload exoplanet_model.joblib             │
│                  │ Return model_cid                          │
└──────────────────┴───────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────────────┐
│ 6. Create & Upload Metadata                                  │
│   - Metadata: title, model_cid, dataset_cid, accuracy, etc.   │
│   - Save as model_metadata.json (tempfile)                   │
│   - Upload to Lighthouse                                     │
│   - Return metadata_cid                                      │
└──────────────────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────────────┐
│ 7. Perform Inference                                         │
│   - Load test sample from trained data                       │
│   - Predict using trained model                              │
│   - Output: "Confirmed Exoplanet" or "Not Confirmed"         │
└──────────────────────────────────────────────────────────────┘
-->

## Explanation of Workflow

- **Data Fetching**: The script queries the NASA Exoplanet Archive for 500 exoplanet records, retrieving orbital period (pl_orbper), radius (pl_rade), mass (pl_bmasse), and stellar temperature (st_teff).
- **Preprocessing**: The data is cleaned (NaN values removed) and labeled with planet_type = 1 (confirmed exoplanet). It’s saved as exoplanet.csv in a temporary directory.
- **Dataset Upload**: The CSV file is uploaded to Lighthouse, and a CID is returned for decentralized access.
- **Model Training**: A Random Forest Classifier is trained on the dataset, using the four features to predict planet_type. The model is serialized as exoplanet_model.joblib.
- **Model Upload**: The trained model is uploaded to Lighthouse, returning another CID.
- **Metadata Creation**: A JSON file (model_metadata.json) is created with details like the model’s title, CIDs, accuracy, and data source. This is also uploaded to Lighthouse.
- **Inference**: The script tests the model on a sample from the test set, printing the prediction.

<!-- 📊 Expected Output
When you run the script, you’ll see logs like:
┌────────────────────────────────────────────────────────────────┐
│     Process : 🌌 Creating NASA Exoplanet dataset subset...  │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│  Process : Uploading astro dataset to Lighthouse Storage...   │
└────────────────────────────────────────────────────────────────┘
✅ Uploaded /tmp/.../exoplanet.csv | CID: Qm...
🎯 Model Accuracy: 1.0000
✅ Uploaded /tmp/.../exoplanet_model.joblib | CID: Qm...
✅ Uploaded /tmp/.../model_metadata.json | CID: Qm...
🌐 Metadata CID: Qm...

🔭 Performing Inference
Sample input: {'pl_orbper': 3.524749, 'pl_rade': 4.21, 'pl_bmasse': 8.52, 'st_teff': 5577}
Prediction: Confirmed Exoplanet


CIDs: Unique identifiers for files stored on Lighthouse (Filecoin/IPFS).
Accuracy: Likely 1.0 since all data is labeled as planet_type = 1 (this is a simplified demo).
Inference: Shows a sample prediction to verify the model works.  -->

## 🌟 Future Improvements

- **Multi-Class Classification**: Extend the model to predict different planet types (e.g., gas giants, terrestrial, neutron-stars).
- **Feature Engineering**: Add more features from the NASA archive (e.g., eccentricity, distance).
- **Dynamic Labeling**: Fetch non-exoplanet data to create a balanced dataset.
- **Model Optimization**: Tune hyperparameters or try other algorithms (e.g., XGBoost).
- **Interactive UI**: Build a web interface to input data and view predictions.
- **Decentralized Retrieval**: Automate downloading and reusing stored models/datasets.


## Endnote
**Happy exploring the cosmos with AstroFIL! 🌠**
