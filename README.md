## 🌌 AstroFIL : Realtime AI-Powered Exoplanet Classifier with Filecoin & Lighthouse 🌌
- Have you ever been awed by the vastness of universe, or the formation of stars, pillars of creation, black holes and exoplanets outside Earth, like Earth?!
- Envisioned with the aim of contributing to open science discovery, bringing more clarification and data accessibility within the scientific ecosystem along with efficient outputs of trained AI and ML models on nuances of exoplanets related datasets for quicker solutions, this Project is a step in fulfilling that vision. 
- AstroFIL demonstrates an end-to-end workflow for building a ML model to classify exoplanets using data from the NASA Exoplanet Archive. It integrates decentralized storage via Lighthouse (a Filecoin-based storage solution) to store datasets, trained models, and metadata. The project showcases data retrieval, preprocessing, model training, decentralized storage, and inference, all in a streamlined pipeline.
- Fosters decentralized security, real-time adaptability, efficiency and scientific collaboration.

## Preview:

https://www.loom.com/share/494de5ba0e214daf99716afbb5b0f5d2?sid=3fbca0db-0624-419c-80cb-247c093769bc
---

## 🌌 Project Overview

1. **Fetch Realtime Scientific Papers**: Queries latest [arXiv astro-ph](https://arxiv.org/list/astro-ph/new) abstracts and extracts scientific keywords using NER (BERT).
2. **Generate Dynamic Dataset**: Retrieves exoplanet data from NASA, synthesizes negative samples for classification, and labels accordingly.
3. **Train ML Model**: Uses a Random Forest classifier to learn exoplanet classification based on four physical features.
4. **Store on Lighthouse/Filecoin**: Uploads dataset, model (`.joblib`), and metadata (`.json`) to Lighthouse Storage and returns IPFS CIDs.
5. **Inference + Decentralized Retrieval**: Model is reloaded from CID, and predictions are made on test data.

## 🎯 Features

- 🔭 **Data Source**: NASA Exoplanet Archive API.
- 🧠 **ML Model**: Random Forest Classifier (scikit-learn).
- 📦 **Decentralized Storage**: Lighthouse + Filecoin/IPFS.
- 🧪 **Inference Ready**: Demonstrates real-time sample classification.
- 🔐 **Robust Handling**: Upload, download, and failure-safe CID operations.
- ♻️ **Temp Management**: Efficient tempfile cleanup.
- 📰 **NER on ArXiv Abstracts**: Keyword extraction from latest papers.

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
   - `fetch_arxiv_astro_papers()` – Get abstracts from arXiv (astro-ph)
   - `extract_keywords()` – Use BERT NER to extract topic keywords
   - `train_model()` - Trains a Random Forest Classifier on the dataset and evaluates accuracy.
   - `main()` - Orchestrates the workflow: dataset creation, training, storage, and inference.

---

## Key Libraries

- `pandas`, `numpy`, `scikit-learn`, `joblib` – ML pipeline
- `requests`, `feedparser` – Data fetching
- `lighthouseweb3` – IPFS/Filecoin Storage
- `transformers`, `pipeline` – Keyword extraction (NER)

---

## 🧠 Realtime Pipeline Explained

- **ArXiv paper titles + abstracts** ⟶ Keywords
- **Keywords** drive context, tracked with dataset & metadata
- **NASA exoplanet dataset** ⟶ Classifier ⟶ Decentralized upload
- Run inference on downloaded **model** + **test** data

---

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


## Explanation of Workflow

- **Data Fetching**: The script queries the NASA Exoplanet Archive for 500 exoplanet records, retrieving orbital period (pl_orbper), radius (pl_rade), mass (pl_bmasse), and stellar temperature (st_teff).
- **Preprocessing**: The data is cleaned (NaN values removed) and labeled with planet_type = 1 (confirmed exoplanet). It’s saved as exoplanet.csv in a temporary directory.
- **Dataset Upload**: The CSV file is uploaded to Lighthouse, and a CID is returned for decentralized access.
- **Model Training**: A Random Forest Classifier is trained on the dataset, using the four features to predict planet_type. The model is serialized as exoplanet_model.joblib.
- **Model Upload**: The trained model is uploaded to Lighthouse, returning another CID.
- **Metadata Creation**: A JSON file (model_metadata.json) is created with details like the model’s title, CIDs, accuracy, and data source. This is also uploaded to Lighthouse.
- **Inference**: The script tests the model on a sample from the test set, printing the prediction.
-->

## 🌟 Future Improvements

- 🌐 Multi-label: Classify gas giants, terrestrials, and neutron stars.
- 🌌 Expand features: Add stellar eccentricity, distance, and magnitude.
- 🔄 Label diversity: Add real-world unconfirmed objects.
- 🧪 AutoML: Try XGBoost or GridSearch tuning.
- 🕸️ IPFS-based UI: Build browser-based querying via CID.


## Endnote
**Built with curiosity and cosmos in mind. Explore decentralized space research with AstroFIL 🌠😊😍**
