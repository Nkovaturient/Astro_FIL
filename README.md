# ML Workflow with Filecoin & Lighthouse Storage

This project demonstrates a complete machine learning workflow leveraging decentralized storage through Filecoin and Lighthouse Storage. It showcases how to store datasets and trained ML models on the decentralized Filecoin network, providing persistence, availability, and verifiability for machine learning assets.

## Overview

The workflow includes the following steps:

1. **Dataset Creation & Upload**: Generates a synthetic diabetes prediction dataset and uploads it to Filecoin via Lighthouse
2. **Model Training**: Downloads the dataset from Filecoin, trains a Random Forest classifier model
3. **Model Storage**: Uploads the trained model and its metadata to Filecoin
4. **Model Inference**: Downloads the model and performs predictions on sample data

## Prerequisites

- Python 3.8+
- Required Python packages:
  ```
  pip install lighthouseweb3 scikit-learn pandas numpy joblib requests
  ```
- A Lighthouse Storage API key (sign up at [lighthouse.storage](https://files.lighthouse.storage/dashboard))

## Installation

1. Clone this repository
2. Replace the `LIGHTHOUSE_API_KEY` variable in the script with your actual Lighthouse API key

## Usage

Simply run the script:

```
python modelTrainingDemo.py
```

## Workflow Details

### 1. Dataset Management

The script first creates a synthetic diabetes prediction dataset with features like age, BMI, blood pressure, and glucose levels. It then uploads this dataset to Filecoin via Lighthouse Storage, generating a Content Identifier (CID) that can be used to retrieve the dataset later.

### 2. Model Training

The script downloads the dataset from Filecoin using its CID, then trains a Random Forest classifier to predict diabetes based on the features. The model's performance is evaluated using accuracy metrics.

### 3. Model Storage

Both the trained model (serialized with joblib) and a JSON metadata file containing information about the model's architecture, features, and performance are uploaded to Filecoin. This creates permanent, verifiable storage for your ML assets.

### 4. Model Inference

Finally, the script demonstrates how to download the model from Filecoin and use it to make predictions on new data. This shows the complete lifecycle of ML assets on decentralized storage.

## Benefits of Using Filecoin for ML Workflows

- **Persistence**: Models and datasets are stored on a decentralized network designed for long-term storage
- **Verifiability**: Content addressing ensures data integrity
- **Availability**: Data can be accessed from anywhere through IPFS gateways
- **Censorship Resistance**: No single entity controls access to your ML assets

## Security Note

The example includes an API key for demonstration purposes. In a real implementation, you should:
- Never hardcode API keys in your source code
- Use environment variables or a secure configuration management system
- Consider setting up access controls for your uploaded ML assets

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.