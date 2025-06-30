# 🛡️ NetworkSecurity : An End-to-End MLOps Pipeline for Cyber Threat Detection

A production-ready MLOps pipeline designed to detect and classify potential network security threats, a key area in cybersecurity. It automates the complete machine learning lifecycle: from data ingestion, validation and transformation to model training and experiment tracking with **MLflow** via **DagsHub**. The system serves predictions through a **FastAPI** app and supports full CI/CD automation using **GitHub Actions**, **Docker**, and cloud deployment to **AWS ECR** and **EC2**


## 🚀 Features

- ✅ **Automated Data Ingestion**  
  Load raw data from MongoDB or local/remote sources and initiate the ML pipeline.

- ✅ **Schema Validation & Data Drift Detection**  
  Ensure incoming data quality and detect anomalies before training.

- ✅ **Data Transformation**  
  Apply preprocessing steps using Scikit-learn pipelines for model-ready data.

- ✅ **Model Training & Evaluation**  
  Train multiple classification models with hyperparameter tuning and evaluation using classification metrics.

- ✅ **Experiment Tracking**  
  Track metrics, models, and runs using **MLflow** integrated via **DagsHub**.

- ✅ **Model Packaging**  
  Combine the trained model and preprocessor using a custom `NetworkModel` class for unified inference.

- ✅ **Batch Prediction via FastAPI**  
  Serve predictions on a web-based interface that accepts uploaded CSVs and displays output in a browser.
  *(Can also be run locally using `python app.py` or `uvicorn app:app --reload`.)*

- ✅ **CI/CD Pipeline**  
  Automate testing, Docker image builds, and deployment using **GitHub Actions**.

- ✅ **Containerization & Deployment**  
  Use **Docker** for reproducibility and deploy via **AWS ECR** to **EC2** instances.


## 🛠️ Tech Stack

- **Programming Language:** Python  
- **ML Frameworks:** scikit-learn  
- **MLOps Tools:** MLflow, DagsHub  
- **API Framework:** FastAPI  
- **Containerization:** Docker  
- **CI/CD:** GitHub Actions  
- **Cloud:** AWS (ECR, EC2)  
- **Database:** MongoDB  
- **Utilities & Libraries:** pandas, numpy, uvicorn, pymongo
