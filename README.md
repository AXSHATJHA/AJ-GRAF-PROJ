# 🚀 Atlas: End-to-End Production-Grade ML Pipeline

[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blueviolet?style=flat-square&logo=mlflow)](https://dagshub.com)
[![DVC](https://img.shields.io/badge/DVC-Data_Versioning-9cf?style=flat-square&logo=data-version-control)](https://dvc.org)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue?style=flat-square&logo=docker)](https://www.docker.com/)
[![AWS EKS](https://img.shields.io/badge/AWS-EKS_Deployment-orange?style=flat-square&logo=amazon-aws)](https://aws.amazon.com/eks/)
[![Monitoring](https://img.shields.io/badge/Prometheus-Grafana-red?style=flat-square&logo=grafana)](https://grafana.com)

**Atlas** is a robust, enterprise-standard Machine Learning Operations (MLOps) system. It automates the entire lifecycle of a model—from data ingestion and versioning to containerized deployment on AWS Kubernetes (EKS) with real-time monitoring.

---

## 🏗️ Architecture Overview

The project follows a modular "Pipeline-as-Code" philosophy:
* **Orchestration:** DVC (Data Version Control) for reproducible pipelines.
* **Experiment Tracking:** MLflow integrated with **Dagshub** for remote logging.
* **Storage:** AWS S3 as the remote backend for data and model artifacts.
* **CI/CD:** GitHub Actions for automated testing and pushing images to **AWS ECR**.
* **Deployment:** Scalable **Kubernetes (EKS)** cluster managing the Flask application.
* **Observability:** **Prometheus** for metric scraping and **Grafana** for visual dashboards.

---

## 🛠️ Tech Stack

* **Languages:** Python 3.10
* **Frameworks:** Flask, FastAPI
* **ML Tools:** Scikit-learn, MLflow, DVC, Cookiecutter
* **Cloud & DevOps:** AWS (S3, ECR, EKS, IAM), Docker, GitHub Actions
* **Monitoring:** Prometheus, Grafana

---

## 📁 Project Structure

```text
├── .github/workflows/   # CI/CD pipelines (Build & Push to ECR)
├── flask_app/           # Containerized Flask application
├── src/                 # Modular source code
│   ├── logger/          # Custom logging utility
│   ├── data_ingestion   # Fetching data from remote sources
│   ├── preprocessing    # Cleaning and transformation
│   ├── model_building   # Model architecture & training
│   └── model_evaluation # Metrics & MLflow registration
├── dvc.yaml             # DVC pipeline stages definition
├── params.yaml          # Hyperparameters & configuration
└── Dockerfile           # Multi-stage container build
```

🚀 Execution Guide

1. Environment Setup
   
   conda create -n atlas python=3.10 -y
   conda activate atlas
   pip install -r requirements.txt

2. DVC Pipeline (Reproducibility)
   Execute the end-to-end pipeline (Ingestion ➔ Evaluation):

   dvc repro
   dvc push  # Sync data/models to AWS S3

3. Kubernetes Deployment (EKS)
   The application is deployed on AWS EKS. To verify the cluster:

   eksctl get cluster --name flask-app-cluster-aj
   kubectl get nodes
   kubectl get svc flask-app-service

📊 Monitoring Dashboard

* Prometheus: Scrapes metrics from the Flask endpoint at port 5000.
* Grafana: Visualizes request latency, error rates, and model performance.
* Data Source: Prometheus connected via EC2 Public IP.

👨‍💻 Author

Akshat Jha
Third-year Engineering Student at DTU
Top 10 Finalist - Samsung Solve for Tomorrow
