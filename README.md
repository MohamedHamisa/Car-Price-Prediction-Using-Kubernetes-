# Car Price Prediction using Machine Learning and Kubernetes

## Overview
This project aims to predict car prices using machine learning techniques and deploy the model as a REST API using FastAPI, Docker, and Kubernetes. The project includes the following steps:
- Exploratory Data Analysis (EDA)
- Data preprocessing and handling missing values
- Model training and evaluation
- API development to serve the predictions
- Docker containerization
- Kubernetes deployment

## Project Structure

```bash
car-price-prediction/
│
├── app/                           # Contains FastAPI app
│   ├── __init__.py
│   ├── app.py                     # FastAPI API server
│   └── model/                     # Contains the serialized model
│       └── car_price_model.pkl     # Trained model
│
├── data/                          # Folder containing datasets
│   └── CarPrice_Assignment.csv     # Original car dataset
│
├── docker/                        # Docker and Kubernetes deployment files
│   ├── Dockerfile                 # Dockerfile for building the app container
│   ├── requirements.txt           # Python dependencies
│   ├── deployment.yaml            # Kubernetes deployment file
│   └── service.yaml               # Kubernetes service file
│
├── README.md                      # Project overview and setup instructions
