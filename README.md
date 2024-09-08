# Car-Price-Prediction-Using-Kubernetes-

car-price-prediction/
│
├── app/                           # Folder containing FastAPI app
│   ├── __init__.py
│   ├── app.py                     # FastAPI API server
│   └── model/                     # Folder for the serialized model
│       └── car_price_model.pkl     # Trained model
│
├── data/                          # Folder containing datasets
│   └── CarPrice_Assignment.csv     # Original car dataset
│
├── docker/                        # Folder for Docker and Kubernetes deployment files
│   ├── Dockerfile                 # Dockerfile for building the app container
│   ├── requirements.txt           # Python dependencies
│   ├── deployment.yaml            # Kubernetes deployment file
│   └── service.yaml               # Kubernetes service file
│
├── notebooks/                     # Folder for Jupyter notebooks
│   └── EDA_and_Modeling.ipynb     # Jupyter notebook for EDA and model training
│
├── README.md                      # Project overview and setup instructions
└── .gitignore                     # Files to ignore for Git
