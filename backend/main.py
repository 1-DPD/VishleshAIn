from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import json
import io
import os
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import base64
from io import BytesIO
import zipfile

app = FastAPI(title="VishleshAIn API", description="API for data analytics and machine learning")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Models
class PreprocessingOptions(BaseModel):
    handleMissingValues: bool = True
    removeDuplicates: bool = True
    encodeCategorial: bool = True
    fixInconsistentData: bool = True
    handleOutliers: bool = True
    normalization: bool = False
    scaling: bool = False
    featureEngineering: bool = False
    dimensionalityReduction: bool = False
    dataCompression: bool = False

class ModelParams(BaseModel):
    modelType: str
    trainTestSplit: float = 0.8
    params: Dict[str, Any] = {}

class ReportConfig(BaseModel):
    sections: List[str]
    visualizations: List[str]
    format: str = "pdf"

# Global storage for the current session
class DataSession:
    def __init__(self):
        self.raw_data = None
        self.processed_data = None
        self.model = None
        self.model_results = None
        self.visualizations = {}
        self.report = None

data_session = DataSession()

# Helper functions
def read_file(file: UploadFile) -> pd.DataFrame:
    """Read uploaded file into a pandas DataFrame"""
    content = file.file.read()
    
    if file.filename.endswith('.csv'):
        return pd.read_csv(io.StringIO(content.decode('utf-8')))
    elif file.filename.endswith('.json'):
        return pd.read_json(io.StringIO(content.decode('utf-8')))
    elif file.filename.endswith('.xlsx') or file.filename.endswith('.xls'):
        return pd.read_excel(io.BytesIO(content))
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {file.filename}")

def preprocess_data(df: pd.DataFrame, options: PreprocessingOptions) -> pd.DataFrame:
    """Preprocess data based on selected options"""
    processed_df = df.copy()
    
    # Handle missing values
    if options.handleMissingValues:
        # For numerical columns
        num_cols = processed_df.select_dtypes(include=['number']).columns
        if len(num_cols) > 0:
            imputer = SimpleImputer(strategy='mean')
            processed_df[num_cols] = imputer.fit_transform(processed_df[num_cols])
        
        # For categorical columns
        cat_cols = processed_df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            processed_df[cat_cols] = processed_df[cat_cols].fillna(processed_df[cat_cols].mode().iloc[0])
    
    # Remove duplicates
    if options.removeDuplicates:
        processed_df = processed_df.drop_duplicates()
    
    # Encode categorical variables
    if options.encodeCategorial:
        cat_cols = processed_df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            processed_df[col] = LabelEncoder().fit_transform(processed_df[col])
    
    # Handle outliers
    if options.handleOutliers:
        num_cols = processed_df.select_dtypes(include=['number']).columns
        for col in num_cols:
            Q1 = processed_df[col].quantile(0.25)
            Q3 = processed_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            processed_df[col
