from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Telco Churn Prediction API", version="1.0")

preprocessor = joblib.load('churn_preprocessor.pkl')
model = joblib.load('churn_model.pkl')

#? Data Input (Pydantic Model)
class CustomerData(BaseModel):
  gender: str
  SeniorCitizen: int
  Partner: str
  Dependents: str
  tenure: int
  PhoneService: str
  MultipleLines: str
  InternetService: str
  OnlineSecurity: str
  OnlineBackup: str
  DeviceProtection: str
  TechSupport: str
  StreamingTV: str
  StreamingMovies: str
  Contract: str
  PaperlessBilling: str
  PaymentMethod: str
  MonthlyCharges: float
  TotalCharges: float

@app.post("/predict")
def predict_churn(data: CustomerData):
  df_input = pd.DataFrame([data.dict()]) #! mengubah object Pydantic menjadi dictionary Python
  
  df_processed = preprocessor.transform(df_input)
  
  #! Prediksi
  prediction = model.predict(df_processed)[0]
  probability = model.predict_proba(df_processed)[0][1]
  hasil = "Churn" if predict_churn == 1 else "Not Churn"
  
  return {
    "prediction": hasil,
    "churn_probability_percentage": round(probability * 100, 2),
    "status": "success"
  }
