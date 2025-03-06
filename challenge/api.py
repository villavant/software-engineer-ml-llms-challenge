import fastapi
from fastapi import HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from model import DelayPredictionModel
import uvicorn
import os
from typing import Optional, List

app = fastapi.FastAPI()

model = DelayPredictionModel()
if not model.load_model():
    raise RuntimeError("Modelo no encontrado. Por favor, entrena el modelo primero.")

class FlightData(BaseModel):
    OPERA: str = Field(..., description="Código de la aerolínea")
    TIPOVUELO: str = Field(..., description="Tipo de vuelo (I/N)")
    MES: int = Field(..., description="Mes (1-12)", ge=1, le=12)
    SIGLADES: Optional[str] = Field(None, description="Código del aeropuerto destino")
    DIANOM: Optional[str] = Field(None, description="Día de la semana")
    DIA: Optional[int] = Field(None, description="Día del mes")
    FECHAI: str = Field(..., description="Hora de salida programada (YYYY-MM-DD HH:MM:SS)")
    FECHAO: Optional[str] = Field(None, description="Hora de salida real")
    
    class Config:
        schema_extra = {
            "example": {
                "OPERA": "Grupo LATAM",
                "TIPOVUELO": "I",
                "MES": 7,
                "SIGLADES": "SCL",
                "DIANOM": "Lunes",
                "DIA": 15,
                "FECHAI": "2023-07-15 10:30:00"
            }
        }
        
class PredictionResponse(BaseModel):
    delay_prediction: bool
    delay_probability: float
    
class BatchFlightData(BaseModel):
    flights: List[FlightData]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
        
@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}

@app.post("/predict", status_code=200, response_model=PredictionResponse)
async def post_predict(flight_data: FlightData) -> dict:
    try:
        # Convertir los datos a DataFrame
        df = pd.DataFrame([flight_data.dict()])
        # Renombrar campos para que coincidan con lo que espera el modelo
        df = df.rename(columns={"FECHAI": "Fecha-I", "FECHAO": "Fecha-O"})
        # Realizar la predicción
        delay_prob = model.predict_proba(df)[0]
        delay_prediction = model.predict(df)[0]
        return {
            "delay_prediction": bool(delay_prediction),
            "delay_probability": float(delay_prob)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")

@app.post("/predict/batch", status_code=200, response_model=BatchPredictionResponse)
async def post_predict_batch(batch_data: BatchFlightData) -> dict:
    try:
        flight_dicts = [flight.dict() for flight in batch_data.flights]
        df = pd.DataFrame(flight_dicts)
        df = df.rename(columns={"FECHAI": "Fecha-I", "FECHAO": "Fecha-O"})
        delay_probs = model.predict_proba(df)
        delay_predictions = model.predict(df)
        predictions = [
            {"delay_prediction": bool(pred), "delay_probability": float(prob)}
            for pred, prob in zip(delay_predictions, delay_probs)
        ]
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción batch: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)