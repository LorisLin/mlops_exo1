from fastapi import FastAPI
from pydantic import BaseModel
import joblib

model = joblib.load("regression.joblib")
class HouseData(BaseModel):
    size: float
    nb_rooms: int
    garden: int

app = FastAPI()

@app.post("/predict")
def predict(data: HouseData):
    features = [[data.size, data.nb_rooms, data.garden]]
    prediction = model.predict(features)
    return {"predicted_price": prediction[0]}

