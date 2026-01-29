import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os


app = FastAPI(title="Salary Prediction API")


MODEL_PATH = "artifacts/models/model.pkl"
PREPROCESSOR_PATH = "artifacts/preprocessor.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
    raise FileNotFoundError("Model or preprocessor artifacts not found. Please run training first.")

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)




from typing import Annotated
from pydantic import BaseModel, Field

# Define Input Schema
class PredictionInput(BaseModel):
    Age: Annotated[float, Field(description="Age of the employee", example=30)]
    Gender: Annotated[str, Field(description="Gender of the employee", example="Male")]
    Education_Level: Annotated[str, Field(description="Highest education level attainment", example="Bachelor's")]
    Job_Title: Annotated[str, Field(description="Current job title", example="Software Engineer")]
    Years_of_Experience: Annotated[float, Field(description="Total years of professional experience", example=5)]

@app.get("/")
def home():
    return {"message": "Welcome to the Salary Prediction API"}

@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        
        data = {
            "Age": [input_data.Age],
            "Gender": [input_data.Gender],
            "Education Level": [input_data.Education_Level],
            "Job Title": [input_data.Job_Title],
            "Years of Experience": [input_data.Years_of_Experience]
        }
        
        df = pd.DataFrame(data)

        df_processed = preprocessor.transform(df)
        
        

        prediction = model.predict(df_processed)
        
        return {"prediction": float(prediction[0])}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=True)
