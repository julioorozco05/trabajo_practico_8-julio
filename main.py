from fastapi import FastAPI, status, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import joblib

app = FastAPI(
    title= "predict linear regression model",
    version= "0.0.1"
)

model = joblib.load("model/logistic_regression_v01.pkl")
print(model)


@app.post("/api/v1/prediction-linear-regression", tags=["linear-regression"])
async def predict(
        Pregnancies: float,
        Glucose: float,
        BloodPressure: float,
        BMI: float,
        DiabetesPedigreeFunction: float,
        Age: float
):
    try:
        # Crear el DataFrame directamente desde el diccionario
        data_frame = pd.DataFrame({
            'Pregnancies': [Pregnancies],
            'Glucose': [Glucose],
            'BloodPressure': [BloodPressure],
            'BMI': [BMI],
            'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
            'Age': [Age]
        })

        # Realizar la predicción
        prediction = model.predict(data_frame)

        prediction_str = "Diabetes" if prediction[0] == 1 else "No Diabetes"

        # Devolver la predicción como JSON
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"prediction": prediction_str}  # Devolver la predicción dentro de un diccionario
        )
    except Exception as e:
        # Manejar los errores
        raise HTTPException(
            detail=str(e),
            status_code=status.HTTP_400_BAD_REQUEST
        )