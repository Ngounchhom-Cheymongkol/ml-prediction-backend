from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from model.PredictionModel import PredictionModel
from service.PredictionService import *
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.post("/preprediction/")
async def say_hello(request: PredictionModel):
    input_data = {
        'CreditScore': request.CreditScore,
        'Geography': request.Geography,
        'Gender': request.Gender,
        'Age': request.Age,
        'Tenure': request.Tenure,
        'Balance': request.Balance,
        'NumOfProducts': request.NumOfProducts,
        'HasCrCard': request.HasCrCard,
        'IsActiveMember': request.IsActiveMember,
        'EstimatedSalary': request.EstimatedSalary
    }
    predict = predict_churn(input_data)
    num = np.float32(predict)

    result = 'Customer will unlikely churn.' if float(num) > 0.5 else 'Customer is likely to churn.'
    return {
        "probability": float(num),
        "prediction": result
    }

