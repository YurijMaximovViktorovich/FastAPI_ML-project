from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
from src.python.model import Model
from utils.params import MODEL_FILE
from fastapi.exceptions import RequestValidationError



FEATURE_NAMES = [
    "age", "cholesterol", "heart_rate", "diabetes", "family_history",
    "alcohol_consumption", "exercise_hours_per_week", "diet",
    "previous_heart_problems", "medication_use", "stress_level",
    "sedentary_hours_per_day", "income", "bmi", "triglycerides",
    "physical_activity_days_per_week", "sleep_hours_per_day", "blood_sugar",
    "ck_mb", "troponin", "gender", "systolic_blood_pressure", "diastolic_blood_pressure"
]


app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

model = Model(model_path=str(MODEL_FILE), threshold=0.425)



@app.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("form.html", {
        "request": request,
        "fields": FEATURE_NAMES,  
        "data": {},
        "prediction": None
    })

@app.post("/predict_manual", response_class=HTMLResponse)
async def predict_manual(
    request: Request,
    age: float = Form(...),
    cholesterol: float = Form(...),
    heart_rate: float = Form(...),
    diabetes: int = Form(...),  
    family_history: int = Form(...),  
    alcohol_consumption: int = Form(...),  
    exercise_hours_per_week: float = Form(...),
    diet: int = Form(...),  
    previous_heart_problems: int = Form(...),  
    medication_use: int = Form(...),  
    stress_level: int = Form(...),  
    sedentary_hours_per_day: float = Form(...),
    income: float = Form(...),
    bmi: float = Form(...),
    triglycerides: float = Form(...),
    physical_activity_days_per_week: int = Form(...),  
    sleep_hours_per_day: float = Form(...),
    blood_sugar: float = Form(...),
    ck_mb: float = Form(...),
    troponin: float = Form(...),
    gender: int = Form(...),  
    systolic_blood_pressure: float = Form(...),
    diastolic_blood_pressure: float = Form(...)
):
    input_data = {
        "age": age,
        "cholesterol": cholesterol,
        "heart_rate": heart_rate,
        "diabetes": diabetes,
        "family_history": family_history,
        "alcohol_consumption": alcohol_consumption,
        "exercise_hours_per_week": exercise_hours_per_week,
        "diet": diet,
        "previous_heart_problems": previous_heart_problems,
        "medication_use": medication_use,
        "stress_level": stress_level,
        "sedentary_hours_per_day": sedentary_hours_per_day,
        "income": income,
        "bmi": bmi,
        "triglycerides": triglycerides,
        "physical_activity_days_per_week": physical_activity_days_per_week,
        "sleep_hours_per_day": sleep_hours_per_day,
        "blood_sugar": blood_sugar,
        "ck_mb": ck_mb,
        "troponin": troponin,
        "gender": gender,
        "systolic_blood_pressure": systolic_blood_pressure,
        "diastolic_blood_pressure": diastolic_blood_pressure
    }
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]

    return templates.TemplateResponse(
        "form.html",
        {
            "request": request,
            "prediction": prediction,
            "data": input_data  
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return templates.TemplateResponse(
        "form.html",
        {
            "request": request,
            "prediction": None,
            "data": {},
            "error": "Некорректные или пустые данные. Пожалуйста, заполните все поля."
        },
        status_code=200 
    )


import re



@app.post("/predict_csv", response_class=HTMLResponse)
async def predict_csv(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        df.columns = df.columns.str.lower().str.replace(r'[\s\-\.]+', '_', regex=True)

        if 'gender' in df.columns:
            df['gender'] = df['gender'].str.lower().map({'male': 1, 'female': 0})

        df = df[[col for col in df.columns if col in FEATURE_NAMES]]

        missing = [feat for feat in FEATURE_NAMES if feat not in df.columns]
        if missing:
            return templates.TemplateResponse(
                "form.html",
                {
                    "request": request,
                    "prediction": None,
                    "data": {},
                    "error": f"В файле не хватает признаков: {', '.join(missing)}"
                },
                status_code=200
            )

        df = df[FEATURE_NAMES]
        predictions = model.predict(df)
        df["prediction"] = predictions

        result_html = df.to_html(classes="table", index=False)

        return templates.TemplateResponse("form.html", {
            "request": request,
            "table": result_html,
            "data": {},
            "prediction": None
        })

    except Exception as e:
        return templates.TemplateResponse(
            "form.html",
            {
                "request": request,
                "error": f"Ошибка при обработке файла: {str(e)}",
                "data": {},
                "prediction": None
            },
            status_code=200
        )
