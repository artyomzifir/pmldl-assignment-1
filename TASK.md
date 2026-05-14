# PMLDL Assignment 1: Deployment

## Overview

By the end of this tutorial, you will be able to deploy a trained machine learning model as a containerized REST API and connect it to a web interface — the same pattern used in production ML systems at companies like Yandex, Sber, and Avito.

---

## Prerequisites & Tools

Before you start, make sure you have the following installed and configured:

**Environment**
- Python 3.8+
- Docker and Docker Compose
- Git

**Knowledge**
- Basic Python and ML model training
- Familiarity with REST APIs (what a request/response looks like)

**Tools used in this tutorial**
- [FastAPI](https://fastapi.tiangolo.com/) — framework for building the model API
- [Streamlit](https://docs.streamlit.io/) — framework for the web interface
- [Docker](https://docker-curriculum.com/) — containerization for both services

---

## Steps

### 1. Set Up the Repository

Create a public GitHub repository and clone it locally:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

Create the project structure:

```
├── code
│   ├── datasets
│   ├── deployment
│   │   ├── api
│   │   └── app
│   └── models
├── data
└── models
```

**Expected result:** Your local repository matches the structure above.

---

### 2. Train and Save the Model

Write a Python script to train your model and save it to the `models/` folder. Place the training code in `code/models/`.

Here is an example of how your training script might look:

```python
# example: code/models/train.py
import pickle
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier

X, y = load_digits(return_X_y=True)
model = MLPClassifier().fit(X, y)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)
```

**Expected result:** `models/model.pkl` exists and can be loaded with `pickle.load()`.

---

### 3. Create the Model API

Write a FastAPI script that loads your model and exposes a prediction endpoint. Place it in `code/deployment/api/`.

```python
# example: code/deployment/api/main.py
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

class InputData(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: InputData):
    prediction = model.predict([data.features])
    return {"prediction": int(prediction[0])}
```

Then write a Dockerfile for the API:

```dockerfile
# example: code/deployment/api/Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install fastapi uvicorn scikit-learn numpy
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Expected result:** API container starts without errors and responds at `http://localhost:8000/predict`.

---

### 4. Create the Web Application

Write a Streamlit app that sends input data to the API and displays the prediction. Place it in `code/deployment/app/`.

```python
# example: code/deployment/app/app.py
import streamlit as st
import requests

st.title("ML Model Prediction")

features = st.text_input("Enter features (comma-separated)", "0.1, 0.2, 0.3")

if st.button("Predict"):
    data = {"features": [float(x) for x in features.split(",")]}
    response = requests.post("http://api:8000/predict", json=data)
    st.write("Prediction:", response.json()["prediction"])
```

Then write a Dockerfile for the app:

```dockerfile
# example: code/deployment/app/Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install streamlit requests
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

**Expected result:** Web application opens at `http://localhost:8501`, accepts input, and displays a prediction from the API.

---

### 5. Connect the Services with Docker Compose

Write a `docker-compose.yml` that runs both containers together. Place it in `code/deployment/`.

```yaml
# example: code/deployment/docker-compose.yml
services:
  api:
    build: ./api
    ports:
      - "8000:8000"
    volumes:
      - ../../../models:/app/models

  app:
    build: ./app
    ports:
      - "8501:8501"
    depends_on:
      - api
```

To build and run both services:

```bash
docker-compose up --build
```

**Expected result:** Both containers start successfully. The web application at `http://localhost:8501` communicates with the API at `http://localhost:8000`.

---

## Common Mistakes

**Volume mount path in Docker Compose**
The `models/` folder must be mounted correctly relative to the `docker-compose.yml` location. If the API container can't find the model file, double-check the volume path:
```yaml
volumes:
  - ../../../models:/app/models
```

**Mismatched endpoints**
The URL in the Streamlit app must exactly match the API endpoint name and port. If you rename `/predict` in FastAPI, update it in the Streamlit `requests.post()` call as well.

**Absolute vs relative paths**
Avoid hardcoded absolute paths like `/home/user/project/models/model.pkl` inside containers — they will break. Use paths relative to `WORKDIR` defined in your Dockerfile.

---

## Recap & Next Steps

In this tutorial, you have:

- Trained and saved an ML model
- Built a REST API with FastAPI and containerized it with Docker
- Built a web interface with Streamlit and containerized it with Docker
- Connected both services with Docker Compose

**Next Steps**

If you want to go further, consider:

- Adding input validation and error handling to the API
- Logging predictions to a database
- Automating the full pipeline with Airflow and MLflow

---

## Grading Criteria

| Criteria | Points |
|---|---|
| Model API container works correctly | 2 |
| Web application container works correctly | 2 |
| Repository is structured logically | 1 |
| **Total** | **5** |
