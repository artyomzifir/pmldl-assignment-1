# PMLDL Assignment 1 – MNIST Deployment

This repository contains a minimal solution for **Assignment 1** in the PMLDL course.  
The goal is to prepare data, train a simple CNN on MNIST, expose it via a FastAPI backend, and build a small Streamlit frontend to draw digits and get predictions.

---

## 📂 Project structure

```
.
├── docker-compose.yml
├── README.md
└── src
    ├── api
    │   ├── api.py
    │   ├── Dockerfile
    │   └── requirements.txt
    ├── app
    │   ├── app.py
    │   ├── Dockerfile
    │   └── requirements.txt
    └── prepare
        ├── dataset
        │   └── data_processing.py
        ├── Dockerfile
        ├── model
        │   └── model.py
        └── requirements.txt

````
---

## 🚀 How to run (with Docker)

1. Build and start all services:
   ```bash
   docker compose up --build
   ```

2. Workflow:

   * **prepare**: downloads MNIST, saves `train.pt` / `test.pt`, trains CNN, and saves weights to `simple_nn.pth`.
   * **api**: loads the trained model and serves endpoints:

     * `GET /health` → returns `{"status": "ok"}`
     * `POST /predict` → takes a 28×28 digit array and returns prediction + probabilities.
   * **app**: launches Streamlit UI to draw digits and call the API.

3. Open the frontend in your browser:
   [http://localhost:8501](http://localhost:8501)
