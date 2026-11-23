"""FastAPI app entrypoint."""

from fastapi import FastAPI

from api.routers import batch, model, predict

app = FastAPI(title="Customer Behavior Prediction API", version="0.1.0")

app.include_router(model.router)
app.include_router(predict.router)
app.include_router(batch.router)


@app.get("/health")
def health():
    return {"status": "ok"}
