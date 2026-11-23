"""FastAPI app entrypoint."""

from fastapi import FastAPI

from api.routers import batch, explain, model, predict, predictions

app = FastAPI(title="Customer Behavior Prediction API", version="0.1.0")

app.include_router(model.router)
app.include_router(predict.router)
app.include_router(batch.router)
app.include_router(predictions.router)
app.include_router(explain.router)


@app.get("/health")
def health():
    return {"status": "ok"}
