from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import joblib
import os

app = FastAPI()

ARTIFACT_BUCKET = os.environ["ARTIFACT_BUCKET"]
MODEL_KEY = "artifacts/current/model.joblib"
MODEL_PATH = os.path.expanduser("~/models/model.joblib")

# Nhan tra ve tuong ung voi gia tri du doan cua mo hinh
LABELS = {0: "thu_nhap_thap", 1: "thu_nhap_cao"}
N_FEATURES = 10


def download_model():
    """
    Tai file model.joblib tu S3 ve may khi server khoi dong.

    Ham nay duoc goi mot lan khi module duoc import. boto3 tu tim credentials
    theo thu tu: bien moi truong AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY,
    file ~/.aws/credentials, roi den IAM role gan vao EC2 instance.
    """
    s3 = boto3.client("s3")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    s3.download_file(ARTIFACT_BUCKET, MODEL_KEY, MODEL_PATH)

    print(f"Model da duoc tai xuong tu s3://{ARTIFACT_BUCKET}/{MODEL_KEY}")


download_model()
model = joblib.load(MODEL_PATH)


class ScoreRequest(BaseModel):
    features: list[float]


@app.get("/healthz")
def healthz():
    """
    Endpoint kiem tra suc khoe server.
    GitHub Actions goi endpoint nay sau khi deploy de xac nhan server dang chay.

    Tra ve: {"status": "ok"}
    """
    return {"status": "ok"}


@app.post("/score")
def score(req: ScoreRequest):
    """
    Endpoint suy luan chinh.

    Dau vao : JSON {"features": [f1, f2, ..., f10]}
    Dau ra  : JSON {"prediction": <0|1>, "label": <"thu_nhap_thap"|"thu_nhap_cao">}

    Thu tu 10 dac trung (khop voi thu tu trong FEATURE_NAMES cua test):
        age, workclass, education_num, marital_status, occupation,
        relationship, sex, capital_gain, capital_loss, hours_per_week
    """
    if len(req.features) != N_FEATURES:
        raise HTTPException(
            status_code=400,
            detail="Expected 10 features (adult income)",
        )

    pred = int(model.predict([req.features])[0])

    return {"prediction": pred, "label": LABELS[pred]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
