import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
# Ly do: bo du lieu Adult co ty le lop 75/25. Mot mo hinh doan bua
# "thu nhap thap" cho moi mau da dat accuracy 0.75 ma khong hoc duoc gi.
F1_THRESHOLD = 0.65

# Ty le lop duong tham chieu cua bo du lieu goc (Bonus 5).
# Lech qua REFERENCE_TOLERANCE so voi muc nay thi canh bao lech lac du lieu.
REFERENCE_POSITIVE_RATE = 0.248
REFERENCE_TOLERANCE = 0.05


def sweep_threshold(y_true, probs):
    """
    Bonus 2: quet nguong quyet dinh tu 0.1 den 0.9 (buoc 0.05) va tra ve
    nguong cho f1 cao nhat.

    model.predict() mac dinh gan nhan 1 khi xac suat vuot 0.5. Voi du lieu
    mat can bang, nguong do hiem khi toi uu vi lop duong bi mo hinh danh gia
    thap mot cach he thong.

    Tra ve: (best_threshold, best_f1)
    """
    best_threshold, best_f1 = 0.5, -1.0

    for threshold in np.arange(0.10, 0.901, 0.05):
        preds = (probs >= threshold).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_threshold, best_f1 = round(float(threshold), 2), float(score)

    return best_threshold, best_f1


def write_detail_report(y_true, preds, path="outputs/detail.txt"):
    """
    Bonus 3: ghi confusion matrix va precision/recall tung lop ra file van ban.
    """
    matrix = confusion_matrix(y_true, preds, labels=[0, 1])
    (tn, fp), (fn, tp) = matrix

    lines = [
        "BAO CAO CHI TIET - PHAN LOAI THU NHAP",
        "",
        "Confusion matrix (hang = thuc te, cot = du doan):",
        "                 du_doan_thap  du_doan_cao",
        f"  thuc_te_thap   {tn:>12}  {fp:>11}",
        f"  thuc_te_cao    {fn:>12}  {tp:>11}",
        "",
        "Chi so tung lop:",
        f"{'lop':<16}{'precision':>12}{'recall':>10}{'support':>10}",
    ]

    for label, name in [(0, "thu_nhap_thap"), (1, "thu_nhap_cao")]:
        p = precision_score(y_true, preds, pos_label=label, zero_division=0)
        r = recall_score(y_true, preds, pos_label=label, zero_division=0)
        support = int((np.asarray(y_true) == label).sum())
        lines.append(f"{name:<16}{p:>12.4f}{r:>10.4f}{support:>10}")

    lines += [
        "",
        f"Bo sot nguoi thu nhap cao (false negative): {fn}",
        f"Gan nham nguoi thu nhap thap (false positive): {fp}",
    ]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))


def check_drift(y_train):
    """
    Bonus 5: canh bao khi ty le lop duong cua tap huan luyen lech qua 5 diem
    phan tram so voi ty le tham chieu 24.8%.

    Tra ve: ty le lop duong thuc te (float).
    """
    positive_rate = float(np.mean(y_train))
    deviation = abs(positive_rate - REFERENCE_POSITIVE_RATE)

    if deviation > REFERENCE_TOLERANCE:
        print(
            f"CANH BAO LECH LAC DU LIEU: ty le lop duong {positive_rate:.1%} "
            f"lech {deviation:.1%} so voi tham chieu {REFERENCE_POSITIVE_RATE:.1%} "
            f"(nguong cho phep {REFERENCE_TOLERANCE:.0%})."
        )
    else:
        print(
            f"Ty le lop duong {positive_rate:.1%}, lech {deviation:.1%} "
            f"so voi tham chieu - trong nguong cho phep."
        )

    return positive_rate


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho GradientBoostingClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia (holdout).

    Tra ve:
        f1 (float): diem F1 cua lop duong (thu nhap > 50K) tren tap holdout,
        tinh o nguong mac dinh 0.5.
    """

    # Doc du lieu huan luyen va danh gia
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # Bonus 5: kiem tra phan phoi truoc khi huan luyen
    positive_rate = check_drift(y_train)

    with mlflow.start_run():

        # Ghi nhan cac sieu tham so
        mlflow.log_params(params)

        # Khoi tao va huan luyen GradientBoostingClassifier
        # random_state=42 de dam bao tinh tai tao giua cac lan chay
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # Du doan tren tap holdout va tinh chi so
        # Chu y: f1_score o day tinh cho LOP DUONG (target = 1), khong dung average.
        preds = model.predict(X_eval)
        f1 = float(f1_score(y_eval, preds))
        acc = float(accuracy_score(y_eval, preds))

        # Bonus 2: quet nguong quyet dinh de tim diem f1 tot nhat
        probs = model.predict_proba(X_eval)[:, 1]
        best_threshold, best_f1 = sweep_threshold(y_eval, probs)

        # Bonus 3: bao cao chi tiet precision/recall tung lop
        write_detail_report(y_eval, preds)

        # Ghi nhan chi so vao MLflow
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("f1_at_best_threshold", best_f1)
        mlflow.log_metric("positive_rate", positive_rate)
        mlflow.sklearn.log_model(model, "model")

        # In ket qua ra man hinh
        print(f"F1: {f1:.4f} | Accuracy: {acc:.4f}")
        print(
            f"Nguong tot nhat: {best_threshold:.2f} -> f1 {best_f1:.4f} "
            f"(nguong mac dinh 0.50 -> f1 {f1:.4f}, chenh {best_f1 - f1:+.4f})"
        )

        # Luu metrics ra file outputs/report.json
        # File nay duoc doc boi GitHub Actions o Buoc 2
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.json", "w") as f:
            json.dump(
                {
                    "f1_score": f1,
                    "accuracy": acc,
                    "best_threshold": best_threshold,
                    "f1_at_best_threshold": best_f1,
                    "positive_rate": positive_rate,
                },
                f,
            )

        # Luu mo hinh ra file models/model.joblib
        # File nay duoc upload len cloud storage o Buoc 2
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
