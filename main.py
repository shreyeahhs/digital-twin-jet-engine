import os
import math
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "./artifacts_v4")
SCALER_PATH  = os.path.join(ARTIFACT_DIR, "scaler.joblib")
POINT_PATH   = os.path.join(ARTIFACT_DIR, "gb_point.joblib")
LO_PATH      = os.path.join(ARTIFACT_DIR, "gb_lo_p10.joblib")
HI_PATH      = os.path.join(ARTIFACT_DIR, "gb_hi_p90.joblib")
CARD_PATH    = os.path.join(ARTIFACT_DIR, "model_card.json")  # optional, for metadata

# ---------- Load artifacts ----------
try:
    scaler = joblib.load(SCALER_PATH)
    gb_point = joblib.load(POINT_PATH)
    gb_lo = joblib.load(LO_PATH)
    gb_hi = joblib.load(HI_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load artifacts from {ARTIFACT_DIR}: {e}")

# Feature order (critical!)
FEATURE_ORDER: Optional[List[str]] = None
if hasattr(scaler, "feature_names_in_"):
    FEATURE_ORDER = list(scaler.feature_names_in_)
else:
    # Optional: try to read from a feature_cols.json you may export later
    feat_json = os.path.join(ARTIFACT_DIR, "feature_cols.json")
    if os.path.exists(feat_json):
        FEATURE_ORDER = json.load(open(feat_json))
if not FEATURE_ORDER:
    raise RuntimeError(
        "Could not determine feature order. Ensure scaler was fitted with a pandas DataFrame "
        "so `scaler.feature_names_in_` is available, or provide artifacts_v4/feature_cols.json."
    )

# Infer KEY_SENSORS from FEATURE_ORDER (e.g., s2, s2_ma, s2_std, s2_diff, ..., plus HI)
def infer_key_sensors(feature_order: List[str]) -> List[str]:
    bases = set()
    for f in feature_order:
        if f == "HI":
            continue
        if f.endswith("_ma") or f.endswith("_std") or f.endswith("_diff"):
            bases.add(f.split("_")[0])
        else:
            # raw sensor like 's2' or 's13' show up raw
            if f.startswith("s"):
                bases.add(f)
    # Sort sensors numerically when possible (s1, s2, s10) and fall back to lexical order
    def _key(x: str):
        # If name is like 's12' -> numeric key (0, 12) so numbers come first and are ordered numerically
        # Otherwise return (1, x) so non-numeric fall after numeric and are ordered lexically
        tail = x[1:]
        if tail.isdigit():
            return (0, int(tail))
        return (1, x)

    return sorted(bases, key=_key)

KEY_SENSORS = infer_key_sensors(FEATURE_ORDER)

# ---------- FastAPI ----------
app = FastAPI(title="Jet Engine RUL Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Schemas ----------
class FeaturePayload(BaseModel):
    """Send precomputed features matching FEATURE_ORDER keys."""
    features: Dict[str, float] = Field(..., description="Mapping of feature name -> value (must include all trained features)")

class WindowRow(BaseModel):
    """One telemetry row from the recent window."""
    engine_id: Optional[int] = None
    cycle: Optional[int] = None
    setting1: Optional[float] = None
    setting2: Optional[float] = None
    setting3: Optional[float] = None
    # sensors s1..s26 (you can send fewer; only those in KEY_SENSORS are used)
    # Pydantic will accept extra keys, which we ignore safely.

class WindowPayload(BaseModel):
    """Recent window of raw rows; service computes rolling features + HI."""
    window: List[Dict[str, float]] = Field(..., description="List of recent rows; must include sX for sensors in KEY_SENSORS (e.g., s2, s3...). Minimum 2 rows recommended.")

class BatchPredictRequest(BaseModel):
    items: List[FeaturePayload]

class PredictResponse(BaseModel):
    rul: float
    p10: float
    p90: float

class MetadataResponse(BaseModel):
    feature_order: List[str]
    key_sensors: List[str]
    artifacts_dir: str
    model_card: Optional[dict] = None

# ---------- Feature builders ----------
def build_features_from_window(window_rows: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Mirror the notebook's OnlineState.features() logic:
    - For each sensor in KEY_SENSORS: raw (last), ma (mean), std, diff (last - prev)
    - HI = -mean(z-score across KEY_SENSORS), computed over the window
    """
    if not window_rows or len(window_rows) < 2:
        raise HTTPException(status_code=400, detail="Window must contain at least 2 rows.")

    df = pd.DataFrame(window_rows).reset_index(drop=True)
    # Ensure sensors exist
    missing = [s for s in KEY_SENSORS if s not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required sensors for this model: {missing}")

    feats = {}
    for c in KEY_SENSORS:
        s = pd.to_numeric(df[c], errors="coerce")
        feats[c] = float(s.iloc[-1])
        feats[f"{c}_ma"] = float(np.nanmean(s))
        feats[f"{c}_std"] = float(np.nanstd(s, ddof=1)) if len(s) > 1 else 0.0
        feats[f"{c}_diff"] = float(s.iloc[-1] - s.iloc[-2]) if len(s) > 1 else 0.0

    # Health Index (HI)
    arr = df[KEY_SENSORS].to_numpy(dtype=float)
    mu = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0) + 1e-6
    z = (arr - mu) / sd
    feats["HI"] = -float(np.nanmean(z))
    return feats

def predict_from_features(feat_map: Dict[str, float]) -> PredictResponse:
    # Check / arrange in training order
    missing = [f for f in FEATURE_ORDER if f not in feat_map]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")
    x = pd.DataFrame([[feat_map[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)
    xs = scaler.transform(x)
    p = float(gb_point.predict(xs)[0])
    lo = float(gb_lo.predict(xs)[0])
    hi = float(gb_hi.predict(xs)[0])
    return PredictResponse(rul=p, p10=lo, p90=hi)

# ---------- Routes ----------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metadata", response_model=MetadataResponse)
def metadata():
    card = None
    if os.path.exists(CARD_PATH):
        try:
            card = json.load(open(CARD_PATH))
        except Exception:
            card = None
    return MetadataResponse(
        feature_order=FEATURE_ORDER,
        key_sensors=KEY_SENSORS,
        artifacts_dir=os.path.abspath(ARTIFACT_DIR),
        model_card=card
    )

@app.post("/predict/features", response_model=PredictResponse)
def predict_with_features(payload: FeaturePayload):
    return predict_from_features(payload.features)

@app.post("/predict/window", response_model=PredictResponse)
def predict_with_window(payload: WindowPayload):
    feat_map = build_features_from_window(payload.window)
    return predict_from_features(feat_map)

@app.post("/predict/batch", response_model=List[PredictResponse])
def predict_batch(req: BatchPredictRequest):
    out = []
    for item in req.items:
        out.append(predict_from_features(item.features))
    return out
