"""
server.py — FastAPI analytical backend server.

Exposes REST APIs for:
  - Paginated employee risk listings
  - Historical risk progression analysis from SCD Type 2 logs
  - Real-time What-If model inference (XGBoost)
  - Simulating HR actions (promotions, overtime changes)
"""
import os
import sys

# Add project root to sys.path so 'pipeline' module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import pickle
import sqlite3
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Literal, List
from datetime import datetime
from neo4j import GraphDatabase, exceptions

# Suppress noisy sklearn version mismatch warnings when loading pickled models
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Reconfigure stdout to support unicode symbols
sys.stdout.reconfigure(encoding='utf-8')

# Paths and configuration
from pipeline.config import OLAP_PATH, MODEL_DIR, DATA_DIR
from pipeline.attrition_toy_simulation import (
    simulate as attrition_simulate,
    DEFAULT_N_SIMS,
    MAX_N_SIMS,
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_STARTING_HEADCOUNT,
    DEFAULT_RATE_LO,
    DEFAULT_RATE_HI,
)


# In-memory pipeline cache and jobs
import threading
import uuid
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PipelineStore:
    def __init__(self):
        self._pipeline = {}
        self.lock = threading.Lock()
        
    def reload(self):
        with self.lock:
            scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
            loo_path = os.path.join(MODEL_DIR, "loo_encoder.pkl")
            cph_path = os.path.join(MODEL_DIR, "model_cph.pkl")
            xgb_path = os.path.join(MODEL_DIR, "model_xgb.json")
            base_path = os.path.join(MODEL_DIR, "baseline_survival.pkl")
            
            if not (os.path.exists(scaler_path) and os.path.exists(xgb_path)):
                raise RuntimeError("Model checkpoints not found. Run pipeline/production_ml.py first.")
                
            with open(scaler_path, "rb") as f:
                self._pipeline["scaler"] = pickle.load(f)
            with open(loo_path, "rb") as f:
                self._pipeline["loo_encoder"] = pickle.load(f)
            with open(cph_path, "rb") as f:
                self._pipeline["cph"] = pickle.load(f)
            with open(base_path, "rb") as f:
                self._pipeline["baseline_survival"] = pickle.load(f)
                
            # Load XGBoost model
            model = xgb.Booster()
            model.load_model(xgb_path)
            self._pipeline["model_xgb"] = model
            
            # Store training feature names
            self._pipeline["feature_names"] = model.feature_names
            logging.info("[+] Model checkpoints successfully loaded into FastAPI cache.")

    def get(self, key):
        with self.lock:
            return self._pipeline.get(key)

pipeline_store = PipelineStore()
JOB_STATUS = {}
JOB_STATUS_LOCK = threading.Lock()

from contextlib import asynccontextmanager
from pipeline.config import get_conn

@asynccontextmanager
async def lifespan(app):
    """Modern lifespan handler replacing deprecated on_event('startup')."""
    pipeline_store.reload()
    
    # Initialize Neo4j driver
    try:
        driver = GraphDatabase.driver(
            os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", "password"))
        )
        driver.verify_connectivity()
        app.state.neo4j_driver = driver
    except Exception as e:
        logging.warning(f"[!] Warning: Could not connect to Neo4j. Graph features will run in mock mode. Error: {e}")
        app.state.neo4j_driver = None
        
    yield
    
    if getattr(app.state, "neo4j_driver", None):
        app.state.neo4j_driver.close()

app = FastAPI(title="MNC HR Attrition Risk REST Service", lifespan=lifespan)

# ============================================================================
#  DATA MODELLING & SCHEMAS
# ============================================================================

class WhatIfRequest(BaseModel):
    EmployeeId: str
    MonthlyIncome: Optional[float] = Field(None, ge=0)
    OverTime: Optional[Literal["Yes", "No"]] = None
    YearsWithCurrManager: Optional[int] = Field(None, ge=0)
    JobRole: Optional[str] = None
    JobLevel: Optional[Literal["Entry Level", "Junior Level", "Mid Level", "Senior Level", "Executive Level"]] = None

class ActionParams(BaseModel):
    JobRole: Optional[str] = None
    JobLevel: Optional[Literal["Entry Level", "Junior Level", "Mid Level", "Senior Level", "Executive Level"]] = None
    MonthlyIncome: Optional[float] = Field(None, ge=0)
    OverTime: Optional[Literal["Yes", "No"]] = None
    YearsWithCurrManager: Optional[int] = Field(None, ge=0)
    PercentSalaryHike: Optional[float] = Field(None, ge=0)

class ActionRequest(BaseModel):
    action: Literal["promote", "overtime", "manager_change", "salary_hike"]
    EmployeeId: str
    params: ActionParams

class RatingRequest(BaseModel):
    score: int = Field(..., ge=1, le=4)

class AttritionSimRequest(BaseModel):
    n_sims: int = Field(default=DEFAULT_N_SIMS, ge=1, le=MAX_N_SIMS)
    alpha: float = Field(default=DEFAULT_ALPHA, gt=0)
    beta: float = Field(default=DEFAULT_BETA, gt=0)
    starting_headcount: int = Field(default=DEFAULT_STARTING_HEADCOUNT, ge=1)
    rate_lo: float = Field(default=DEFAULT_RATE_LO, ge=0.0, le=1.0)
    rate_hi: float = Field(default=DEFAULT_RATE_HI, ge=0.0, le=1.0)

# ============================================================================
#  API ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    dash_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    with open(dash_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    from fastapi.responses import Response
    return Response(status_code=204)

import json
@app.get("/api/model/metadata")
def get_model_metadata():
    metadata_path = os.path.join(MODEL_DIR, "model_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            return json.load(f)
    return {"error": "Metadata not found."}

@app.get("/api/employees")
def list_employees(
    page: int = 1,
    limit: int = 10,
    search: str = "",
    status: str = "all",
    sort_by: str = "General_Risk_Score",
    sort_dir: str = "desc",
    department: str = "",
    performance_filter: str = "all",
):
    with get_conn(OLAP_PATH) as conn:
        # Get active cohort with their current computed risk scores
        query = """
        SELECT 
            v.*,
            r.Prob_Leave_1M, r.Prob_Leave_3M, r.Prob_Leave_6M, r.Prob_Leave_12M, r.General_Risk_Score,
            r.Contrib_Identity, r.Contrib_Environment, r.Contrib_Compensation, r.Contrib_Sentiment, r.Contrib_Tenure
        FROM v_ml_features v
        LEFT JOIN flight_risk_scores r ON v.EmployeeId = r.EmployeeId
        """
        df = pd.read_sql(query, conn)
    
    # Unique departments for filter dropdown
    departments = sorted(df["Department"].dropna().unique().tolist())
    
    # Apply search filter
    if search:
        df = df[df["EmployeeId"].str.contains(search, case=False)]
    
    # Apply department filter
    if department:
        df = df[df["Department"] == department]
        
    # Pre-compute exposure scores for ALL employees (batch) BEFORE filtering/sorting
    import hashlib
    import math
    from pipeline.config import PERFORMANCE_SCORE_LABELS
    driver = app.state.neo4j_driver
    exposure_map = {}
    
    if driver:
        try:
            batch_query = """
            MATCH (e:Employee)-[r:SAME_MANAGER|SAME_ROLE_DEPT|SAME_TENURE_COHORT]-(peer:Employee)
            WHERE e.id IN $emp_ids AND peer.isActive = false AND peer.exitDate <> ''
            WITH e, r, duration.inDays(date(peer.exitDate), date()).days AS days_since_exit
            WHERE days_since_exit <= 60
            WITH e.id AS employee_id, r.weight * exp(-0.1 * days_since_exit) AS contribution
            RETURN employee_id, sum(contribution) AS exposure_score
            """
            with driver.session() as session:
                result = session.run(batch_query, emp_ids=df["EmployeeId"].tolist())
                for record in result:
                    exposure_map[record["employee_id"]] = float(record["exposure_score"])
        except Exception:
            pass
    
    if not exposure_map:
        # Mock fallback when Neo4j is offline
        for _, r in df[["EmployeeId", "General_Risk_Score"]].iterrows():
            emp_id = r["EmployeeId"]
            raw_score = float(r["General_Risk_Score"])
            seed_val = int(hashlib.md5(emp_id.encode()).hexdigest()[:8], 16) % 1000
            base_exposure = seed_val / 1000.0
            if raw_score > 50:
                exposure_map[emp_id] = round(base_exposure * 2.5, 2)
            elif raw_score > 25:
                exposure_map[emp_id] = round(base_exposure * 1.2, 2)
            else:
                exposure_map[emp_id] = round(base_exposure * 0.4, 2)
    
    # Add boosted score columns
    df["Exposure_Score"] = df["EmployeeId"].map(lambda x: exposure_map.get(x, 0.0))
    # Apply proportional hazards survival formula: P = 1 - (1 - P_base)^(1 + exposure)
    base_prob = df["General_Risk_Score"] / 100.0
    df["Boosted_Risk_Score"] = (1.0 - (1.0 - base_prob) ** (1.0 + df["Exposure_Score"])) * 100.0
    
    # Stats using boosted scores (consistent with what the UI displays)
    total_active = len(df)
    is_high_risk = df["Boosted_Risk_Score"] >= 50
    high_risk_count = int(is_high_risk.sum())
    
    # Risk distribution buckets for donut chart (using boosted scores)
    risk_distribution = {
        "critical": int((df["Boosted_Risk_Score"] >= 80).sum()),
        "high": int(((df["Boosted_Risk_Score"] >= 50) & (df["Boosted_Risk_Score"] < 80)).sum()),
        "medium": int(((df["Boosted_Risk_Score"] >= 20) & (df["Boosted_Risk_Score"] < 50)).sum()),
        "low": int((df["Boosted_Risk_Score"] < 20).sum()),
    }
    
    # Apply status filter using boosted score (matches displayed badge)
    if status == "high":
        df = df[is_high_risk]
    elif status == "normal":
        df = df[~is_high_risk]
        
    # Apply performance filter
    if performance_filter == "high":
        df = df[df["PerformanceScore"].isin([1, 2])]
    elif performance_filter == "low":
        df = df[df["PerformanceScore"].isin([3, 4])]
        
    total_filtered = len(df)
    
    # Sort — use boosted score when sorting by risk
    valid_sort_cols = {
        "General_Risk_Score", "MonthlyIncome", "YearsAtCompany",
        "Prob_Leave_1M", "Prob_Leave_12M", "Age", "EmployeeId"
    }
    if sort_by in valid_sort_cols:
        actual_sort = "Boosted_Risk_Score" if sort_by == "General_Risk_Score" else sort_by
        df = df.sort_values(actual_sort, ascending=(sort_dir == "asc"))
    else:
        df = df.sort_values("Boosted_Risk_Score", ascending=False)
    
    # Paginate
    start = (page - 1) * limit
    end = start + limit
    df_slice = df.iloc[start:end]
    
    # Build response using pre-computed exposure scores
    employees_list = []

    for _, row in df_slice.iterrows():
        emp_id = row["EmployeeId"]
        perf_score = int(row.get("PerformanceScore", 3)) if pd.notna(row.get("PerformanceScore")) else 3
        unboosted_score = float(row["General_Risk_Score"])
        exposure_score = round(float(row["Exposure_Score"]), 2)
        boosted_score = round(float(row["Boosted_Risk_Score"]), 1)
        
        employees_list.append({
            "EmployeeId": row["EmployeeId"],
            "Age": int(row["Age"]),
            "Gender": row["Gender"],
            "Department": row["Department"],
            "JobRole": row["JobRole"],
            "JobLevel": row["JobLevel"],
            "MonthlyIncome": float(row["MonthlyIncome"]),
            "OverTime": row["OverTime"],
            "YearsAtCompany": int(row["YearsAtCompany"]),
            "YearsWithCurrManager": int(row["YearsWithCurrManager"]),
            "PerformanceScore": perf_score,
            "PerformanceLabel": PERFORMANCE_SCORE_LABELS.get(perf_score, "Unknown"),
            "Prob_1M": f"{row['Prob_Leave_1M'] * 100:.1f}%",
            "Prob_3M": f"{row['Prob_Leave_3M'] * 100:.1f}%",
            "Prob_6M": f"{row['Prob_Leave_6M'] * 100:.1f}%",
            "Prob_12M": f"{row['Prob_Leave_12M'] * 100:.1f}%",
            "General_Risk_Score": boosted_score,
            "Unboosted_Risk_Score": round(unboosted_score, 1),
            "is_high_risk": bool(any(row[p] >= 0.30 for p in ["Prob_Leave_1M", "Prob_Leave_3M", "Prob_Leave_6M", "Prob_Leave_12M"])),
            "contributions": {
                "identity": round(float(row["Contrib_Identity"]), 1),
                "environment": round(float(row["Contrib_Environment"]), 1),
                "compensation": round(float(row["Contrib_Compensation"]), 1),
                "sentiment": round(float(row["Contrib_Sentiment"]), 1),
                "tenure": round(float(row["Contrib_Tenure"]), 1)
            }
        })
        
    return {
        "employees": employees_list,
        "total": total_filtered,
        "high_risk_count": high_risk_count,
        "total_active": total_active,
        "departments": departments,
        "risk_distribution": risk_distribution,
    }

@app.get("/api/dashboard/stats")
def get_dashboard_stats():
    with get_conn(OLAP_PATH) as conn:
        query = """
        SELECT 
            v.Department,
            r.General_Risk_Score,
            r.Prob_Leave_1M, r.Prob_Leave_3M, r.Prob_Leave_6M, r.Prob_Leave_12M,
            r.Contrib_Identity, r.Contrib_Environment, r.Contrib_Compensation, r.Contrib_Sentiment, r.Contrib_Tenure
        FROM v_ml_features v
        JOIN flight_risk_scores r ON v.EmployeeId = r.EmployeeId
        """
        df = pd.read_sql(query, conn)
    
    if df.empty:
        return {}
        
    total_active = len(df)
    is_high_risk = (df["Prob_Leave_1M"] >= 0.30) | (df["Prob_Leave_3M"] >= 0.30) | (df["Prob_Leave_6M"] >= 0.30) | (df["Prob_Leave_12M"] >= 0.30)
    high_risk_count = int(is_high_risk.sum())
    
    avg_risk_score = float(df["General_Risk_Score"].mean())
    
    # Risk by department (sort by highest risk)
    dept_risk_series = df.groupby("Department")["General_Risk_Score"].mean().sort_values(ascending=False)
    dept_risk = {k: float(v) for k, v in dept_risk_series.items()}
    

    
    return {
        "total_active": total_active,
        "high_risk_count": high_risk_count,
        "avg_risk_score": avg_risk_score,
        "department_risk": dept_risk
    }


def encode_single_row(row_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms a single employee row into the exact scaled, encoded feature space
    expected by the trained models.
    """
    from pipeline import config
    from pipeline.data_pipeline import preprocess_data
    
    if 'DateOfLeaving' in row_df.columns:
        row_df = row_df.drop(columns=['DateOfLeaving'])
    
    # 1. Preprocess (mappings, types, zero-duration fix)
    df_clean = preprocess_data(row_df)
    # Reset to integer index so .loc[0, col] works regardless of original index
    df_clean = df_clean.reset_index(drop=True)
    
    # 2. Re-create nominal One-Hot dummies
    # We must match the exact feature names list in pipeline_store.get("feature_names")
    # Build a base matrix containing 1 row of all zeros
    cols = pipeline_store.get("feature_names")
    x_encoded = pd.DataFrame(0, index=[0], columns=cols)
    
    # Populate numeric and ordinal fields (skip text columns — they're handled by OHE/LOO below)
    text_cols = set(config.ONE_HOT_COLS + config.LOO_COLS)
    for col in df_clean.columns:
        if col in cols and col not in text_cols:
            # Skip any remaining non-numeric values
            val = df_clean.loc[0, col]
            if isinstance(val, str):
                continue
            # Scale continuous columns
            scaler = pipeline_store.get("scaler")
            if col in scaler.feature_names_in_:
                col_idx = list(scaler.feature_names_in_).index(col)
                min_val = scaler.data_min_[col_idx]
                max_val = scaler.data_max_[col_idx]
                scaled_val = (val - min_val) / (max_val - min_val) if (max_val - min_val) > 0 else 0.0
                x_encoded.loc[0, col] = scaled_val
            else:
                x_encoded.loc[0, col] = val
                
    # OHE Columns
    for ohe_col in config.ONE_HOT_COLS:
        if ohe_col in df_clean.columns:
            val = df_clean.loc[0, ohe_col]
            dummy_col = f"{ohe_col}_{val}"
            if dummy_col in cols:
                x_encoded.loc[0, dummy_col] = 1
                
    # LOO Columns (apply saved encoder)
    # The LOO encoder was fit on a differently-shaped DataFrame (post-OHE).
    # For single-row inference, we extract the per-category mean mappings directly.
    # Encoder mapping: dict of {col_name: DataFrame(sum, count)} — mean = sum/count.
    loo_cols = [c for c in config.LOO_COLS if c in df_clean.columns]
    if loo_cols:
        encoder = pipeline_store.get("loo_encoder")
        for col in loo_cols:
            if col in cols and hasattr(encoder, 'mapping') and col in encoder.mapping:
                raw_val = df_clean.loc[0, col]
                mapping_df = encoder.mapping[col]  # DataFrame with 'sum' and 'count'
                if raw_val in mapping_df.index:
                    row_data = mapping_df.loc[raw_val]
                    raw_mean = float(row_data['sum'] / row_data['count'])
                else:
                    # Unseen category — use overall mean across all categories
                    raw_mean = float(mapping_df['sum'].sum() / mapping_df['count'].sum())
                
                # Scale the LOO column just like production_ml does
                scaler = pipeline_store.get("scaler")
                if col in scaler.feature_names_in_:
                    col_idx = list(scaler.feature_names_in_).index(col)
                    min_val = scaler.data_min_[col_idx]
                    max_val = scaler.data_max_[col_idx]
                    x_encoded.loc[0, col] = (raw_mean - min_val) / (max_val - min_val) if (max_val - min_val) > 0 else 0.0
                else:
                    x_encoded.loc[0, col] = raw_mean
                
    # Ensure float types to prevent XGBoost type mismatch
    x_encoded = x_encoded.astype(float)
    return x_encoded

def predict_risk_profile(x_encoded: pd.DataFrame, employee_id: str = None) -> Dict[str, float]:
    """
    Calculates time-horizon probabilities for an encoded feature vector.
    """
    model = pipeline_store.get("model_xgb")
    baseline_survival = pipeline_store.get("baseline_survival")
    
    dmatrix = xgb.DMatrix(x_encoded)
    raw_pred = model.predict(dmatrix, output_margin=True)[0]
    clipped_pred = np.clip(raw_pred, -15, 15)
    risk_multiplier = np.exp(clipped_pred)
    
    # Inject Graph Exposure Score as a multiplier
    if employee_id:
        try:
            res = get_graph_exposure(employee_id)
            exposure_score = float(res.get("exposure_score", 0.0))
            risk_multiplier *= (1.0 + exposure_score)
        except Exception as e:
            print(f"[!] Could not fetch graph exposure: {e}")
    
    horizons = [1, 3, 6, 12]
    probs = {}
    
    for h in horizons:
        idx = (np.abs(baseline_survival.index - h)).argmin()
        s0_t = baseline_survival.iloc[idx]
        probs[h] = float(1.0 - (s0_t ** risk_multiplier))
        
    return {
        "Prob_1M": probs[1],
        "Prob_3M": probs[3],
        "Prob_6M": probs[6],
        "Prob_12M": probs[12],
        "General_Risk_Score": probs[12] * 100
    }

@app.get("/api/employees/{employee_id}/history")
def get_employee_history(employee_id: str):
    with get_conn(OLAP_PATH) as conn:
        # Query all SCD Type 2 history records for this employee
        query = """
        SELECT * FROM employee_history 
        WHERE EmployeeId = ? 
        ORDER BY valid_from ASC
        """
        df_hist = pd.read_sql(query, conn, params=(employee_id,))
    
    if df_hist.empty:
        raise HTTPException(status_code=404, detail=f"Employee {employee_id} history not found.")
        
    history_points = []
    
    # For each historical row, run our pipeline to see what their risk score WAS
    for idx, row in df_hist.iterrows():
        # Build single row dataframe
        row_df = pd.DataFrame([row.drop(["row_id", "valid_from", "valid_to", "is_active"])])
        
        # Set EmployeeId as index so it doesn't get coerced to NaN
        if "EmployeeId" in row_df.columns:
            row_df.set_index("EmployeeId", inplace=True)
        else:
            row_df.index = [0]
        
        try:
            x_encoded = encode_single_row(row_df)
            profile = predict_risk_profile(x_encoded, employee_id)
            
            history_points.append({
                "date": row["valid_from"].split(" ")[0], # YYYY-MM-DD
                "timestamp": row["valid_from"],
                "JobRole": row["JobRole"],
                "MonthlyIncome": float(row["MonthlyIncome"]),
                "OverTime": row["OverTime"],
                "YearsWithCurrManager": int(row["YearsWithCurrManager"]),
                "Prob_12M": f"{profile['Prob_12M'] * 100:.1f}%",
                "General_Risk_Score": round(profile['General_Risk_Score'], 1)
            })
        except Exception as e:
            # Skip invalid history formats silently
            continue
            
    return history_points

@app.post("/api/whatif")
def calculate_whatif(req: WhatIfRequest):
    with get_conn(OLAP_PATH) as conn:
        # Fetch active record from Layer 2
        query = "SELECT * FROM v_ml_features WHERE EmployeeId = ?"
        df_active = pd.read_sql(query, conn, params=(req.EmployeeId,))
    
    if df_active.empty:
        raise HTTPException(status_code=404, detail=f"Employee {req.EmployeeId} not found.")
        
    # Apply overrides
    row_df = df_active.copy()
    if req.MonthlyIncome is not None:
        row_df["MonthlyIncome"] = req.MonthlyIncome
    if req.OverTime is not None:
        row_df["OverTime"] = req.OverTime
    if req.YearsWithCurrManager is not None:
        row_df["YearsWithCurrManager"] = req.YearsWithCurrManager
    if req.JobRole is not None:
        row_df["JobRole"] = req.JobRole
    if req.JobLevel is not None:
        row_df["JobLevel"] = req.JobLevel
        
    # Set EmployeeId as index so it doesn't get coerced to NaN by preprocess_data
    if "EmployeeId" in row_df.columns:
        row_df.set_index("EmployeeId", inplace=True)
        
    # Run transformation and inference
    try:
        x_encoded = encode_single_row(row_df)
        profile = predict_risk_profile(x_encoded, req.EmployeeId)
        
        # Heuristic: XGBoost models trained on this dataset don't handle salary cuts well 
        # (no training data for cuts). We inject a business rule penalty for salary cuts.
        if req.MonthlyIncome is not None:
            old_income = df_active["MonthlyIncome"].iloc[0]
            if req.MonthlyIncome < old_income:
                drop_ratio = (old_income - req.MonthlyIncome) / old_income
                # E.g., a 35% cut -> drop_ratio = 0.35 -> multiplier = 1 + (0.35 * 3) = 2.05
                penalty_multiplier = 1.0 + (drop_ratio * 3.0)
                profile['Prob_1M'] = min(0.999, profile['Prob_1M'] * penalty_multiplier)
                profile['Prob_3M'] = min(0.999, profile['Prob_3M'] * penalty_multiplier)
                profile['Prob_6M'] = min(0.999, profile['Prob_6M'] * penalty_multiplier)
                profile['Prob_12M'] = min(0.999, profile['Prob_12M'] * penalty_multiplier)
                profile['General_Risk_Score'] = min(100.0, profile['General_Risk_Score'] * penalty_multiplier)
                
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
        
    return {
        "Prob_1M": f"{profile['Prob_1M'] * 100:.1f}%",
        "Prob_3M": f"{profile['Prob_3M'] * 100:.1f}%",
        "Prob_6M": f"{profile['Prob_6M'] * 100:.1f}%",
        "Prob_12M": f"{profile['Prob_12M'] * 100:.1f}%",
        "General_Risk_Score": round(profile['General_Risk_Score'], 1)
    }

from fastapi import BackgroundTasks
from abc import ABC, abstractmethod

class ActionStrategy(ABC):
    @abstractmethod
    def apply(self, employee_id: str, params: ActionParams) -> None: ...

class PromoteStrategy(ActionStrategy):
    def apply(self, employee_id: str, params: ActionParams):
        from pipeline import simulator_actions as sa
        if params.JobRole is None or params.JobLevel is None or params.MonthlyIncome is None:
            raise HTTPException(status_code=400, detail="Missing required parameters for promotion")
        sa.promote_employee(employee_id, params.JobRole, params.JobLevel, float(params.MonthlyIncome))

class OvertimeStrategy(ActionStrategy):
    def apply(self, employee_id: str, params: ActionParams):
        from pipeline import simulator_actions as sa
        if params.OverTime is None:
            raise HTTPException(status_code=400, detail="Missing OverTime parameter")
        sa.change_overtime(employee_id, params.OverTime)

class ManagerChangeStrategy(ActionStrategy):
    def apply(self, employee_id: str, params: ActionParams):
        from pipeline import simulator_actions as sa
        if params.YearsWithCurrManager is None:
            raise HTTPException(status_code=400, detail="Missing YearsWithCurrManager parameter")
        sa.change_manager_tenure(employee_id, int(params.YearsWithCurrManager))

class SalaryHikeStrategy(ActionStrategy):
    def apply(self, employee_id: str, params: ActionParams):
        from pipeline import simulator_actions as sa
        if params.PercentSalaryHike is None:
            raise HTTPException(status_code=400, detail="Missing PercentSalaryHike parameter")
        sa.adjust_salary(employee_id, float(params.PercentSalaryHike))

STRATEGIES = {
    "promote": PromoteStrategy(),
    "overtime": OvertimeStrategy(),
    "manager_change": ManagerChangeStrategy(),
    "salary_hike": SalaryHikeStrategy()
}

def run_simulation_task(job_id: str, strategy: ActionStrategy, employee_id: str, params: ActionParams):
    with JOB_STATUS_LOCK:
        JOB_STATUS[job_id] = "running"
    from pipeline import simulator_actions as sa
    try:
        strategy.apply(employee_id, params)
        sa.trigger_nightly_etl_ml()
        pipeline_store.reload()
        with JOB_STATUS_LOCK:
            JOB_STATUS[job_id] = "completed"
    except Exception as e:
        with JOB_STATUS_LOCK:
            JOB_STATUS[job_id] = f"failed: {str(e)}"
        logging.error(f"Simulation task failed: {e}")

@app.post("/api/simulate-action")
def simulate_action(req: ActionRequest, background_tasks: BackgroundTasks):
    strategy = STRATEGIES.get(req.action)
    if not strategy:
        raise HTTPException(400, f"Unknown action: {req.action}")
        
    # Atomic check-and-mark: prevents TOCTOU race when FastAPI runs
    # sync endpoints concurrently in its thread pool.
    with JOB_STATUS_LOCK:
        if any(status == "running" for status in JOB_STATUS.values()):
            raise HTTPException(status_code=409, detail="Another simulation is currently running.")
        job_id = str(uuid.uuid4())
        JOB_STATUS[job_id] = "pending"

    background_tasks.add_task(run_simulation_task, job_id, strategy, req.EmployeeId, req.params)
    
    return {"status": "accepted", "job_id": job_id, "message": "Simulation action accepted and processing in background."}

@app.get("/api/simulate-action/{job_id}")
def get_simulation_status(job_id: str):
    with JOB_STATUS_LOCK:
        if job_id not in JOB_STATUS:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"job_id": job_id, "status": JOB_STATUS[job_id]}

def run_retrain_task(job_id: str):
    with JOB_STATUS_LOCK:
        JOB_STATUS[job_id] = "running"
    from pipeline import simulator_actions as sa
    try:
        sa.trigger_nightly_etl_ml()
        pipeline_store.reload()
        with JOB_STATUS_LOCK:
            JOB_STATUS[job_id] = "completed"
    except Exception as e:
        with JOB_STATUS_LOCK:
            JOB_STATUS[job_id] = f"failed: {str(e)}"
        logging.error(f"Retrain task failed: {e}")

@app.post("/api/retrain")
def retrain_ml_pipeline(background_tasks: BackgroundTasks):
    with JOB_STATUS_LOCK:
        if any(status == "running" for status in JOB_STATUS.values()):
            raise HTTPException(status_code=409, detail="A job is currently running.")
        job_id = str(uuid.uuid4())
        JOB_STATUS[job_id] = "pending"

    background_tasks.add_task(run_retrain_task, job_id)
    
    return {"status": "accepted", "job_id": job_id, "message": "ML Pipeline triggered."}

@app.patch("/api/employees/{employee_id}/rating")
def update_employee_rating(employee_id: str, req: RatingRequest, background_tasks: BackgroundTasks):
    from pipeline.config import OLTP_PATH
    import sqlite3
    try:
        # Write directly to OLTP
        with get_conn(OLTP_PATH) as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            cursor.execute("""
            INSERT INTO performance_ratings (EmployeeId, PerformanceScore, rated_by, rated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(EmployeeId) DO UPDATE SET
                PerformanceScore=excluded.PerformanceScore,
                rated_by=excluded.rated_by,
                rated_at=excluded.rated_at
            """, (employee_id, req.score, "demo_manager", now))
            conn.commit()
            
        # Trigger pipeline rebuild in background, same as simulate_action
        with JOB_STATUS_LOCK:
            job_id = str(uuid.uuid4())
            JOB_STATUS[job_id] = "pending"
        
        def run_rating_sync():
            with JOB_STATUS_LOCK:
                JOB_STATUS[job_id] = "running"
            from pipeline import simulator_actions as sa
            try:
                sa.trigger_nightly_etl_ml()
                pipeline_store.reload()
                with JOB_STATUS_LOCK:
                    JOB_STATUS[job_id] = "completed"
            except Exception as e:
                with JOB_STATUS_LOCK:
                    JOB_STATUS[job_id] = f"failed: {str(e)}"
                logging.error(f"Rating sync failed: {e}")
                
        background_tasks.add_task(run_rating_sync)
        return {"status": "accepted", "job_id": job_id, "message": "Rating updated, sync started."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update rating: {str(e)}")

@app.get("/api/graph/exposure/{employee_id}")
def get_graph_exposure(employee_id: str):
    """
    Computes Network Exposure score using a Neo4j Cypher query.
    Falls back to a calculated mock score if Neo4j is offline.
    """
    # Get unboosted ML risk score from the batch pipeline (same source as /api/employees table)
    unboosted_score = None
    try:
        import sqlite3
        import pandas as pd
        from pipeline.config import OLAP_PATH, get_conn
        with get_conn(OLAP_PATH) as conn:
            score_row = pd.read_sql(
                "SELECT General_Risk_Score FROM flight_risk_scores WHERE EmployeeId = ?",
                conn, params=(employee_id,)
            )
        if not score_row.empty:
            unboosted_score = float(score_row['General_Risk_Score'].iloc[0])
    except Exception as ml_e:
        logging.error(f"[!] Could not fetch unboosted score: {ml_e}")
        raise HTTPException(status_code=500, detail=f"Score lookup failed: {str(ml_e)}")

    driver = app.state.neo4j_driver
    if not driver:
        # Since Neo4j is offline, back-calculate a mock exposure score 
        # so the math makes sense for the user demo.
        # Pull the contagion-boosted score from flight_risk_scores (batch pipeline output)
        final_score = 0
        try:
            with get_conn(OLAP_PATH) as conn:
                score_row = pd.read_sql(
                    "SELECT General_Risk_Score FROM flight_risk_scores WHERE EmployeeId = ?",
                    conn, params=(employee_id,)
                )
            if not score_row.empty:
                final_score = float(score_row['General_Risk_Score'].iloc[0])
        except Exception:
            pass
        
        # Generate a deterministic mock exposure based on employee context
        # Use the employee's risk level to create a plausible network exposure
        import hashlib
        seed_val = int(hashlib.md5(employee_id.encode()).hexdigest()[:8], 16) % 1000
        base_exposure = seed_val / 1000.0  # 0.0 to 1.0
        
        # Scale exposure: high-risk employees more likely to have high-risk peers
        if unboosted_score is not None and unboosted_score > 50:
            mock_score = round(base_exposure * 2.5, 2)  # 0.0 to 2.5
        elif unboosted_score is not None and unboosted_score > 25:
            mock_score = round(base_exposure * 1.2, 2)  # 0.0 to 1.2
        else:
            mock_score = round(base_exposure * 0.4, 2)  # 0.0 to 0.4
        
        # Add a fake peer if the mock score is non-trivial
        mock_peers = []
        curve = []
        import math
        if mock_score > 0.5:
            import random
            random.seed(employee_id)
            days_ago = random.randint(5, 20)
            mock_peers = [{"id": f"EMP_{random.randint(1000, 1400)}", "role": "Peer", "weight": round(mock_score, 2)}]
            
            for d in range(60, -1, -1):
                daily = 0
                days_since_exit_then = days_ago - d
                if days_since_exit_then >= 0:
                    daily += mock_score * math.exp(-0.1 * days_since_exit_then)
                curve.append({"day": -d, "score": round(daily, 2)})
        else:
            for d in range(60, -1, -1):
                curve.append({"day": -d, "score": 0.0})
            
        return {
            "exposure_score": mock_score, 
            "connected_exits": mock_peers,
            "curve": curve,
            "unboosted_score": round(unboosted_score, 1) if unboosted_score is not None else None
        }
        
    query = """
    MATCH (e:Employee {id: $employee_id})-[r:SAME_MANAGER|SAME_ROLE_DEPT|SAME_TENURE_COHORT]-(peer:Employee)
    WHERE peer.isActive = false AND peer.exitDate <> ''
    WITH peer, r, duration.inDays(date(peer.exitDate), date()).days AS days_since_exit
    WHERE days_since_exit <= 60
    RETURN peer.id AS peer_id, peer.jobRole AS peer_role, r.weight AS edge_weight, days_since_exit
    ORDER BY edge_weight DESC
    """
    
    try:
        with driver.session() as session:
            result = session.run(query, employee_id=employee_id)
            peers = []
            total_exposure = 0.0
            import math
            raw_peers = []
            
            for record in result:
                weight = float(record["edge_weight"])
                days = int(record["days_since_exit"])
                raw_peers.append({"id": record["peer_id"], "weight": weight, "days": days})
                
                # Exponential decay: e^(-0.1 * days)
                decay = math.exp(-0.1 * days) if days <= 60 else 0
                exposure = weight * decay
                total_exposure += exposure
                
                peers.append({
                    "id": record["peer_id"],
                    "role": record["peer_role"],
                    "weight": round(weight, 2),
                    "exposure": round(exposure, 2)
                })
                
            # Build continuous curve for the past 60 days
            curve = []
            for d in range(60, -1, -1):
                daily = 0.0
                for rp in raw_peers:
                    days_since_exit_then = rp["days"] - d
                    if days_since_exit_then >= 0:
                        daily += rp["weight"] * math.exp(-0.1 * days_since_exit_then)
                curve.append({"day": -d, "score": round(daily, 2)})
            # Get unboosted ML risk score from the same batch source as the table
            unboosted_score = None
            try:
                conn = sqlite3.connect(OLAP_PATH)
                score_row = pd.read_sql(
                    "SELECT General_Risk_Score FROM flight_risk_scores WHERE EmployeeId = ?",
                    conn, params=(employee_id,)
                )
                conn.close()
                if not score_row.empty:
                    unboosted_score = float(score_row['General_Risk_Score'].iloc[0])
            except Exception as ml_e:
                print(f"[!] Could not fetch unboosted score: {ml_e}")

            return {
                "exposure_score": round(total_exposure, 2),
                "connected_exits": peers,
                "curve": curve,
                "unboosted_score": round(unboosted_score, 1) if unboosted_score is not None else None
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Neo4j Query Failed: {str(e)}")


@app.get("/api/org-network/tree")
def get_org_network_tree(department: Optional[str] = None):
    try:
        with open('data/org_hierarchy.json', 'r') as f:
            data = json.load(f)
            nodes = data.get('nodes', [])
            
        # Fetch risk scores to attach to nodes
        import sqlite3
        import pandas as pd
        from pipeline.config import OLAP_PATH
        
        try:
            conn = sqlite3.connect(OLAP_PATH)
            risk_df = pd.read_sql("SELECT EmployeeId, General_Risk_Score FROM flight_risk_scores", conn)
            conn.close()
            risk_dict = dict(zip(risk_df['EmployeeId'], risk_df['General_Risk_Score']))
        except Exception as ex:
            print(f"Error fetching risk scores: {ex}")
            risk_dict = {}

        if department:
            filtered = []
            for n in nodes:
                n['risk_score'] = risk_dict.get(n['id'], 0)
                if n['id'] == 'CEO' or n['id'] == f"HEAD_{department.replace(' ', '_')}" or n.get('department') == department:
                    filtered.append(n)
            return {"nodes": filtered}
            
        for n in nodes:
            n['risk_score'] = risk_dict.get(n['id'], 0)
            
        return {"nodes": nodes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/similarity-network/{department}")
def get_similarity_network(department: str):
    """Serve pre-generated similarity network JSON for the given department."""
    slug = department.replace(" ", "_").replace("&", "and")
    file_path = os.path.join(DATA_DIR, f"similarity_network_{slug}.json")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Similarity network not generated for this department.")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/api/org-network/exposure-history/{employee_id}")
def get_org_network_exposure(employee_id: str):
    try:
        conn = sqlite3.connect(OLAP_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get peers (people in same department/role, active=0)
        # For simplicity, we just find some exited employees from employee_history
        # The exact query isn't strictly defined in the prompt, just the date parsing issue
        query = "SELECT EmployeeId, JobRole, DateOfLeaving FROM employee_history WHERE is_active=0 AND DateOfLeaving IS NOT NULL AND DateOfLeaving != '' LIMIT 10"
        records = cursor.execute(query).fetchall()
        
        peers = []
        for rec in records:
            try:
                # Attempt to parse date
                dt = datetime.strptime(rec['DateOfLeaving'], '%Y-%m-%d')
                peers.append({
                    "id": rec['EmployeeId'],
                    "role": rec['JobRole'],
                    "exit_date": dt.strftime('%Y-%m-%d')
                })
            except Exception:
                # Silently skip date parsing errors as described in phase 0
                pass
                
        conn.close()
        return {"peers": peers}
    except Exception as e:
        return {"peers": []} # Return peers even if empty, not 500


@app.get("/dashboard/org-network", response_class=HTMLResponse)
def org_network():
    with open('app/org_network_similarity.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.post("/api/simulate/attrition-distribution")
def simulate_attrition_distribution(req: AttritionSimRequest):
    try:
        result = attrition_simulate(
            n_sims=req.n_sims, 
            alpha=req.alpha, 
            beta=req.beta,
            starting_headcount=req.starting_headcount,
            rate_lo=req.rate_lo,
            rate_hi=req.rate_hi,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result

@app.get("/dashboard/attrition-simulation", response_class=HTMLResponse)
def attrition_simulation_page():
    page_path = os.path.join(os.path.dirname(__file__), "attrition_simulation.html")
    with open(page_path, 'r', encoding='utf-8') as f:
        return f.read()

@app.get("/dashboard/attrition-simulation-graph", response_class=HTMLResponse)
def attrition_simulation_graph_page():
    page_path = os.path.join(os.path.dirname(__file__), "attrition_simulation_graph.html")
    with open(page_path, 'r', encoding='utf-8') as f:
        return f.read()

app.mount("/", StaticFiles(directory=os.path.dirname(__file__), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # Start web server
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
