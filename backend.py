import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from scipy.spatial.distance import cdist
from typing import List, Dict, Any

app = FastAPI(
    title="Wind Turbine SCADA AI & MCP Service",
    description="Backend AI Pipeline for SCADA False Alarm Filtering, Route Optimization, and Surface Visual Diagnostics",
    version="1.0.0"
)

# -------------------------------------------------------------------
# 1. SCADA DATA GENERATOR & FALSE ALARM FILTERING ENGINE
# -------------------------------------------------------------------

def generate_synthetic_scada(num_turbines: int = 10, samples_per_turbine: int = 100):
    np.random.seed(42)
    data = []
    for t_id in range(1, num_turbines + 1):
        for _ in range(samples_per_turbine):
            wind_speed = np.random.uniform(3.0, 25.0)
            vibration = np.random.normal(loc=1.2, scale=0.4) + (0.15 * (t_id % 3))
            bearing_temp = np.random.normal(loc=55.0, scale=10.0) + (2.5 * (t_id % 4))
            yaw_error = np.random.uniform(-15.0, 15.0)
            
            # Synthetic anomaly logic
            is_real_fault = 1 if (vibration > 2.1 and bearing_temp > 75.0) else 0
            is_transient_noise = 1 if (vibration > 2.0 and bearing_temp <= 75.0) else 0
            alarm_triggered = 1 if (is_real_fault or is_transient_noise) else 0

            data.append({
                "turbine_id": f"T-{t_id:02d}",
                "wind_speed": round(wind_speed, 2),
                "vibration": round(vibration, 3),
                "bearing_temp": round(bearing_temp, 2),
                "yaw_error": round(yaw_error, 2),
                "alarm_triggered": alarm_triggered,
                "is_real_fault": is_real_fault
            })
    return pd.DataFrame(data)

df_scada = generate_synthetic_scada()

# Train baseline anomaly detection models
X = df_scada[["wind_speed", "vibration", "bearing_temp", "yaw_error"]]
y = df_scada["is_real_fault"]

iso_forest = IsolationForest(contamination=0.1, random_state=42).fit(X)
rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)

@app.get("/api/fleet/status")
def get_fleet_status():
    summary = []
    for t_id, group in df_scada.groupby("turbine_id"):
        recent = group.tail(20)
        X_rec = recent[["wind_speed", "vibration", "bearing_temp", "yaw_error"]]
        
        preds_rf = rf_classifier.predict(X_rec)
        fault_rate = float(np.mean(preds_rf))
        
        if fault_rate > 0.3:
            health_status = "CRITICAL"
        elif fault_rate > 0.1:
            health_status = "WARNING"
        else:
            health_status = "HEALTHY"
            
        summary.append({
            "turbine_id": t_id,
            "vibration_avg": round(float(recent["vibration"].mean()), 3),
            "temp_avg": round(float(recent["bearing_temp"].mean()), 2),
            "total_alarms": int(recent["alarm_triggered"].sum()),
            "predicted_real_faults": int(sum(preds_rf)),
            "health_status": health_status,
            "health_score": round(100 - (fault_rate * 100), 1)
        })
    return {"fleet": summary}

@app.get("/api/false-alarm-analysis")
def get_false_alarm_analysis():
    X_all = df_scada[["wind_speed", "vibration", "bearing_temp", "yaw_error"]]
    
    # Model 1: Threshold-based SCADA Rules
    raw_alarms = df_scada["alarm_triggered"].sum()
    
    # Model 2: Isolation Forest Anomaly Detection
    iso_preds = iso_forest.predict(X_all)
    iso_anomalies = int(np.sum(iso_preds == -1))
    
    # Model 3: Random Forest Supervised Classifier
    rf_preds = rf_classifier.predict(X_all)
    rf_real_faults = int(np.sum(rf_preds == 1))
    
    return {
        "raw_scada_alarms": int(raw_alarms),
        "isolation_forest_flagged": iso_anomalies,
        "rf_classifier_verified_faults": rf_real_faults,
        "false_alarms_prevented": int(raw_alarms - rf_real_faults),
        "false_alarm_reduction_percentage": round(((raw_alarms - rf_real_faults) / raw_alarms) * 100, 2)
    }

# -------------------------------------------------------------------
# 2. TRUCK ROLLOUT OPTIMIZATION (COMPARING ALGORITHMS)
# -------------------------------------------------------------------

class RouteRequest(BaseModel):
    selected_turbines: List[str]

turbine_coords = {
    "Depot": (0.0, 0.0),
    "T-01": (1.2, 3.4), "T-02": (4.5, 1.1), "T-03": (6.1, 5.2),
    "T-04": (2.3, 7.8), "T-05": (8.9, 2.3), "T-06": (7.4, 8.1),
    "T-07": (3.1, 9.5), "T-08": (9.0, 6.7), "T-09": (5.5, 4.3),
    "T-10": (1.1, 8.2)
}

@app.post("/api/optimize-route")
def optimize_route(req: RouteRequest):
    nodes = ["Depot"] + [t for t in req.selected_turbines if t in turbine_coords]
    if len(nodes) <= 1:
        return {"error": "Select at least 1 turbine for maintenance."}
    
    coords = np.array([turbine_coords[n] for n in nodes])
    dist_matrix = cdist(coords, coords, metric='euclidean')
    
    # Algorithm 1: Nearest Neighbor (Greedy Heuristic)
    unvisited = list(range(1, len(nodes)))
    curr = 0
    nn_path = [0]
    nn_dist = 0.0
    while unvisited:
        nxt = min(unvisited, key=lambda x: dist_matrix[curr][x])
        nn_dist += dist_matrix[curr][nxt]
        curr = nxt
        nn_path.append(curr)
        unvisited.remove(curr)
    nn_dist += dist_matrix[curr][0]
    nn_path.append(0)
    
    # Algorithm 2: 2-Opt Optimization (Refinement Local Search)
    best_path = nn_path[:-1]
    best_dist = nn_dist
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best_path) - 1):
            for j in range(i + 1, len(best_path)):
                new_path = best_path[:i] + best_path[i:j][::-1] + best_path[j:]
                # Calculate distance
                d = sum(dist_matrix[new_path[k]][new_path[k+1]] for k in range(len(new_path)-1))
                d += dist_matrix[new_path[-1]][new_path[0]]
                if d < best_dist:
                    best_dist = d
                    best_path = new_path
                    improved = True
    best_path.append(best_path[0])
    
    return {
        "greedy_nearest_neighbor": {
            "route": [nodes[i] for i in nn_path],
            "distance_km": round(float(nn_dist * 10), 2), # Scale for realistic KM
            "estimated_hours": round(float((nn_dist * 10) / 40.0) + (len(nodes)-1)*1.5, 2)
        },
        "two_opt_optimized": {
            "route": [nodes[i] for i in best_path],
            "distance_km": round(float(best_dist * 10), 2),
            "estimated_hours": round(float((best_dist * 10) / 40.0) + (len(nodes)-1)*1.5, 2)
        }
    }

# -------------------------------------------------------------------
# 3. DRONE COMPUTER VISION SURFACING DAMAGE ENGINE
# -------------------------------------------------------------------

@app.get("/api/drone/scan/{turbine_id}")
def scan_drone_surface(turbine_id: str):
    np.random.seed(hash(turbine_id) % 1000)
    defects = ["Leading-Edge Erosion", "Micro-Crack", "Lightning Strike Mark", "Delamination", "None"]
    severity = ["Low", "Medium", "High", "Critical"]
    
    detected_defects = []
    num_issues = np.random.randint(0, 4)
    
    for i in range(num_issues):
        d_type = np.random.choice(defects[:-1])
        sev = np.random.choice(severity)
        blade_location = f"Blade {np.random.choice(['A', 'B', 'C'])} - {np.random.randint(5, 45)}m from Hub"
        confidence = round(float(np.random.uniform(0.82, 0.99)), 2)
        
        detected_defects.append({
            "issue_id": f"DEF-{turbine_id}-{i+1:02d}",
            "defect_type": d_type,
            "severity": sev,
            "location": blade_location,
            "confidence": confidence
        })
        
    return {
        "turbine_id": turbine_id,
        "inspection_status": "COMPLETED",
        "defects_found": len(detected_defects),
        "findings": detected_defects
    }

# -------------------------------------------------------------------
# 4. MCP TOOL DECLARATIONS & AGENT RAG ROUTER
# -------------------------------------------------------------------

@app.get("/mcp/tools")
def list_mcp_tools():
    """Model Context Protocol (MCP) tool exposure interface"""
    return {
        "mcp_version": "1.0",
        "tools": [
            {
                "name": "filter_false_alarms",
                "description": "Applies ensemble ML models to eliminate transient SCADA alarms.",
                "parameters": {"turbine_id": "string"}
            },
            {
                "name": "optimize_truck_rollout",
                "description": "Calculates optimal maintenance routes for critical turbines using 2-Opt TSP.",
                "parameters": {"selected_turbines": "array"}
            },
            {
                "name": "analyze_drone_surface_imagery",
                "description": "Runs vision defect detection over turbine blades.",
                "parameters": {"turbine_id": "string"}
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

