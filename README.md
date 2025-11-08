

------
# Twin Engine — Jet Engine Remaining Useful Life (RUL) Predictor

A modern **Digital Twin interface** for visualizing and interacting with jet engine health data using **React + TypeScript + FastAPI**.  
This project showcases how data-driven models can predict **Remaining Useful Life (RUL)** of aircraft engines based on sensor telemetry, enabling **predictive maintenance** and **failure prevention** in aerospace systems.

---

## Project Overview

The **Twin Engine** project is a **digital twin simulation** that uses machine learning models trained on NASA's **CMAPSS dataset** to estimate the **remaining operational life** of a turbofan engine.  

This frontend application provides a **clear and interactive user interface** to explore model metadata, understand the meaning of each input feature, and perform live predictions through your deployed **FastAPI backend**.

### Purpose

The primary goal is to **bridge machine learning outputs with real-world engineering insights**.  
This application demonstrates:
- How **engine telemetry data** (sensor readings and operational settings) can be converted into **prognostic features**
- How **data science models** provide interpretable RUL estimates and confidence intervals
- How a **digital twin interface** supports engineers in assessing equipment health and scheduling maintenance efficiently

---

## Key Features

| Feature | Description |
|----------|--------------|
| **Dynamic Metadata Loading** | Automatically fetches `/metadata` from the backend and configures UI dynamically |
| **Feature Explanation Panel** | Displays every model feature (`setting1`, `s1`, `s1_ma`, `s1_std`, `s1_diff`, `HI`) and explains its purpose |
| **Predictive Dashboard** | Allows live inference using either `/predict/window` or `/predict/features` |
| **Visualization and Results** | Displays RUL results with color-coded interpretation and uncertainty bounds |
| **Professional Design** | Clean glassmorphism-inspired interface suitable for research demos or portfolio showcases |
| **Adaptive Architecture** | Frontend-agnostic API integration — works with any FastAPI RUL model following the same schema |

---

## What the Outputs Mean

The model returns three key outputs, each with a distinct engineering interpretation:

| Output | Meaning | Engineering Insight |
|---------|----------|----------------------|
| **`rul`** | The predicted **Remaining Useful Life** (in cycles) before engine failure | Represents the estimated number of future operational cycles before maintenance or failure is expected |
| **`p10`** | Lower bound (10th percentile) | Conservative estimate — 90% confidence that actual life is *greater* than this value |
| **`p90`** | Upper bound (90th percentile) | Optimistic estimate — 90% confidence that actual life is *less* than this value |

Together, the interval `[p10, p90]` represents the **confidence range** for engine health prediction.  
A **narrower range** indicates high confidence in prediction, while a **wider range** suggests greater uncertainty.

---

## Model Inputs Explained

Each input feature corresponds to an operational or sensor variable from the CMAPSS dataset.

### Operational Settings
| Feature | Description |
|----------|-------------|
| `setting1`, `setting2`, `setting3` | Control parameters affecting the engine’s operating regime (altitude, throttle, pressure ratios) |

### Sensor Features
| Feature | Description |
|----------|-------------|
| `s1..s26` | Raw sensor measurements such as temperatures, pressures, vibration readings, or speed ratios |

### Derived Statistical Features
| Feature | Description |
|----------|-------------|
| `*_ma` | Moving average — smooths sensor trends over time |
| `*_std` | Standard deviation — measures sensor variability |
| `*_diff` | Difference — change from the previous reading, capturing acceleration/deceleration in degradation patterns |
| `HI` | Health Index — aggregated normalized indicator of overall engine health (higher = healthier) |

---

## Technology Stack

| Layer | Framework / Tool | Purpose |
|--------|------------------|----------|
| **Frontend** | React + TypeScript (Vite) | Interactive UI for model exploration |
| **Backend** | FastAPI (Python) | REST API serving RUL predictions and metadata |
| **Machine Learning** | Scikit-Learn + Gradient Boosting | Predictive model for RUL estimation |
| **Styling** | Tailored CSS (glassmorphic design) | Professional, minimal interface |
| **Packaging** | Vite Build Tool | High-performance development and production build pipeline |

---

## Project Structure


```

twin-engine-ui/  
├── src/  
│ ├── App.tsx # Main UI component  
│ ├── main.tsx # Entry point  
│ ├── index.css # Styling and theme  
│ └── components/ # Optional modular components  
├── public/  
├── .env # API configuration  
├── package.json  
└── README.md

```

---

## Installation Guide

### Prerequisites
- Node.js (v18 or later)
- FastAPI backend running (example below)

---

### 1. Clone the Repository
```bash
git clone https://github.com/yourname/twin-engine-ui.git
cd twin-engine-ui

```

### 2. Install Dependencies

```bash
npm install

```

### 3. Set Up Environment Variables

Create a `.env` file at the root:

```bash
VITE_API_BASE=http://localhost:8000

```

Modify the URL if your API is hosted remotely.

### 4. Run the Development Server

```bash
npm run dev

```

Open the printed URL (e.g., `http://localhost:5173`) in your browser.

### 5. Build for Production

```bash
npm run build
npm run preview

```

This generates an optimized production build in the `dist/` folder.

----------

## Backend Reference

Your backend should be a FastAPI application exposing the following routes:

Method

Endpoint

Description

`GET`

`/metadata`

Returns model metadata, feature names, and artifacts directory

`POST`

`/predict/window`

Accepts a recent telemetry window and returns RUL predictions

`POST`

`/predict/features`

Accepts precomputed feature maps for direct inference

`POST`

`/predict/batch`

Performs batch inference over multiple feature inputs

### Example Response

```json
{
  "rul": 143.86,
  "p10": 80.11,
  "p90": 125.00
}

```

----------

## Using the Frontend

### 1. Metadata View

Automatically loads model information, including:

-   Feature list (`feature_order`)
    
-   Key sensors used (`key_sensors`)
    
-   Artifact directory
    
-   Optional model card (training details and metrics)
    

### 2. Feature Explanation

Displays each feature and its derived versions (`_ma`, `_std`, `_diff`, `HI`) with plain-language definitions.

### 3. Prediction Console

Interactive forms for:

-   **Window-based prediction:** paste recent telemetry, the backend computes rolling stats.
    
-   **Feature-based prediction:** manually input all numeric feature values.
    

### 4. Output Interpretation

Shows predicted RUL, P10, and P90 values in a clear color-coded layout:

-   Green = current RUL estimate
    
-   Orange = lower confidence limit
    
-   Blue = upper confidence limit
    

----------

## Purpose and Impact

### Engineering Purpose

This project demonstrates how digital twins can:

-   Model complex aerospace systems using real sensor data
    
-   Predict equipment degradation and optimize maintenance scheduling
    
-   Visualize model transparency and interpretability through intuitive UIs
    

### Educational and Research Use

-   Ideal for **university research projects**, **data science portfolios**, and **predictive maintenance demos**
    
-   Suitable for showcasing at **engineering exhibitions** or **graduate recruitment assessments**
    

----------

## Keywords and SEO Tags

> Digital Twin, Jet Engine Prognostics, Remaining Useful Life, RUL Prediction, Predictive Maintenance, Aerospace Engineering, Machine Learning, FastAPI, React TypeScript, Data Visualization, CMAPSS Dataset, Health Index, Sensor Data, AI for Engineering, Condition Monitoring.

----------

## Example Interpretation

Scenario

Interpretation

`rul = 150, p10 = 130, p90 = 170`

Engine is healthy with high confidence (±20 cycles uncertainty).

`rul = 40, p10 = 15, p90 = 60`

Engine nearing failure; maintenance required soon.

`rul = 100, p10 = 40, p90 = 120`

Moderate uncertainty; suggests inconsistent sensor behavior or varying conditions.

----------

## License

This project is distributed under the **MIT License**.  
You are free to modify, use, and integrate it into academic, personal, or commercial projects.

----------

## Author

**Shreyas Gowda B**  
MSc Data Science, University of Glasgow  
LinkedIn: [Shreyas Gowda](https://www.linkedin.com/in/shreyas-gowda-5316b51b1/)

----------

## Citation

If you use this project or derivative work in your research or publication, please cite the reference dataset:

> A. Saxena, K. Goebel, D. Simon, and N. Eklund,  
> _Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation_,  
> Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver, CO, 2008.

----------

## Future Work

-   Add line charts showing real-time RUL trajectory updates
    
-   Integrate real CMAPSS telemetry playback for streaming simulation
    
-   Include conformal prediction intervals for calibrated uncertainty
    
-   Deploy via Docker for easy backend–frontend integration
    

----------

## Summary

This project combines **engineering insight**, **data science**, and **UI design** to deliver an interactive **digital twin dashboard** for jet engine health monitoring.  
It serves as a strong portfolio demonstration of your ability to build **end-to-end AI-driven systems** — from predictive modeling to visualization and deployment.



---

