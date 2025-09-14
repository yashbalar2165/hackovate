<video width="600" controls>
  <source src="D:/hackovate/AirGuard Weather System .mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
# AI/ML Weather Prediction & Alert System for Airfields

## Overview
This project aims to develop an AI/ML-based system to predict and alert airfields about thunderstorms and gale force winds in real-time. Severe weather events can disrupt airfield operations, affect flight safety, and pose risks to ground staff. Traditional forecasting often lacks the precision for localized, short-duration events, making AI/ML an effective solution.

## Features
- *Real-time Data Collection:* Integrates radar, satellite imagery, weather stations, and historical records.
- *Predictive Models:*
  - *Nowcasting (0–3 hrs):* LSTM for wind gusts, CNN for radar/satellite patterns, ensemble models (Random Forest/GBM) for storm likelihood.
  - *Medium-term (up to 24 hrs):* Synoptic-scale forecasting with continuous retraining.
- *Alerts & Explainability:* Generates probability-based alerts with explanations for decisions.
- *Dashboard:* Interactive maps, storm movement animations, color-coded risk levels, custom thresholds per airfield, and alert notifications.

## Example Scenarios
- *Thunderstorm Warning:* 80% likelihood in 2 hrs → Orange alert → Delayed landings.
- *Severe Winds:* 65 km/h in 30 min → Red alert → Ground crew secures equipment.

## Goal
Enable timely, accurate, and actionable weather insights to ensure safe and efficient airfield operations.

# AirGuard Weather Dashboard

A modern, full-stack aviation weather dashboard using **FastAPI** (backend) and **Next.js + Tailwind CSS** (frontend).  
It provides live weather data, ML-based weather code prediction, and interactive charts for airports.

---

## Project Structure

```
fastapi-streamlit-app/
├── backend/
│   └── app/
│       ├── main.py                # FastAPI app entrypoint
│       ├── api/
│       │   └── routes.py          # API endpoints (weather, prediction, alerts)
│       ├── utils/
|       |   └── forcast_of_24.py   # ML model: WeatherAlertPredictor (RandomForest) for next 24 hour prediction
│       │   └── prediction.py      # ML model: ImprovedWeatherCodePredictor (XGBoost) for next 24 hour prediction
│       └── requirements.txt       # Backend dependencies
├── frontend/
│   └── my-app/
│       ├── app/
│       │   └── globals.css              # Tailwind & custom styles
│       │   └── page.tsx                 # Home page and header
│       ├── components/
│       │   ├── weather-charts.tsx       # Weather charts (live data)
│       │   ├── weather-map-layers.tsx   # Map layers (live data)
│       │   ├── weather-map.tsx          # Map (live data)
│       │   ├── weather-sidebar.tsx      # Sidebar handler (live data)
│       │   └── ui/
│       │       ├── card.tsx             # Card UI component
│       │       ├── button.tsx           # Button UI component
│       │       ├── modal.tsx            # Modal UI component
│       │       ├── input.tsx            # Input UI component
│       │       ├── select.tsx           # Select UI component
│       │       └── ...                  # Other UI components
│       ├── pages/
│       │   └── index.tsx                # Main dashboard page
│       ├── public/
│       │   └── favicon.ico              # Site favicon
│       ├── package.json
│       ├── postcss.config.js
│       ├── tailwind.config.js
│       ├── tsconfig.json
│       └── README.md                    # Frontend-specific documentation (optional)
│   └── requirements.txt                 # Frontend dependencies (if using Streamlit)
├── README.md                            # Project documentation
└── .gitignore
```

---

## Features

- **Live Weather Data:**  
  Uses [Open-Meteo API](https://open-meteo.com/) for real-time weather (no API key required).

- **ML Weather Prediction:**  
  Predicts `next3h_max_weathercode` using a robust XGBoost model trained on historical data.

- **Interactive Dashboard:**  
  - Weather charts (temperature, humidity, wind, pressure, precipitation, thunderstorm risk, visibility)
  - Map layers for airport weather visualization
  - Aviation-focused color palette and dark mode

- **Backend API:**  
  - `/forecast`: Returns live weather and ML prediction for given latitude/longitude
  - `/thunderstrome`: (custom) Aviation weather alerts
  - `/model-status`: Model health/status endpoint

- **Frontend:**  
  - Built with Next.js, React, Tailwind CSS
  - Uses live weather data for charts and layers
  - Connects to backend for ML predictions and alerts

- **Streamlit Integration:**  
  - Optionally, use Streamlit for rapid prototyping or additional dashboards

---

## Technologies Used

- **Backend:**  
  - FastAPI
  - Open-Meteo API (live weather)
  - XGBoost, scikit-learn (ML model)
  - Pandas, NumPy
  - Joblib (model persistence)
  - Requests, requests-cache, retry_requests

- **Frontend:**  
  - Next.js (React)
  - Tailwind CSS (custom aviation theme)
  - Chart.js or similar (for charts)
  - Streamlit (optional)

---

## Setup Instructions

### 1. Backend (FastAPI)

```bash
cd backend/app
python -m venv venv
venv\Scripts\activate        # On Windows
pip install -r requirements.txt
uvicorn main:app --reload    # Run FastAPI server
```

- **Model Training:**  
  Place your CSV data in `backend/app/` and run:
  ```bash
  python utils/prediction.py
  ```
  This will train and save `improved_weather_predictor.joblib`.

### 2. Frontend (Next.js)

```bash
cd frontend/my-app
npm install
npm run dev                  # Start frontend server
```

- **Tailwind/PostCSS:**  
  Make sure you have `@tailwindcss/postcss` installed and configured in `postcss.config.js`.

### 3. Streamlit (Optional)

```bash
cd frontend
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## Usage

- **Access the dashboard:**  
  - Frontend: [http://localhost:3000](http://localhost:3000)
  - Backend API: [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI)

- **Live Weather:**  
  - Charts and map layers use Open-Meteo for live data.
  - ML predictions and alerts use FastAPI backend.

---

## Customization

- **Add new airports:**  
  Update airport list in frontend.
- **Change color theme:**  
  Edit `globals.css` for aviation palette.
- **Improve ML model:**  
  Update `prediction.py` and retrain with new data.

---

## Credits

- [Open-Meteo](https://open-meteo.com/) for free weather API
- [FastAPI](https://fastapi.tiangolo.com/)
- [Next.js](https://nextjs.org/)
- [Tailwind CSS](https://tailwindcss.com/)
- [Streamlit](https://streamlit.io/)
- [XGBoost](https://xgboost.readthedocs.io/)

---

## License

MIT License

---

## Contact

For questions or contributions, open an issue or pull request on GitHub.
