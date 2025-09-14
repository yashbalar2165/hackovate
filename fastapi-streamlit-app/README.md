# FastAPI and Streamlit Application

This project is a web application that utilizes FastAPI as the backend and Streamlit as the frontend. The backend handles API requests and data processing, while the frontend provides an interactive user interface.

## Project Structure

```
fastapi-streamlit-app
├── backend
│   ├── app
│   │   ├── main.py          # Entry point for the FastAPI application
│   │   ├── api
│   │   │   └── routes.py    # API routes for handling requests
│   │   ├── models
│   │   │   └── schemas.py    # Pydantic models for data validation
│   │   └── utils
│   │       └── helpers.py    # Utility functions for the backend
│   └── requirements.txt      # Backend dependencies
├── frontend
│   ├── streamlit_app.py      # Main entry point for the Streamlit application
│   └── requirements.txt       # Frontend dependencies
└── README.md                  # Project documentation
```

## Setup Instructions

### Backend Setup

1. Navigate to the `backend` directory:
   ```
   cd backend
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the FastAPI application:
   ```
   uvicorn app.main:app --reload
   ```

### Frontend Setup

1. Navigate to the `frontend` directory:
   ```
   cd frontend
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```
   streamlit run streamlit_app.py
   ```

## Usage

- Access the FastAPI backend at `http://127.0.0.1:8000`.
- Access the Streamlit frontend at `http://localhost:8501`.

## Contributing

Feel free to submit issues or pull requests for improvements or bug fixes.