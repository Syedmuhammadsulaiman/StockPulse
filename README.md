# Stock Pulse - Stock Price Anomaly Detection System

## 📌 Overview
Stock Pulse is a stock price anomaly detection system built with **Flask**.  
It allows users to:
- Upload stock datasets (CSV format)
- Detect anomalies using machine learning algorithms
- Visualize results with charts and tables
- Export reports in **Excel** and **PDF** formats

---

## ⚙️ Setup Instructions

### 1. Create a Virtual Environment
```bash
python -m venv venv
```

### 2. Activate the Virtual Environment
```bash
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install flask flask-pymongo flask-wtf flask-bcrypt flask-login python-dotenv numpy matplotlib plotly scikit-learn email-validator reportlab
```

### 🔑 Environment Variables
```bash
SECRET_KEY=your-secret-key
MONGO_URI=your-mongodb-uri
```

### 🚀 Run the Application
```bash
python run.py
```