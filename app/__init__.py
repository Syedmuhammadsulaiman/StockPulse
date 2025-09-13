from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from dotenv import load_dotenv
from functools import wraps
import os
import pickle
from flask_bcrypt import Bcrypt
from app.forms import RegisterForm
import pandas as pd
import numpy as np
import plotly.express as px
import json
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib
matplotlib.use("Agg")  # use non-GUI backend
import matplotlib.pyplot as plt

# ------------------ LOAD ENV ------------------
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
MONGO_URI = os.getenv("MONGO_URI")

# ------------------ CONFIG ------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = SECRET_KEY

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# MongoDB setup
bcrypt = Bcrypt(app)
client = MongoClient(MONGO_URI)
db = client.get_default_database()
users_collection = db["users"]
datasets_collection = db["datasets"]

ALLOWED_EXTENSIONS = {"csv", "xlsx"}

# ------------------ LOAD MODELS ------------------
BASE_MODELS_DIR = os.path.abspath(os.path.join(app.root_path, "..", "models"))
RIDGE_MODELS_DIR = os.path.join(BASE_MODELS_DIR, "ridge_models")
FEATURES_DIR = os.path.join(BASE_MODELS_DIR, "features")
DF_FEAT_DIR = os.path.join(BASE_MODELS_DIR, "df_feat")

ridge_models = {}
features_dict = {}
df_feat_dict = {}

def load_pickles(dir_path, container_dict):
    if os.path.exists(dir_path):
        for fname in os.listdir(dir_path):
            if fname.endswith(".pkl"):
                ticker = fname.split("_")[0].upper()
                with open(os.path.join(dir_path, fname), "rb") as f:
                    container_dict[ticker] = pickle.load(f)
    else:
        os.makedirs(dir_path)

load_pickles(RIDGE_MODELS_DIR, ridge_models)
load_pickles(FEATURES_DIR, features_dict)
load_pickles(DF_FEAT_DIR, df_feat_dict)

print(f"✅ Loaded Ridge models: {list(ridge_models.keys())}")
print(f"✅ Loaded Features: {list(features_dict.keys())}")
print(f"✅ Loaded df_feat: {list(df_feat_dict.keys())}")

# ------------------ HELPERS ------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "email" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

def redirect_if_logged_in():
    if "email" in session:
        return redirect(url_for("analysis"))
    return None

# ------------------ GLOBAL VARIABLE FOR EXPORT ------------------
last_results_df = None

# ------------------ ROUTES ------------------

@app.route("/")
def welcome():
    redirect_check = redirect_if_logged_in()
    if redirect_check: 
        return redirect_check
    return render_template("welcome.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    redirect_check = redirect_if_logged_in()
    if redirect_check: 
        return redirect_check

    form = RegisterForm()
    if request.method == "POST":
        if form.validate_on_submit():
            username = form.username.data.strip()
            email = form.email.data.strip()
            password = form.password.data.strip()

            if users_collection.find_one({"email": email}):
                return jsonify({"status": "error", "message": "Email already registered."}), 400

            hashed_pw = bcrypt.generate_password_hash(password).decode("utf-8")

            users_collection.insert_one({
                "username": username,
                "email": email,
                "password": hashed_pw
            })

            return jsonify({"status": "success", "message": "Registration successful!"}), 201
        else:
            errors = [f"{field}: {', '.join(msgs)}" for field, msgs in form.errors.items()]
            return jsonify({"status": "error", "message": " ".join(errors)}), 400

    return render_template("register.html", form=form)

@app.route("/login", methods=["GET", "POST"])
def login():
    redirect_check = redirect_if_logged_in()
    if redirect_check: 
        return redirect_check

    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        user = users_collection.find_one({"email": email})

        if user and bcrypt.check_password_hash(user["password"], password):
            session["email"] = email
            session["username"] = user["username"]
            return jsonify({"status": "success", "message": "Login successful!"}), 200
        else:
            return jsonify({"status": "error", "message": "Invalid email or password."}), 401

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("email", None)
    session.pop("username", None)
    return redirect(url_for("login"))

@app.route("/analysis", methods=["GET", "POST"])
@login_required
def analysis():
    global last_results_df

    results = None
    chartJSON = None
    uploaded_filename = None
    chartDict = {}

    if request.method == "POST":
        if "dataset" not in request.files:
            results = "<p class='text-red-600'>No file uploaded.</p>"
            return render_template("analysis.html", results=results)

        file = request.files["dataset"]
        if file.filename == "":
            results = "<p class='text-red-600'>No file selected.</p>"
            return render_template("analysis.html", results=results)

        # Save uploaded CSV
        filename = secure_filename(file.filename)
        uploaded_filename = filename
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Extract ticker
        ticker = filename.split("_")[0].upper()

        # Load df_feat if exists
        df_feat_path = os.path.join(DF_FEAT_DIR, f"{ticker}_df_feat.pkl")
        if not os.path.exists(df_feat_path):
            results = f"<p class='text-red-600'>No df_feat pickle found for {ticker}.</p>"
            return render_template("analysis.html", results=results)

        with open(df_feat_path, "rb") as f:
            df_feat = pickle.load(f)

        # Flatten MultiIndex if needed
        if isinstance(df_feat.columns, pd.MultiIndex):
            df_feat.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df_feat.columns]

        df_feat.columns = [col.rstrip('_') for col in df_feat.columns]

        if 'Date' not in df_feat.columns:
            date_col = [c for c in df_feat.columns if 'Date' in c][0]
            df_feat.rename(columns={date_col: 'Date'}, inplace=True)

        df_feat = df_feat.reset_index(drop=True)

        # Load uploaded CSV
        df = pd.read_csv(filepath)
        if 'Date' not in df.columns:
            df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df.sort_values("Date", inplace=True)

        # Merge with predictions
        merge_cols = ["Date", "Predicted", "Residual", "Anomaly"]
        merge_cols = [c for c in merge_cols if c in df_feat.columns]
        df_merged = pd.merge(df, df_feat[merge_cols], on="Date", how="left")

        # Save global df for export
        last_results_df = df_merged.copy()

        # Charts
        chart_dict = {}
        x_col = "Date"

        fig_close = px.line(df_merged, x=x_col, y="Close", title=f"{ticker} Closing Price Trend")
        fig_close.update_traces(line=dict(color="green"))
        fig_close.update_layout(template="plotly_white")
        chart_dict["Closing Price Trend"] = json.loads(fig_close.to_json())

        if "Predicted" in df_merged.columns:
            fig_pred = px.line(
                df_merged,
                x=x_col,
                y=["Close", "Predicted"],
                title=f"{ticker} Predicted vs Actual",
                color_discrete_map={"Close": "green", "Predicted": "red"}
            )
            fig_pred.update_layout(template="plotly_white")
            chart_dict["Predicted vs Actual"] = json.loads(fig_pred.to_json())

        if "Residual" in df_merged.columns:
            fig_res = px.line(df_merged, x=x_col, y="Residual", title=f"{ticker} Residuals")
            fig_res.update_layout(template="plotly_white")
            chart_dict["Residuals"] = json.loads(fig_res.to_json())

        if "Anomaly" in df_merged.columns:
            fig_anom = px.scatter(df_merged, x=x_col, y="Close", color="Anomaly",
                                  title=f"{ticker} Anomalies", color_discrete_map={1:'red', 0:'blue'})
            fig_anom.update_layout(template="plotly_white")
            chart_dict["Anomalies"] = json.loads(fig_anom.to_json())

        if "Volume" in df_merged.columns:
            fig_bar = px.bar(df_merged.tail(30), x="Date", y="Volume",
                             title=f"{ticker} Volume (Last 30 Days)", color_discrete_sequence=["#16a34a"])
            fig_bar.update_layout(template="plotly_white")
            chart_dict["Volume Bar Chart"] = json.loads(fig_bar.to_json())

        if "Residual" in df_merged.columns:
            abs_res = df_merged["Residual"].abs().dropna()
            if len(abs_res) > 0:
                res_std = abs_res.std()
                if res_std == 0 or np.isnan(res_std):
                    q1 = abs_res.quantile(0.33)
                    q2 = abs_res.quantile(0.66)
                    bins = [0.0, q1, q2, abs_res.max() + 1e-9]
                else:
                    bins = [0.0, 0.5 * res_std, 1.5 * res_std, abs_res.max() + 1e-9]
                labels = ["Small error", "Medium error", "Large error"]
                res_cat = pd.cut(abs_res, bins=bins, labels=labels, include_lowest=True)
                res_counts = res_cat.value_counts().reset_index()
                res_counts.columns = ["Residual Magnitude", "Count"]
                fig_pie_res = px.pie(res_counts, names="Residual Magnitude", values="Count",
                                     title=f"{ticker} Residual Magnitude Distribution")
                fig_pie_res.update_layout(template="plotly_white")
                chart_dict["Residual Magnitude Pie"] = json.loads(fig_pie_res.to_json())

        chartJSON = json.dumps(chart_dict)
        chartDict = chart_dict

        results = df_merged.tail(50).to_html(classes="table-auto border border-gray-300 text-sm", index=False)

    return render_template("index.html", results=results, chartJSON=chartJSON,
                           chartDict=chartDict, uploaded_filename=uploaded_filename)

@app.route("/export_report", methods=["POST"])
def export_report():
    global last_results_df
    if last_results_df is None:
        return "No results to export", 400

    output = io.StringIO()
    last_results_df.to_csv(output, index=False)
    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="analysis_report.csv"
    )

@app.route("/export_pdf", methods=["POST"])
@login_required
def export_pdf():
    global last_results_df
    if last_results_df is None or last_results_df.empty:
        return "<p class='text-red-600'>No analysis available to export.</p>"

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # ---------------- PAGE 1: TITLE + SUMMARY ----------------
    elements.append(Paragraph("<b>📊 Stock Analysis Report</b>", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Generated with Stock Pulse", styles["Normal"]))
    elements.append(Spacer(1, 24))

    # Table of contents
    toc = """
    <b>Table of Contents:</b><br/>
    Page 1 - Report Overview<br/>
    Page 2 - Data Preview<br/>
    Page 3 - Closing Price Trend<br/>
    Page 4 - Predicted vs Actual<br/>
    Page 5 - Insights & Conclusion<br/>
    """
    elements.append(Paragraph(toc, styles["BodyText"]))
    elements.append(PageBreak())

    # ---------------- PAGE 2: DATA PREVIEW ----------------
    summary = f"""
    <b>Summary Statistics:</b><br/>
    • Total Records: {len(last_results_df)}<br/>
    • Date Range: {last_results_df['Date'].min()} → {last_results_df['Date'].max()}<br/>
    • Average Close Price: {last_results_df['Close'].mean():.2f}<br/>
    """
    elements.append(Paragraph(summary, styles["BodyText"]))
    elements.append(Spacer(1, 24))

    # Table preview
    preview_df = last_results_df.head(15).fillna("").astype(str)
    data = [preview_df.columns[:6].tolist()] + preview_df.values[:, :6].tolist()
    table = Table(data, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightblue),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold")
    ]))
    elements.append(Paragraph("<b>Data Preview</b>", styles["Heading2"]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(
        "The preview above shows the first 15 records of the dataset, including Open, High, Low, Close, and Volume values.",
        styles["BodyText"]
    ))
    elements.append(PageBreak())

    # ---------------- PAGE 3: CLOSING PRICE TREND ----------------
    fig, ax = plt.subplots(figsize=(6, 3))
    last_results_df.plot(x="Date", y="Close", ax=ax, legend=False, title="Closing Price Trend")
    ax.set_ylabel("Price ($)")
    chart_path = "closing_price.png"
    plt.savefig(chart_path, bbox_inches="tight")
    plt.close(fig)

    elements.append(Paragraph("<b>Closing Price Trend</b>", styles["Heading2"]))
    elements.append(Image(chart_path, width=400, height=200))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(
        "The graph above shows the trend of closing prices over time. "
        "It helps identify bullish (upward) or bearish (downward) patterns in the market.",
        styles["BodyText"]
    ))
    elements.append(PageBreak())

    # ---------------- PAGE 4: PREDICTED VS ACTUAL ----------------
    if "Predicted" in last_results_df.columns:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(last_results_df["Date"], last_results_df["Close"], label="Actual", color="blue")
        ax.plot(last_results_df["Date"], last_results_df["Predicted"], label="Predicted", color="orange")
        if "Anomaly" in last_results_df.columns:
            anomalies = last_results_df[last_results_df["Anomaly"] == True]
            ax.scatter(anomalies["Date"], anomalies["Close"], color="red", marker="x", label="Anomaly")
        ax.set_title("Predicted vs Actual Prices")
        ax.set_ylabel("Price ($)")
        ax.legend()
        pred_chart_path = "pred_vs_actual.png"
        plt.savefig(pred_chart_path, bbox_inches="tight")
        plt.close(fig)

        elements.append(Paragraph("<b>Predicted vs Actual</b>", styles["Heading2"]))
        elements.append(Image(pred_chart_path, width=400, height=200))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(
            "This graph compares the model’s predicted stock prices with the actual market values. "
            "Red markers indicate anomalies where residual errors exceeded the normal range.",
            styles["BodyText"]
        ))
    else:
        elements.append(Paragraph("<b>No prediction data available to display.</b>", styles["BodyText"]))
    elements.append(PageBreak())

    # ---------------- PAGE 5: INSIGHTS & CONCLUSION ----------------
    insights = """
    <b>Key Insights:</b><br/>
    • The closing price trend suggests periods of strong volatility.<br/>
    • Predicted vs Actual comparison shows the model is generally accurate, but anomalies highlight unusual market activity.<br/>
    • Detected anomalies can be signals for potential trading opportunities or risk management.<br/><br/>
    
    <b>Conclusion:</b><br/>
    This stock analysis report combines descriptive statistics, trend visualization, and predictive modeling
    to provide a comprehensive market overview. The approach can be extended to additional tickers for portfolio-level insights.
    """
    elements.append(Paragraph(insights, styles["BodyText"]))

    # ---------------- BUILD PDF ----------------
    doc.build(elements)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="stock_analysis_report.pdf",
        mimetype="application/pdf"
    )

# ------------------ MAIN ------------------
if __name__ == "__main__":
    app.run(debug=True)