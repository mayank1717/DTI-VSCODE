import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="NanoCatalyst AI", layout="wide")

# -------------------------
# TITLE
# -------------------------
st.title("🔬 NanoCatalyst AI Platform")
st.markdown("### XRD-Based Catalytic Activity Prediction System")

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    return pd.read_csv("nanoparticle_catalytic_dataset_5000.csv")

data = load_data()

# -------------------------
# PREPROCESS
# -------------------------
le = LabelEncoder()
data["Nanoparticle"] = le.fit_transform(data["Nanoparticle"])

X = data.drop("Activity", axis=1)
y = data["Activity"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# TRAIN MODELS
# -------------------------
@st.cache_resource
def train_models():
    rf = RandomForestRegressor(n_estimators=200)
    rf.fit(X_train, y_train)

    xgb = XGBRegressor(n_estimators=200, learning_rate=0.1)
    xgb.fit(X_train, y_train)

    return rf, xgb

rf_model, xgb_model = train_models()

# -------------------------
# MODEL PERFORMANCE
# -------------------------
rf_score = r2_score(y_test, rf_model.predict(X_test))
xgb_score = r2_score(y_test, xgb_model.predict(X_test))

col1, col2 = st.columns(2)

col1.metric("🌲 Random Forest R²", f"{rf_score:.3f}")
col2.metric("⚡ XGBoost R²", f"{xgb_score:.3f}")

# -------------------------
# SIDEBAR INPUT
# -------------------------
st.sidebar.header("⚙️ Input Parameters")

nanoparticle = st.sidebar.selectbox("Nanoparticle", le.classes_)
peak = st.sidebar.slider("Peak 2θ", 10.0, 80.0, 50.0)
fwhm = st.sidebar.slider("FWHM", 0.1, 1.0, 0.5)
size = st.sidebar.slider("Size (nm)", 1, 100, 30)
surface = st.sidebar.slider("Surface Area", 10, 200, 70)

nano_encoded = le.transform([nanoparticle])[0]

input_df = pd.DataFrame([{
    "Nanoparticle": nano_encoded,
    "Peak_2theta": peak,
    "FWHM": fwhm,
    "Size": size,
    "Surface Area": surface
}])

# -------------------------
# PREDICTIONS
# -------------------------
rf_pred = rf_model.predict(input_df)[0]
xgb_pred = xgb_model.predict(input_df)[0]

st.subheader("🧠 Model Predictions")

c1, c2 = st.columns(2)
c1.success(f"Random Forest: {rf_pred:.2f}")
c2.info(f"XGBoost: {xgb_pred:.2f}")

final_pred = (rf_pred + xgb_pred) / 2

st.subheader("🎯 Final AI Prediction")
st.success(f"{final_pred:.2f}")

# -------------------------
# AI RECOMMENDATION
# -------------------------
st.subheader("🤖 AI Recommendation Engine")

if final_pred > 80:
    st.success("🔥 High Performance Catalyst")
elif final_pred > 50:
    st.warning("⚡ Moderate Catalyst")
else:
    st.error("❗ Low Activity Catalyst")

# -------------------------
# FEATURE IMPORTANCE
# -------------------------
st.subheader("📊 Feature Importance")

fig1 = px.bar(
    x=X.columns,
    y=rf_model.feature_importances_,
    title="Random Forest Importance"
)
st.plotly_chart(fig1)

fig2 = px.bar(
    x=X.columns,
    y=xgb_model.feature_importances_,
    title="XGBoost Importance"
)
st.plotly_chart(fig2)

# -------------------------
# 3D VISUALIZATION
# -------------------------
st.subheader("📊 3D Feature Space")

fig3d = px.scatter_3d(
    data,
    x="Peak_2theta",
    y="FWHM",
    z="Surface Area",
    color="Activity"
)
st.plotly_chart(fig3d)

# -------------------------
# XRD SIMULATION
# -------------------------
st.subheader("🔬 Simulated XRD Pattern")

theta = np.linspace(10, 80, 200)
intensity = np.exp(-((theta - peak)**2)/(2*fwhm**2))

fig, ax = plt.subplots()
ax.plot(theta, intensity)
ax.set_xlabel("2θ")
ax.set_ylabel("Intensity")
st.pyplot(fig)

# -------------------------
# CSV UPLOAD
# -------------------------
st.subheader("📂 Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    df["Nanoparticle"] = le.transform(df["Nanoparticle"])

    preds_rf = rf_model.predict(df)
    preds_xgb = xgb_model.predict(df)

    df["RF_Prediction"] = preds_rf
    df["XGB_Prediction"] = preds_xgb
    df["Final"] = (preds_rf + preds_xgb) / 2

    st.write(df)

# -------------------------
# DATASET EXPLORER
# -------------------------
st.subheader("📉 Dataset Explorer")

if st.checkbox("Show Dataset"):
    st.write(data)

# -------------------------
# PERFORMANCE METER
# -------------------------
st.subheader("⚡ Performance Meter")

st.progress(min(int(final_pred), 100))
