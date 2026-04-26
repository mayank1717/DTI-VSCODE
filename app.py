import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import plotly.express as px
import matplotlib.pyplot as plt

# ========================
# PAGE CONFIG
# ========================
st.set_page_config(page_title="NanoCatalyst AI", layout="wide", initial_sidebar_state="expanded")

# ========================
# CUSTOM CSS
# ========================
st.markdown("""
<style>
    .main { padding: 0rem 2rem; }
    .metric-card { background-color: #0E1117; padding: 20px; border-radius: 10px; border-left: 5px solid #FF6B35; }
</style>
""", unsafe_allow_html=True)

# ========================
# TITLE & DESCRIPTION
# ========================
st.title("🔬 NanoCatalyst AI Platform")
st.markdown("### XRD-Based Catalytic Activity Prediction System")
st.markdown("---")

# ========================
# LOAD DATA & MODELS
# ========================
@st.cache_data
def load_dataset():
    try:
        # Try multiple file paths
        paths = [
            "nanoparticle_catalytic_dataset_5000.csv",
            "nanoparticle_catalytic_dataset_5000 (1).csv",
            "cli/nanoparticle_catalytic_dataset_5000 (1).csv"
        ]
        for path in paths:
            if os.path.exists(path):
                return pd.read_csv(path)
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

@st.cache_resource
def load_or_train_model():
    try:
        # Load pre-trained model if exists
        if os.path.exists("catalytic_model.pkl"):
            with open("catalytic_model.pkl", "rb") as f:
                model = pickle.load(f)
            with open("encoder.pkl", "rb") as f:
                encoder = pickle.load(f)
            return model, encoder, True
    except:
        pass
    
    # Train new model if not found
    data = load_dataset()
    if data is None:
        st.error("Cannot load dataset")
        return None, None, False
    
    X = data.drop('Catalytic_Activity', axis=1)
    y = data['Catalytic_Activity']
    
    # Encode categorical
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded = encoder.fit_transform(X[['Nanoparticle']])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['Nanoparticle']))
    X = X.drop('Nanoparticle', axis=1)
    X = pd.concat([X.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    
    # Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save
    with open("catalytic_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
    
    return model, encoder, False

# Load data and model
data = load_dataset()
if data is not None:
    model, encoder, model_loaded = load_or_train_model()
else:
    uploaded_file = st.file_uploader("Upload nanoparticle dataset", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        model, encoder, model_loaded = load_or_train_model()
    else:
        st.info("📁 Upload a CSV file to get started or download sample data from the sidebar.")
        st.stop()

# ========================
# MODEL INFO
# ========================
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🌲 Model Type", "Random Forest")
with col2:
    X_val = data.drop('Catalytic_Activity', axis=1)
    y_val = data['Catalytic_Activity']
    if model_loaded:
        st.metric("✅ Status", "Loaded")
    else:
        st.metric("📝 Status", "Trained")
with col3:
    st.metric("📊 Samples", len(data))

st.markdown("---")

# ========================
# MAIN PREDICTION INTERFACE
# ========================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("⚙️ Input Parameters")
    
    col_inputs = st.columns(2)
    
    with col_inputs[0]:
        nanoparticle = st.selectbox("Nanoparticle Type", encoder.categories_[0].tolist())
        peak_2theta = st.slider("Peak 2θ (°)", 10.0, 90.0, 50.0, 1.0)
        fwhm = st.slider("FWHM", 0.1, 1.0, 0.5, 0.1)
        crystallite = st.slider("Crystallite Size (nm)", 1, 150, 50)
    
    with col_inputs[1]:
        lattice = st.slider("Lattice Parameter", 3.0, 10.0, 6.5, 0.5)
        intensity = st.slider("Intensity", 100, 1000, 500, 50)
        surface = st.slider("Surface Area", 10, 200, 70, 10)
        d_spacing = st.slider("d-spacing (Å)", 1.0, 4.0, 2.0, 0.2)

    microstrain = st.slider("Microstrain", 0.01, 0.1, 0.05, 0.01)
    main_phase = st.slider("Main Phase Fraction", 0.5, 1.0, 0.85, 0.05)
    impurity_phase = st.slider("Impurity Phase Fraction", 0.0, 0.5, 0.1, 0.05)

with col2:
    st.subheader("🎯 Prediction")
    
    # Prepare input
    encoded_nano = encoder.transform([[nanoparticle]])
    feature_names = encoder.get_feature_names_out(['Nanoparticle']).tolist()
    
    input_dict = {col: 0 for col in feature_names}
    for i, col in enumerate(feature_names):
        input_dict[col] = encoded_nano[0][i]
    
    # Get feature order from training
    X_sample = data.drop('Catalytic_Activity', axis=1)
    X_encoded_sample = encoder.transform(X_sample[['Nanoparticle']])
    X_encoded_df = pd.DataFrame(X_encoded_sample, columns=feature_names)
    X_sample_processed = X_sample.drop('Nanoparticle', axis=1)
    X_sample_processed = pd.concat([X_sample_processed.reset_index(drop=True), X_encoded_df.reset_index(drop=True)], axis=1)
    
    input_dict.update({
        "Peak_2theta": peak_2theta,
        "FWHM": fwhm,
        "Crystallite_Size_nm": crystallite,
        "Lattice_Param": lattice,
        "Intensity": intensity,
        "Surface_Area": surface,
        "d_spacing_A": d_spacing,
        "Microstrain": microstrain,
        "Main_Phase_Fraction": main_phase,
        "Impurity_Phase_Fraction": impurity_phase
    })
    
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[X_sample_processed.columns]
    
    # Predict
    prediction = model.predict(input_df)[0]
    
    # Display
    st.metric("Predicted Activity", f"{prediction:.2f}", delta=None)
    
    # Recommendation
    if prediction > 70:
        st.success("🔥 Excellent Catalyst", icon="✅")
    elif prediction > 50:
        st.warning("⚡ Good Catalyst", icon="⭐")
    else:
        st.error("❌ Low Activity", icon="⚠️")

st.markdown("---")

# ========================
# FEATURE IMPORTANCE
# ========================
st.subheader("📊 Feature Importance Analysis")

feature_imp = pd.DataFrame({
    'Feature': X_sample_processed.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False).head(10)

fig_imp = px.bar(feature_imp, x='Importance', y='Feature', orientation='h', 
                 title='Top 10 Important Features', color='Importance', 
                 color_continuous_scale='Blues')
st.plotly_chart(fig_imp, use_container_width=True)

# ========================
# TABS
# ========================
tab1, tab2, tab3 = st.tabs(["📈 Dataset Explorer", "🔍 Batch Prediction", "📉 Data Visualization"])

with tab1:
    st.subheader("Dataset Overview")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.checkbox("Show Raw Data"):
            st.dataframe(data, use_container_width=True)
    
    with col2:
        if st.checkbox("Show Statistics"):
            st.dataframe(data.describe(), use_container_width=True)

with tab2:
    st.subheader("Batch Prediction")
    
    uploaded_batch = st.file_uploader("Upload CSV for batch predictions", type=["csv"], key="batch")
    
    if uploaded_batch:
        batch_data = pd.read_csv(uploaded_batch)
        
        try:
            # Prepare batch
            batch_X = batch_data.drop('Catalytic_Activity', axis=1, errors='ignore').copy()
            batch_encoded = encoder.transform(batch_X[['Nanoparticle']])
            batch_encoded_df = pd.DataFrame(batch_encoded, columns=feature_names)
            batch_X = batch_X.drop('Nanoparticle', axis=1)
            batch_X = pd.concat([batch_X.reset_index(drop=True), batch_encoded_df.reset_index(drop=True)], axis=1)
            batch_X = batch_X[X_sample_processed.columns]
            
            # Predict
            batch_preds = model.predict(batch_X)
            batch_data['Predicted_Activity'] = batch_preds
            
            st.dataframe(batch_data, use_container_width=True)
            
            # Download
            csv = batch_data.to_csv(index=False)
            st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error: {e}")

with tab3:
    st.subheader("Data Distributions")
    
    col_dist = st.columns(2)
    
    with col_dist[0]:
        feature_select = st.selectbox("Select Feature", data.columns, key="dist1")
        fig_hist = px.histogram(data, x=feature_select, nbins=50, title=f"{feature_select} Distribution")
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col_dist[1]:
        fig_box = px.box(data, y=feature_select, title=f"{feature_select} Box Plot")
        st.plotly_chart(fig_box, use_container_width=True)

# ========================
# FOOTER
# ========================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 12px;'>
    🔬 NanoCatalyst AI Platform | Data Science & Machine Learning<br>
    Built with Streamlit | Model: Random Forest Regressor
</div>
""", unsafe_allow_html=True)
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import sqlite3
import shap
import base64
import io

# ---------------- LOAD ----------------
data = pd.read_csv("nanoparticle_catalytic_dataset_5000.csv")

rf = joblib.load("rf.pkl")
xgb = joblib.load("xgb.pkl")
le = joblib.load("encoder.pkl")

# Encode dataset
data["Nanoparticle"] = le.transform(data["Nanoparticle"])

# SHAP
explainer = shap.Explainer(rf)

# ---------------- DATABASE ----------------
conn = sqlite3.connect("experiments.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS experiments (
    nanoparticle TEXT,
    peak REAL,
    fwhm REAL,
    size REAL,
    surface REAL,
    prediction REAL
)
""")
conn.commit()

# ---------------- APP ----------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "NanoCatalyst AI"

# ---------------- KPI CARDS ----------------
def kpi_card(title, id):
    return dbc.Card(
        dbc.CardBody([
            html.H6(title, className="text-muted"),
            html.H2(id=id)
        ]),
        className="shadow"
    )

# ---------------- LAYOUT ----------------
app.layout = dbc.Container(fluid=True, children=[

    html.H1("🔬 NanoCatalyst AI Platform", className="text-center my-4"),

    dbc.Row([
        dbc.Col(kpi_card("Random Forest", "rf_score")),
        dbc.Col(kpi_card("XGBoost", "xgb_score")),
        dbc.Col(kpi_card("Final Prediction", "final_pred")),
    ]),

    html.Br(),

    dbc.Tabs([

        # ================= TAB 1 =================
        dbc.Tab(label="🧠 Prediction", children=[

            dbc.Row([

                dbc.Col([
                    html.Label("Nanoparticle"),
                    dcc.Dropdown(
                        id='nano',
                        options=[{"label": i, "value": i} for i in le.classes_],
                        value=le.classes_[0]
                    ),

                    html.Br(),
                    html.Label("Peak 2θ"),
                    dcc.Slider(10, 80, 1, value=50, id='peak'),

                    html.Label("FWHM"),
                    dcc.Slider(0.1, 1.0, 0.1, value=0.5, id='fwhm'),

                    html.Label("Size"),
                    dcc.Slider(1, 100, 1, value=30, id='size'),

                    html.Label("Surface Area"),
                    dcc.Slider(10, 200, 1, value=70, id='surface'),

                ], width=4),

                dbc.Col([
                    html.H3("🤖 AI Recommendation"),
                    html.Div(id="recommendation"),

                    dcc.Graph(id="xrd_plot"),
                    dcc.Graph(id="shap_plot")

                ], width=8)

            ])
        ]),

        # ================= TAB 2 =================
        dbc.Tab(label="📊 Visualization", children=[

            dcc.Graph(id="3d_plot"),
            dcc.Graph(id="rf_importance"),
            dcc.Graph(id="xgb_importance")

        ]),

        # ================= TAB 3 =================
        dbc.Tab(label="📂 Data", children=[

            html.H4("Upload CSV for Batch Prediction"),
            dcc.Upload(
                id='upload',
                children=html.Button("Upload CSV"),
                multiple=False
            ),

            html.Div(id='output-data'),

            html.Hr(),

            html.H4("Stored Experiments"),
            html.Div(id="db_data")

        ])

    ])

])

# ---------------- CALLBACK ----------------
@app.callback(
    [
        Output("rf_score", "children"),
        Output("xgb_score", "children"),
        Output("final_pred", "children"),
        Output("recommendation", "children"),
        Output("xrd_plot", "figure"),
        Output("3d_plot", "figure"),
        Output("rf_importance", "figure"),
        Output("xgb_importance", "figure"),
        Output("shap_plot", "figure"),
        Output("db_data", "children")
    ],
    [
        Input('nano', 'value'),
        Input('peak', 'value'),
        Input('fwhm', 'value'),
        Input('size', 'value'),
        Input('surface', 'value')
    ]
)
def update(nano, peak, fwhm, size, surface):

    nano_enc = le.transform([nano])[0]

    df = pd.DataFrame([{
        "Nanoparticle": nano_enc,
        "Peak_2theta": peak,
        "FWHM": fwhm,
        "Size": size,
        "Surface Area": surface
    }])

    # Predictions
    rf_pred = rf.predict(df)[0]
    xgb_pred = xgb.predict(df)[0]
    final = (rf_pred + xgb_pred) / 2

    # Save to DB
    cursor.execute("""
    INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?)
    """, (nano, peak, fwhm, size, surface, final))
    conn.commit()

    # Recommendation
    if final > 80:
        rec = "🔥 High Performance Catalyst"
    elif final > 50:
        rec = "⚡ Moderate Catalyst"
    else:
        rec = "❗ Low Activity Catalyst"

    # XRD
    theta = np.linspace(10, 80, 200)
    intensity = np.exp(-((theta - peak)**2)/(2*fwhm**2))
    xrd_fig = px.line(x=theta, y=intensity, title="XRD Pattern")

    # 3D
    fig3d = px.scatter_3d(
        data,
        x="Peak_2theta",
        y="FWHM",
        z="Surface Area",
        color="Activity"
    )

    # Feature importance
    rf_imp = px.bar(x=df.columns, y=rf.feature_importances_, title="RF Importance")
    xgb_imp = px.bar(x=df.columns, y=xgb.feature_importances_, title="XGB Importance")

    # SHAP
    shap_values = explainer(df)
    shap_fig = px.bar(
        x=df.columns,
        y=shap_values.values[0],
        title="SHAP Explanation"
    )

    # Load DB
    df_hist = pd.read_sql("SELECT * FROM experiments", conn)

    table = dbc.Table.from_dataframe(df_hist.tail(5), striped=True, bordered=True)

    return (
        f"{rf_pred:.2f}",
        f"{xgb_pred:.2f}",
        f"{final:.2f}",
        rec,
        xrd_fig,
        fig3d,
        rf_imp,
        xgb_imp,
        shap_fig,
        table
    )

# ---------------- CSV UPLOAD ----------------
@app.callback(
    Output('output-data', 'children'),
    Input('upload', 'contents')
)
def upload_data(contents):
    if contents:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

        df["Nanoparticle"] = le.transform(df["Nanoparticle"])

        df["Prediction"] = (rf.predict(df) + xgb.predict(df)) / 2

        return dcc.Graph(
            figure=px.scatter(df, x="Peak_2theta", y="Prediction")
        )

# ---------------- RUN ----------------
if __name__ == '__main__':
    app.run(debug=True)
