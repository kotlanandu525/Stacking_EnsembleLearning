import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

from sklearn.metrics import accuracy_score

# =================================
# PAGE CONFIG
# =================================
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

# =================================
# SESSION STATE
# =================================
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# =================================
# TITLE
# =================================
st.title("🏦 Loan Approval Prediction System")
st.caption("Stacking Ensemble • Reduced Features")
st.markdown("---")

# =================================
# LOAD DATA (SAFE WAY)
# =================================
DATA_FILE = "train_u6lujuX_CVtuZ9i.csv"

if not os.path.exists(DATA_FILE):
    st.error(
        f"❌ Dataset not found!\n\n"
        f"Please make sure **{DATA_FILE}** is in the same folder as **app.py**."
    )
    st.stop()

data = pd.read_csv(DATA_FILE)

# =================================
# FEATURE SELECTION (ONLY 4 FEATURES)
# =================================
selected_features = [
    "Credit_History",
    "ApplicantIncome",
    "LoanAmount",
    "Property_Area"
]

X = data[selected_features]
y = data["Loan_Status"].map({"Y": 1, "N": 0})

num_features = ["Credit_History", "ApplicantIncome", "LoanAmount"]
cat_features = ["Property_Area"]

# =================================
# PREPROCESSING
# =================================
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

# =================================
# STACKING MODEL
# =================================
base_models = [
    ("lr", LogisticRegression(max_iter=1000)),
    ("dt", DecisionTreeClassifier(max_depth=4, random_state=42)),
    ("rf", RandomForestClassifier(
        n_estimators=120,
        max_depth=6,
        random_state=42
    ))
]

stack_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(),
    cv=5,
    n_jobs=-1
)

model = Pipeline([
    ("preprocessing", preprocessor),
    ("stacking", stack_model)
])

# =================================
# TRAIN MODEL
# =================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

with st.spinner("Training model..."):
    model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# =================================
# SIDEBAR INPUTS
# =================================
st.sidebar.header("🧾 Applicant Details")

Credit_History = st.sidebar.selectbox("Credit History", [1.0, 0.0])
ApplicantIncome = st.sidebar.number_input("Applicant Income", min_value=0)
LoanAmount = st.sidebar.number_input("Loan Amount", min_value=0)
Property_Area = st.sidebar.selectbox(
    "Property Area", ["Urban", "Semiurban", "Rural"]
)

if st.sidebar.button("🔮 Predict Loan Status"):
    input_data = pd.DataFrame([{
        "Credit_History": Credit_History,
        "ApplicantIncome": ApplicantIncome,
        "LoanAmount": LoanAmount,
        "Property_Area": Property_Area
    }])

    st.session_state.prediction = model.predict(input_data)[0]

# =================================
# RESULT DISPLAY (DIRECT)
# =================================
st.markdown("## 📌 Prediction Result")

if st.session_state.prediction is None:
    st.info("👈 Enter details and click **Predict Loan Status**")
else:
    if st.session_state.prediction == 1:
        st.success("🎉 **LOAN APPROVED**")
    else:
        st.error("🚫 **LOAN REJECTED**")

# =================================
# MODEL INFO
# =================================
st.markdown("---")
col1, col2, col3 = st.columns(3)

col1.metric("🎯 Accuracy", f"{accuracy*100:.2f}%")
col2.metric("🧠 Features Used", 4)
col3.metric("📦 Dataset Size", data.shape[0])

# =================================
# FOOTER
# =================================
st.markdown("---")
st.markdown(
    "<center><b>Loan Approval Prediction</b> | Stacking Ensemble | Streamlit</center>",
    unsafe_allow_html=True
)
