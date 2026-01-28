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

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Smart Loan Approval System",
    page_icon="🏦",
    layout="wide"
)

# ==================================================
# SESSION STATE
# ==================================================
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "base_preds" not in st.session_state:
    st.session_state.base_preds = {}

# ==================================================
# TITLE & DESCRIPTION
# ==================================================
st.title("🎯 Smart Loan Approval System – Stacking Model")
st.markdown(
    """
    **This system uses a Stacking Ensemble Machine Learning model to predict whether a loan
    will be approved by combining multiple ML models for better decision making.**
    """
)
st.markdown("---")

# ==================================================
# LOAD DATA (SAFE PATH)
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "train_u6lujuX_CVtuZ9i.csv")

if not os.path.exists(DATA_FILE):
    st.error("Dataset not found. Ensure CSV is in the same folder as app.py")
    st.stop()

data = pd.read_csv(DATA_FILE)

# ==================================================
# SELECT FEATURES
# ==================================================
features = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Self_Employed",
    "Property_Area"
]

X = data[features]
y = data["Loan_Status"].map({"Y": 1, "N": 0})

num_features = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]

cat_features = [
    "Self_Employed",
    "Property_Area"
]

# ==================================================
# PREPROCESSING
# ==================================================
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

# ==================================================
# MODELS
# ==================================================
lr = LogisticRegression(max_iter=1000)
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
rf = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42)

stack_model = StackingClassifier(
    estimators=[
        ("Logistic Regression", lr),
        ("Decision Tree", dt),
        ("Random Forest", rf)
    ],
    final_estimator=LogisticRegression(),
    cv=5
)

model = Pipeline([
    ("preprocessing", preprocessor),
    ("stacking", stack_model)
])

# ==================================================
# TRAIN MODEL
# ==================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

# ==================================================
# SIDEBAR INPUTS
# ==================================================
st.sidebar.header("🧾 Applicant Details")

ApplicantIncome = st.sidebar.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.sidebar.number_input("Co-Applicant Income", min_value=0)
LoanAmount = st.sidebar.number_input("Loan Amount", min_value=0)
Loan_Amount_Term = st.sidebar.number_input("Loan Amount Term", min_value=0)

Credit_History = st.sidebar.radio(
    "Credit History",
    ["Yes", "No"]
)

Employment_Status = st.sidebar.selectbox(
    "Employment Status",
    ["Salaried", "Self-Employed"]
)

Property_Area = st.sidebar.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)

# ==================================================
# STACKING ARCHITECTURE DISPLAY
# ==================================================
st.subheader("🧠 Model Architecture (Stacking Ensemble)")
st.markdown(
    """
    **Base Models Used:**
    - Logistic Regression  
    - Decision Tree  
    - Random Forest  

    **Meta Model Used:**
    - Logistic Regression  

    📌 *Base models generate individual predictions which are combined by the meta-model
    to produce the final decision.*
    """
)

st.markdown("---")

# ==================================================
# PREDICTION BUTTON
# ==================================================
if st.sidebar.button("🔘 Check Loan Eligibility (Stacking Model)"):
    input_df = pd.DataFrame([{
        "ApplicantIncome": ApplicantIncome,
        "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount": LoanAmount,
        "Loan_Amount_Term": Loan_Amount_Term,
        "Credit_History": 1.0 if Credit_History == "Yes" else 0.0,
        "Self_Employed": "Yes" if Employment_Status == "Self-Employed" else "No",
        "Property_Area": Property_Area
    }])

    # Base model predictions
    X_processed = preprocessor.fit_transform(X_train)
    X_input_processed = preprocessor.transform(input_df)

    st.session_state.base_preds = {
        "Logistic Regression": lr.fit(X_processed, y_train).predict(X_input_processed)[0],
        "Decision Tree": dt.fit(X_processed, y_train).predict(X_input_processed)[0],
        "Random Forest": rf.fit(X_processed, y_train).predict(X_input_processed)[0],
    }

    st.session_state.prediction = model.predict(input_df)[0]

# ==================================================
# OUTPUT SECTION
# ==================================================
st.subheader("📌 Loan Eligibility Result")

if st.session_state.prediction is not None:
    if st.session_state.prediction == 1:
        st.success("✅ **LOAN APPROVED**")
    else:
        st.error("❌ **LOAN REJECTED**")

    st.markdown("### 📊 Base Model Predictions")
    for model_name, pred in st.session_state.base_preds.items():
        st.write(f"**{model_name}** → {'Approved' if pred == 1 else 'Rejected'}")

    st.markdown("### 🧠 Final Stacking Decision")
    st.write("The meta-model combines all base model outputs to make the final decision.")

    # ==================================================
    # BUSINESS EXPLANATION (MANDATORY)
    # ==================================================
    st.markdown("### 💼 Business Explanation")
    if st.session_state.prediction == 1:
        st.write(
            "Based on income stability, credit history, and combined predictions from multiple models, "
            "the applicant is likely to repay the loan. Therefore, the stacking model predicts **loan approval**."
        )
    else:
        st.write(
            "Based on income patterns, credit risk, and combined predictions from multiple models, "
            "the applicant is unlikely to repay the loan. Therefore, the stacking model predicts **loan rejection**."
        )

else:
    st.info("👈 Enter applicant details and click **Check Loan Eligibility**")

# ==================================================
# FOOTER
# ==================================================
st.markdown("---")
st.markdown(
    "<center><b>Smart Loan Approval System</b> | Stacking Ensemble | Streamlit</center>",
    unsafe_allow_html=True
)
