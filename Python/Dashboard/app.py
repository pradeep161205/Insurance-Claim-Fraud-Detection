import streamlit as st

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Insurance Claim Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown("""
<style>
.main-title{
    font-size:42px;
    font-weight:bold;
    color:#0E76A8;
}

.sub-title{
    font-size:22px;
    color:gray;
}

.card{
    background:#f7f7f7;
    padding:20px;
    border-radius:10px;
    box-shadow:2px 2px 10px lightgray;
}

.footer{
    text-align:center;
    color:gray;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.image(
    "https://img.icons8.com/color/96/shield.png",
    width=80
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "",
    [
        "🏠 Home",
        "📊 Dashboard",
        "🤖 Prediction",
        "📈 Model Performance",
        "ℹ About"
    ]
)

# -------------------------------
# HOME
# -------------------------------

if page == "🏠 Home":

    st.markdown(
        '<p class="main-title">🛡 Insurance Claim Fraud Detection</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="sub-title">Machine Learning Based Fraud Detection System</p>',
        unsafe_allow_html=True
    )

    st.write("---")

    col1, col2 = st.columns([2,1])

    with col1:

        st.header("📌 Project Overview")

        st.write("""
This project predicts whether an insurance claim is **Fraudulent**
or **Genuine** using Machine Learning.

The project includes:

- Data Cleaning
- Feature Engineering
- Exploratory Data Analysis
- Machine Learning
- Model Evaluation
- Interactive Dashboard
""")

    with col2:

        st.success("Model Ready ✅")

        st.info("Dataset : 5000 Claims")

        st.info("Algorithms : 5")

        st.info("Prediction : Fraud / Genuine")

    st.write("---")

    st.header("Project Workflow")

    st.markdown("""
1. Data Collection

2. Data Understanding

3. Data Cleaning

4. EDA

5. Feature Engineering

6. Data Preprocessing

7. Model Building

8. Model Evaluation

9. Dashboard
""")

# -------------------------------
# PLACEHOLDERS
# -------------------------------

elif page == "📊 Dashboard":

    import pandas as pd
    import plotly.express as px

    # Load Dataset
    df = pd.read_csv("Python/Data/final/final_dataset.csv")

    st.title("📊 Insurance Dashboard")

    st.markdown("---")

    # ===============================
    # KPI Cards
    # ===============================

    total = len(df)
    fraud = len(df[df["Fraud_Label"] == 1])
    genuine = len(df[df["Fraud_Label"] == 0])
    percentage = (fraud / total) * 100

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("📄 Total Claims", total)
    c2.metric("🚨 Fraud Claims", fraud)
    c3.metric("✅ Genuine Claims", genuine)
    c4.metric("📈 Fraud %", f"{percentage:.2f}%")

    st.markdown("---")

    # ===============================
    # Charts
    # ===============================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Fraud Distribution")

        pie = px.pie(
            values=df["Fraud_Label"].value_counts().values,
            names=["Genuine", "Fraud"],
            hole=0.45,
            color_discrete_sequence=["green", "red"]
        )

        st.plotly_chart(pie, use_container_width=True)

    with col2:

        st.subheader("Age Distribution")

        hist = px.histogram(
            df,
            x="Age",
            nbins=25,
            color_discrete_sequence=["royalblue"]
        )

        st.plotly_chart(hist, use_container_width=True)

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:

        st.subheader("Annual Income")

        fig = px.box(
            df,
            y="Annual_Income",
            color="Fraud_Label",
            color_discrete_sequence=["green", "red"]
        )

        st.plotly_chart(fig, use_container_width=True)

    with col4:

        st.subheader("Claim Amount")

        fig = px.box(
            df,
            y="Total_Claim_Amount",
            color="Fraud_Label",
            color_discrete_sequence=["green", "red"]
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Correlation Heatmap")

    corr = df.corr(numeric_only=True)

    heat = px.imshow(
        corr,
        text_auto=False,
        aspect="auto",
        color_continuous_scale="RdBu_r"
    )

    st.plotly_chart(heat, use_container_width=True)

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(df.head(10), use_container_width=True)

elif page == "🤖 Prediction":

    import pandas as pd
    import joblib
    import streamlit as st

    st.title("🤖 Insurance Fraud Prediction")

    model = joblib.load("Python/models/dashboard_model.pkl")
    scaler = joblib.load("Python/models/dashboard_scaler.pkl")
    encoders = joblib.load("Python/models/dashboard_encoders.pkl")

    st.subheader("Customer Details")

    col1, col2 = st.columns(2)

    with col1:

        age = st.slider("Age",18,75,35)

        gender = st.selectbox(
            "Gender",
            encoders["Gender"].classes_
        )

        occupation = st.selectbox(
            "Occupation",
            encoders["Occupation"].classes_
        )

        income = st.number_input(
            "Annual Income",
            min_value=100000,
            value=800000
        )

        policy = st.selectbox(
            "Policy Type",
            encoders["Policy_Type"].classes_
        )

        premium = st.number_input(
            "Premium Amount",
            value=25000
        )

        coverage = st.number_input(
            "Coverage Amount",
            value=1500000
        )

        vehicle = st.selectbox(
            "Vehicle Type",
            encoders["Vehicle_Type"].classes_
        )

        brand = st.selectbox(
            "Vehicle Brand",
            encoders["Vehicle_Brand"].classes_
        )

        vehicle_value = st.number_input(
            "Vehicle Value",
            value=900000
        )

    with col2:

        days = st.slider(
            "Days To Report",
            0,
            30,
            5
        )

        weather = st.selectbox(
            "Weather",
            encoders["Weather"].classes_
        )

        police = st.selectbox(
            "Police Report",
            encoders["Police_Report"].classes_
        )

        witness = st.slider(
            "Witness Count",
            0,
            10,
            2
        )

        repair = st.number_input(
            "Repair Cost",
            value=50000
        )

        medical = st.number_input(
            "Medical Cost",
            value=20000
        )

        claim = st.number_input(
            "Total Claim Amount",
            value=100000
        )

        previous = st.slider(
            "Previous Claims",
            0,
            5,
            0
        )

        history = st.selectbox(
            "Previous Fraud History",
            encoders["Previous_Fraud_History"].classes_
        )

    if st.button("🚀 Predict Fraud", use_container_width=True):

        input_data = pd.DataFrame({

            "Age":[age],

            "Gender":[encoders["Gender"].transform([gender])[0]],

            "Occupation":[encoders["Occupation"].transform([occupation])[0]],

            "Annual_Income":[income],

            "Policy_Type":[encoders["Policy_Type"].transform([policy])[0]],

            "Premium_Amount":[premium],

            "Coverage_Amount":[coverage],

            "Vehicle_Type":[encoders["Vehicle_Type"].transform([vehicle])[0]],

            "Vehicle_Brand":[encoders["Vehicle_Brand"].transform([brand])[0]],

            "Vehicle_Value":[vehicle_value],

            "Days_to_Report":[days],

            "Weather":[encoders["Weather"].transform([weather])[0]],

            "Police_Report":[encoders["Police_Report"].transform([police])[0]],

            "Witness_Count":[witness],

            "Repair_Cost":[repair],

            "Medical_Cost":[medical],

            "Total_Claim_Amount":[claim],

            "Previous_Claims":[previous],

            "Previous_Fraud_History":[encoders["Previous_Fraud_History"].transform([history])[0]]

        })

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]

        probability = model.predict_proba(input_scaled)[0][1]

        st.write("---")

        if prediction == 1:
            st.error("🚨 Fraudulent Claim")
        else:
            st.success("✅ Genuine Claim")

        st.metric(
            "Fraud Probability",
            f"{probability*100:.2f}%"
        )

        st.progress(float(probability))

elif page == "📈 Model Performance":

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import joblib

    from sklearn.metrics import (
        confusion_matrix,
        classification_report,
        roc_curve,
        roc_auc_score
    )

    st.title("📈 Model Performance")

    model = joblib.load("Python/models/best_fraud_model.pkl")

    X_test = joblib.load("Python/models/X_test.pkl")
    y_test = joblib.load("Python/models/y_test.pkl")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ==========================
    # Metrics
    # ==========================

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Classification Report")
        report = classification_report(
            y_test,
            y_pred,
            output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose())

    with col2:
        st.subheader("ROC-AUC Score")
        auc = roc_auc_score(y_test, y_prob)
        st.metric("ROC-AUC", f"{auc:.4f}")

    st.markdown("---")

    # ==========================
    # Confusion Matrix
    # ==========================

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Genuine", "Fraud"],
        yticklabels=["Genuine", "Fraud"],
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    st.markdown("---")

    # ==========================
    # ROC Curve
    # ==========================

    st.subheader("ROC Curve")

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    fig2, ax2 = plt.subplots(figsize=(6, 5))

    ax2.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax2.plot([0, 1], [0, 1], "--", color="red")

    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend()

    st.pyplot(fig2)

else:

    st.title("ℹ About Project")

    st.markdown("""
# Insurance Claim Fraud Detection

### Final Year Data Science Project

This project predicts whether an insurance claim is
Fraudulent or Genuine using Machine Learning.

---

### Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Plotly
- Matplotlib
- Seaborn

---

### Machine Learning Workflow

✔ Data Collection

✔ Data Understanding

✔ Data Cleaning

✔ Exploratory Data Analysis

✔ Feature Engineering

✔ Data Preprocessing

✔ Model Building

✔ Model Evaluation

✔ Streamlit Dashboard

---

### Dataset

- 5000 Insurance Claims
- Binary Classification
- Fraud Detection

---

### Developed By

**PRADEEP P**

Final Year Data Science Project
""")