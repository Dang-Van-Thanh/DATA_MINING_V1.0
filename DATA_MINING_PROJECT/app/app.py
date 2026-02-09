import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ===============================
# LOAD MODEL & PIPELINE
# ===============================
@st.cache_resource
def load_artifacts():
    model = joblib.load("outputs/models/best_model.pkl")
    pipeline = joblib.load("outputs/models/feature_pipeline.pkl")
    return model, pipeline


model, pipeline = load_artifacts()

# ===============================
# APP CONFIG
# ===============================
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Bank Marketing – Term Deposit Prediction")
st.markdown("""
Demo app cho **Đề tài 4 – Khai phá dữ liệu giao dịch ngân hàng**  
Dự đoán khả năng **khách hàng đăng ký term deposit**
""")

# ===============================
# SIDEBAR INPUT
# ===============================
st.sidebar.header("📋 Thông tin khách hàng")

age = st.sidebar.slider("Age", 18, 95, 35)
job = st.sidebar.selectbox(
    "Job",
    ["admin.", "technician", "services", "management",
     "retired", "blue-collar", "unemployed",
     "entrepreneur", "housemaid", "student", "self-employed"]
)
marital = st.sidebar.selectbox("Marital status", ["married", "single", "divorced"])
education = st.sidebar.selectbox("Education", ["primary", "secondary", "tertiary"])
default = st.sidebar.selectbox("Has credit default?", ["no", "yes"])
housing = st.sidebar.selectbox("Has housing loan?", ["no", "yes"])
loan = st.sidebar.selectbox("Has personal loan?", ["no", "yes"])
contact = st.sidebar.selectbox("Contact type", ["cellular", "telephone"])
month = st.sidebar.selectbox(
    "Last contact month",
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"]
)

balance = st.sidebar.number_input("Account balance", -5000, 100000, 1500)
campaign = st.sidebar.slider("Number of contacts (campaign)", 1, 50, 2)
pdays = st.sidebar.number_input("Days since last contact", -1, 1000, -1)
previous = st.sidebar.slider("Previous contacts", 0, 50, 0)
poutcome = st.sidebar.selectbox(
    "Previous campaign outcome",
    ["unknown", "failure", "success"]
)

# ===============================
# INPUT DATAFRAME
# ===============================
input_df = pd.DataFrame([{
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "month": month,
    "balance": balance,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome
}])

# ===============================
# PREDICTION
# ===============================
if st.button("🔮 Dự đoán"):
    X_transformed = pipeline.transform(input_df)
    proba = model.predict_proba(X_transformed)[0, 1]

    st.subheader("📊 Kết quả dự đoán")

    st.metric(
        label="Xác suất đăng ký term deposit",
        value=f"{proba:.2%}"
    )

    if proba >= 0.5:
        st.success("✅ Khách hàng CÓ khả năng đăng ký")
    else:
        st.warning("⚠️ Khách hàng KHÓ đăng ký")

    # ===============================
    # INSIGHT
    # ===============================
    st.subheader("📌 Gợi ý hành động")

    if campaign > 4:
        st.write("• Giảm số lần gọi – khách dễ bị làm phiền")
    if balance > 5000:
        st.write("• Ưu tiên gói tiết kiệm giá trị cao")
    if housing == "yes" and loan == "no":
        st.write("• Phù hợp cross-sell term deposit")
    if poutcome == "success":
        st.write("• Khách có lịch sử phản hồi tốt – nên ưu tiên")

st.markdown("---")
st.caption("📚 Data Mining Project – Bank Marketing Dataset (UCI)")
