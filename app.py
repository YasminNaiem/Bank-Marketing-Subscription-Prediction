import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# إعداد الصفحة
st.set_page_config(
    page_title="Bank Marketing Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# تحميل النماذج
@st.cache_resource
def load_models():
    try:
        with open('best_svm_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('pca_final.pkl', 'rb') as f:
            pca = pickle.load(f)
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        return model, pca, preprocessor, True
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, False

model, pca, preprocessor, models_loaded = load_models()

# CSS مخصص
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
    }
    .result-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin-top: 1rem;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        font-size: 1.1rem;
        padding: 0.5rem 2rem;
        border-radius: 30px;
        transition: transform 0.2s;
    }
    .stButton button:hover {
        transform: translateY(-2px);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🏦 Bank Marketing Campaign Predictor</h1>
    <p>AI-Powered Decision Support for Term Deposit Subscriptions</p>
</div>
""", unsafe_allow_html=True)

if not models_loaded:
    st.error("⚠️ Models not loaded. Please check the .pkl files exist.")
    st.stop()

# القوائم الثابتة
jobs = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
        'retired', 'self-employed', 'services', 'student', 'technician',
        'unemployed', 'unknown']

marital = ['divorced', 'married', 'single', 'unknown']
education = ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate',
             'professional.course', 'university.degree', 'unknown']
yes_no = ['no', 'yes', 'unknown']
contact_type = ['cellular', 'telephone']
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
days = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
poutcomes = ['failure', 'nonexistent', 'success']

# الاقتصادية الثابتة
economic = {
    'emp.var.rate': 1.1,
    'cons.price.idx': 93.994,
    'cons.conf.idx': -36.4,
    'euribor3m': 4.857,
    'nr.employed': 5191.0
}

# إنشاء عمودين للبيانات
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Personal Information")
    age = st.slider("Age", 18, 100, 35)
    job = st.selectbox("Job", jobs)
    marital_status = st.selectbox("Marital Status", marital)
    education_level = st.selectbox("Education Level", education)
    default = st.selectbox("Credit Default", yes_no)
    housing = st.selectbox("Housing Loan", yes_no)
    loan = st.selectbox("Personal Loan", yes_no)

with col2:
    st.subheader("📞 Campaign Information")
    contact = st.selectbox("Contact Type", contact_type)
    month = st.selectbox("Contact Month", months)
    day = st.selectbox("Day of Week", days)
    duration = st.number_input("Call Duration (seconds)", 0, 5000, 200)
    campaign = st.slider("Number of Contacts", 1, 50, 1)
    pdays = st.slider("Days Since Last Contact", 0, 999, 999)
    previous = st.slider("Previous Contacts", 0, 50, 0)
    poutcome = st.selectbox("Previous Outcome", poutcomes)

# زر التنبؤ
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_btn = st.button("🔮 Predict Subscription", use_container_width=True)

# التنبؤ
if predict_btn:
    input_data = pd.DataFrame([{
        'age': age, 'job': job, 'marital': marital_status, 'education': education_level,
        'default': default, 'housing': housing, 'loan': loan, 'contact': contact,
        'month': month, 'day_of_week': day, 'duration': duration,
        'campaign': campaign, 'pdays': pdays, 'previous': previous,
        'poutcome': poutcome, **economic
    }])
    
    with st.spinner("Processing..."):
        try:
            processed = preprocessor.transform(input_data)
            pca_data = pca.transform(processed)
            pred = model.predict(pca_data)[0]
            conf = model.decision_function(pca_data)[0]
            
            st.markdown("---")
            if pred == 1:
                st.success(f"✅ **SUBSCRIBE: YES**\n\nThis client is predicted to subscribe to the term deposit.\n\n📊 **Confidence Score:** {conf:.3f}")
            else:
                st.error(f"❌ **SUBSCRIBE: NO**\n\nThis client is predicted NOT to subscribe to the term deposit.\n\n📊 **Confidence Score:** {conf:.3f}")
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# About section
with st.expander("ℹ️ About This App"):
    st.markdown("""
    ### 🎯 Model Overview
    This application uses a **Support Vector Machine (SVM)** optimized with **Grid Search** and PCA for dimensionality reduction.
    
    ### 📈 Model Performance
    - **Accuracy:** 86.77%
    - **ROC AUC:** 92.73%
    - **F1 Score:** 0.59
    
    ### 📁 Dataset
    - **Source:** Bank Marketing Dataset (UCI)
    - **Records:** 41,188
    - **Features:** 20
    """)