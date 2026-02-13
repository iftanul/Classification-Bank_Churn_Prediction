import streamlit as st
import pandas as pd 
import numpy as np 
import joblib
import os

# =========================================================================
# 1. PAGE CONFIG & STYLING (DARK MODE)
# =========================================================================
st.set_page_config(
    page_title="Bank Churn Prediction", 
    page_icon="🏦", 
    layout='wide',
    initial_sidebar_state="collapsed"
)

# Custom CSS untuk Dark Mode dan Tombol Hijau
st.markdown("""
    <style>
    /* Mengembalikan background ke mode gelap (Dark Mode) */
    .stApp { 
        background-color: #0E1117; 
        color: #FAFAFA;
    }
    
    /* Warna angka metrik */
    div[data-testid="stMetricValue"] { 
        color: #9EE05B; /* Hijau terang */
    }
    
    /* Mempercantik tombol prediksi (Warna hijau persis seperti screenshot) */
    div[data-testid="stFormSubmitButton"] > button {
        width: 100%;
        background-color: #9EE05B;
        color: #1e1e1e;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
        border: none;
        transition: 0.3s;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #88c94a;
        color: #1e1e1e;
    }
    
    /* Memastikan teks terbaca dengan baik */
    h1, h2, h3, h4, h5, h6, p, label {
        color: #FAFAFA !important;
    }
    </style>
    """, unsafe_allow_html=True)

# =========================================================================
# 2. CORE FUNCTIONS (Mesin Asli Kamu)
# =========================================================================
def perform_feature_engineering(df):
    df_pipeline = df.copy()

    # 1. age_group
    conditions_age = [
        (df_pipeline['customer_age'] < 30),
        (df_pipeline['customer_age'] >= 30) & (df_pipeline['customer_age'] < 45)
    ]
    choices_age = ['Young', 'Adult']
    df_pipeline['age_group'] = np.select(conditions_age, choices_age, default='Senior')

    # 2. tenure_segment
    conditions_tenure = [
        (df_pipeline['months_on_book'] < 24),
        (df_pipeline['months_on_book'] >= 24) & (df_pipeline['months_on_book'] < 48)
    ]
    choices_tenure = ['New', 'Mid']
    df_pipeline['tenure_segment'] = np.select(conditions_tenure, choices_tenure, default='Long')

    # 3. trans_per_month & amt_per_month
    df_pipeline['trans_per_month'] = df_pipeline['total_trans_ct'] / (df_pipeline['months_on_book'] + 1)
    df_pipeline['amt_per_month'] = df_pipeline['total_trans_amt'] / (df_pipeline['months_on_book'] + 1)

    # 4. utilization_status
    conditions_util = [
        (df_pipeline['avg_utilization_ratio'] < 0.3),
        (df_pipeline['avg_utilization_ratio'] >= 0.3) & (df_pipeline['avg_utilization_ratio'] < 0.7)
    ]
    choices_util = ['Low', 'Medium']
    df_pipeline['utilization_status'] = np.select(conditions_util, choices_util, default='High')

    # 5. revolving_ratio, inactive_ratio, util_gap, product_per_year
    df_pipeline['revolving_ratio'] = df_pipeline['total_revolving_bal'] / (df_pipeline['credit_limit'] + 1)
    df_pipeline['inactive_ratio'] = df_pipeline['months_inactive_12_mon'] / (df_pipeline['months_on_book'] + 1)
    df_pipeline['util_gap'] = 1 - df_pipeline['avg_utilization_ratio']
    df_pipeline['product_per_year'] = df_pipeline['total_relationship_count'] / ((df_pipeline['months_on_book'] / 12) + 1)

    return df_pipeline

# =========================================================================
# 3. LOAD MODEL (Path Asli Kamu + Caching)
# =========================================================================
@st.cache_resource
def load_pipeline():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Path sesuai dengan struktur aslimu
    model_path = os.path.join(BASE_DIR, '..', 'models', 'bank_churn_pipeline.pkl')
    try:
        return joblib.load(model_path), "Success"
    except Exception as e:
        return None, str(e)

pipeline, load_status = load_pipeline()

# =========================================================================
# 4. HEADER & NAVIGATION
# =========================================================================
col_title, col_nav = st.columns([5, 1])
with col_title:
    st.title("🏦 Bank Customer Churn Prediction")
    st.caption("AI-Powered Retention Strategy Tool")
with col_nav:
    st.write("") # Spasi
    st.link_button("⬅️ Back to Portfolio", "https://iftanulibnu.streamlit.app/Projects", use_container_width=True)

st.write("---")

# =========================================================================
# 5. TABS INTERFACE
# =========================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "💼 Business Case", 
    "📊 EDA & Insights", 
    "🤖 Model Performance", 
    "🔮 Live Simulator"
])

# -------------------------------------------------------------------------
# TAB 1: BUSINESS CASE
# -------------------------------------------------------------------------
with tab1:
    st.markdown("### 📌 Executive Summary")
    
    col_fact1, col_fact2, col_fact3 = st.columns(3)
    with col_fact1:
        st.container(border=True)
        st.metric(label="Acquisition Cost", value="5x Higher", delta="vs Retention", delta_color="inverse")
        st.caption("Biaya mencari nasabah baru jauh lebih mahal daripada mempertahankan yang lama.")
    with col_fact2:
        st.container(border=True)
        st.metric(label="Profit Impact", value="+25% to 95%", delta="with 5% Churn Drop")
        st.caption("Menurunkan churn 5% saja bisa melipatgandakan keuntungan.")
    with col_fact3:
        st.container(border=True)
        st.metric(label="Current Strategy", value="Reactive", delta="No AI Warning", delta_color="inverse")
        st.caption("Tanpa sistem AI, intervensi sering terlambat dilakukan.")

    st.write("---")
    
    col_prob, col_obj = st.columns(2, gap="large")
    with col_prob:
        st.error("🎯 Problem Statement")
        st.markdown("""
        **Bank saat ini menghadapi kendala:**
        * ❌ **Blind Spot:** Tidak ada sistem untuk mengidentifikasi nasabah berisiko churn.
        * ❌ **Late Detection:** Penurunan engagement baru disadari setelah nasabah menutup akun.
        * ❌ **Inefficient Budget:** Biaya marketing terbuang untuk retensi yang tidak tepat sasaran.
        """)
    with col_obj:
        st.success("🎯 Project Objective")
        st.markdown("""
        **Membangun Machine Learning untuk:**
        * ✅ **Predictive:** Menghitung probabilitas churn tiap nasabah secara presisi.
        * ✅ **Early Signal:** Mendeteksi pola perilaku (transaksi menurun, inaktif) lebih dini.
        * ✅ **Actionable:** Memberikan daftar prioritas nasabah untuk tim marketing.
        """)

    st.write("---")
    st.subheader("💰 Business Impact Simulation")
    
    with st.container(border=True):
        c_calc1, c_calc2, c_calc3 = st.columns(3)
        with c_calc1:
            total_cust = st.number_input("Total Nasabah Berisiko", value=1000, step=100)
        with c_calc2:
            avg_clv = st.number_input("Rata-rata Customer Value ($)", value=5000, step=500)
        with c_calc3:
            model_success = st.slider("Efektivitas Retensi (%)", 10, 50, 20)
        
        saved_revenue = total_cust * avg_clv * (model_success / 100)
        st.markdown(f"### 💵 Potential Revenue Saved: <span style='color:#9EE05B'>${saved_revenue:,.0f}</span>", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# TAB 2: EDA & INSIGHTS
# -------------------------------------------------------------------------
with tab2:
    st.markdown("### 🔎 Key Data Insights")
    st.write("Analisis mengungkap bahwa **Churn terjadi bukan karena 'Siapa Nasabahnya', melainkan 'Bagaimana Perilakunya'.**")
    st.write("---")

    st.subheader("1️⃣ Demographics vs Behavior (The Power of Activity)")
    col_f1_text, col_f1_viz = st.columns([1, 1.5])
    with col_f1_text:
        st.info("Variabel demografis (Gender, Education, Marital) memiliki Predictive Power yang rendah. Sebaliknya, variabel perilaku (Transaksi) menjadi prediktor utama.")
    with col_f1_viz:
        eda_importance = pd.DataFrame({
            'Feature': ['Total Trans Count', 'Trans Ct Change Q4-Q1', 'Total Revolving Bal', 'Contacts 12mon', 'Gender', 'Income', 'Education'],
            'Predictive Power': [0.85, 0.78, 0.70, 0.60, 0.15, 0.12, 0.05]
        }).sort_values('Predictive Power', ascending=True)
        st.bar_chart(eda_importance.set_index('Feature'), color='#9EE05B')

    st.divider()

    st.subheader("2️⃣ Transaction Behavior & Friction Signals")
    c_sig1, c_sig2 = st.columns(2)
    with c_sig1:
        st.markdown("#### 📉 Declining Activity")
        st.write("Nasabah Churn memiliki Total Transaksi jauh lebih rendah dibanding nasabah loyal.")
        st.bar_chart(pd.DataFrame({'Loyal': [70], 'Churn': [35]}).T, color=['#9EE05B'])
    with c_sig2:
        st.markdown("#### 💳 Low Credit Utilization")
        st.write("Nasabah Churn memiliki saldo berjalan sangat rendah (Minim penggunaan produk).")
        st.progress(15, text="Rata-rata Utilisasi Nasabah Churn (Sangat Rendah)")

# -------------------------------------------------------------------------
# TAB 3: MODEL PERFORMANCE
# -------------------------------------------------------------------------
with tab3:
    st.markdown("### 🤖 Champion Model: XGBoost Classifier")
    st.write("Model dievaluasi menggunakan Test Set terpisah untuk menjamin validitas prediksi.")
    
    c_m1, c_m2, c_m3, c_m4, c_m5 = st.columns(5)
    c_m1.metric("Accuracy", "94%", "Overall")
    c_m2.metric("Precision", "85%", "Churn")
    c_m3.metric("Recall", "77%", "Target") 
    c_m4.metric("F1 Score", "81%", "Balance")
    c_m5.metric("ROC AUC", "0.96", "Excellent")

    st.divider()

    col_matrix, col_tradeoff = st.columns([1.5, 1], gap="large")
    with col_matrix:
        st.subheader("📉 Confusion Matrix Insight")
        st.write("Dari total **325** nasabah yang sebenarnya Churn di data test:")
        m1, m2 = st.columns(2)
        with m1:
            st.success("✅ **251 Terdeteksi**\n\nTrue Positive (Capture Rate: 77%)")
        with m2:
            st.error("⚠️ **74 Terlewat**\n\nFalse Negative (Miss Rate: 23%)")

    with col_tradeoff:
        st.subheader("🎯 Why Recall Matters?")
        st.markdown("""
        **False Negative (Gagal Deteksi)** = Kehilangan Revenue Permanen 💸
        
        **False Positive (Salah Deteksi)** = Biaya Promo Tambahan (Risk Kecil)
        
        **Strategi:** Meminimalkan *False Negative* agar tidak kehilangan nasabah berharga.
        """)

# -------------------------------------------------------------------------
# TAB 4: LIVE SIMULATOR (Inti Mesin Kamu)
# -------------------------------------------------------------------------
with tab4:
    st.write("### 🔮 Prediction Simulator")
    st.write("Uji coba model secara real-time. Masukkan data profil nasabah di bawah ini.")
    
    if pipeline is None:
        st.error(f"🚨 Gagal memuat model. Error: {load_status}")
    else:
        # Menggunakan form agar rapi dan tidak auto-refresh
        with st.form("prediction_form"):
            col_in1, col_in2, col_in3 = st.columns(3)
            
            # --- INPUT ASLI KAMU (DIKELOMPOKKAN DENGAN RAPI) ---
            with col_in1:
                st.markdown("**👤 Demographics**")
                age = st.number_input("Customer Age", min_value=18, max_value=100, value=45)
                gender = st.selectbox("Gender", ["M", "F"])
                dependents = st.number_input("Dependent Count", 0, 10, 2)
                education = st.selectbox("Education Level", ['Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate'])
                marital = st.selectbox("Marital Status", ['Divorced', 'Married', 'Single', 'Unknown'])
                income = st.selectbox("Income Category", ['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +'])
                
            with col_in2:
                st.markdown("**💳 Account Details**")
                card = st.selectbox("Card Category", ['Blue', 'Silver', 'Gold', 'Platinum'])
                months_on_book = st.number_input("Months on Book", 0, 120, 36)
                rel_count = st.number_input("Total Relationship Count", 1, 10, 3)
                inactive = st.number_input("Months Inactive (12 mon)", 0, 12, 1)
                contacts = st.number_input("Contacts Count (12 mon)", 0, 12, 2)
                credit_limit = st.number_input("Credit Limit", 0, 50000, 10000)
                
            with col_in3:
                st.markdown("**💰 Transaction Behavior**")
                revolving = st.number_input("Total Revolving Bal", 0, 5000, 1000)
                amt_chng = st.number_input("Total Amt Chng Q4 Q1", 0.0, 5.0, 0.7)
                ct_chng = st.number_input("Total Ct Chng Q4 Q1", 0.0, 5.0, 0.7)
                trans_amt = st.number_input("Total Trans Amt", 0, 20000, 4000)
                trans_ct = st.number_input("Total Trans Ct", 0, 200, 60)
                util_ratio = st.slider("Avg Utilization Ratio", 0.0, 1.0, 0.2)
                
            st.write("")
            submit_btn = st.form_submit_button("🔍 Analisis Risiko Churn")

        # --- LOGIKA PREDIKSI ASLI KAMU ---
        if submit_btn:
            # Dataframe input sama persis dengan yang kamu buat
            input_df = pd.DataFrame({
                'customer_age': [age],
                'gender': [gender],
                'dependent_count': [dependents],
                'education_level': [education],
                'marital_status': [marital],
                'income_category': [income],
                'card_category': [card],
                'months_on_book': [months_on_book],
                'total_relationship_count': [rel_count],
                'months_inactive_12_mon': [inactive],
                'contacts_count_12_mon': [contacts],
                'credit_limit': [credit_limit],
                'total_revolving_bal': [revolving],
                'total_amt_chng_q4_q1': [amt_chng],
                'total_trans_amt': [trans_amt],
                'total_trans_ct': [trans_ct],
                'total_ct_chng_q4_q1': [ct_chng],
                'avg_utilization_ratio': [util_ratio]
            })

            try:
                # Prediksi menggunakan pipeline kamu
                risk_score = pipeline.predict_proba(input_df)[0][1]
                risk_percent = risk_score * 100

                # Tampilkan Hasil Asli Kamu
                st.divider()
                st.subheader("🎯 Hasil Prediksi AI")
                
                col_res1, col_res2 = st.columns([1, 2])
                with col_res1:
                    # Logika warna dan pesan persis seperti kodemu
                    if risk_percent > 70:
                        st.error(f"🔴 **Risiko Tinggi!** Probabilitas Churn: {risk_percent:.2f}%")
                    elif risk_percent > 30:
                        st.warning(f"🟡 **Risiko Sedang.** Probabilitas Churn: {risk_percent:.2f}%")
                    else:
                        st.success(f"🟢 **Risiko Rendah.** Probabilitas Churn: {risk_percent:.2f}%")
                
                with col_res2:
                    st.write("**Visualisasi Tingkat Risiko:**")
                    # Progress bar asli kamu
                    progress_value = float(np.clip(risk_score, 0.0, 1.0))
                    st.progress(progress_value)
                    
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat memproses prediksi: {e}")