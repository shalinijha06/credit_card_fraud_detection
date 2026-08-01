import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# -------------------------------
# Load Model
# -------------------------------
model = joblib.load("fraud_detection_model.pkl")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("💳 Credit Card Fraud Detection")

st.sidebar.markdown("""
### About
This application uses a **Machine Learning Random Forest model**
to detect fraudulent credit card transactions.

### Dataset
- Kaggle Credit Card Fraud Detection Dataset

### Features
- Upload CSV
- Fraud Detection
- Prediction Summary
- Download Results
""")

st.sidebar.markdown("---")
st.sidebar.info("Developed by **Shalini Jha**")

# -------------------------------
# Title
# -------------------------------
st.title("💳 Credit Card Fraud Detection")

st.markdown("""
Detect potentially fraudulent credit card transactions using a trained **Machine Learning** model.

Upload a CSV file containing transaction data (excluding the **Class** column), then click **Analyze Transactions**.
""")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        data = pd.read_csv(uploaded_file)

        # Remove target column if present
        if "Class" in data.columns:
            data = data.drop("Class", axis=1)

        st.markdown("---")

        st.subheader("📊 Dataset Overview")

        col1, col2, col3 = st.columns(3)

        col1.metric("Transactions", len(data))
        col2.metric("Features", len(data.columns))
        col3.metric("Missing Values", data.isnull().sum().sum())

        with st.expander("👀 Preview Uploaded Data"):
            st.dataframe(data.head(10), use_container_width=True)

        st.markdown("")

        if st.button("🔍 Analyze Transactions", use_container_width=True):

            with st.spinner("Analyzing transactions..."):

                prediction = model.predict(data)

            result = data.copy()

            result["Prediction"] = prediction

            result["Prediction"] = result["Prediction"].map(
                {
                    0: "Legitimate",
                    1: "Fraud"
                }
            )

            fraud = (result["Prediction"] == "Fraud").sum()
            legit = (result["Prediction"] == "Legitimate").sum()

            st.success("Analysis Completed Successfully!")

            st.markdown("---")

            st.subheader("📈 Prediction Summary")

            c1, c2 = st.columns(2)

            c1.metric(
                "Legitimate Transactions",
                legit
            )

            c2.metric(
                "Fraudulent Transactions",
                fraud
            )

            # -----------------------
            # Bar Chart
            # -----------------------

            chart = result["Prediction"].value_counts()

            st.subheader("📊 Prediction Distribution")

            fig, ax = plt.subplots(figsize=(5,4))
            ax.bar(chart.index, chart.values)
            ax.set_ylabel("Number of Transactions")
            st.pyplot(fig)

            # -----------------------
            # Results Table
            # -----------------------

            st.subheader("📋 Prediction Results")

            st.dataframe(
                result,
                use_container_width=True
            )

            # -----------------------
            # Warning / Success
            # -----------------------

            if fraud > 0:
                st.warning(
                    f"⚠️ {fraud} fraudulent transaction(s) detected."
                )
            else:
                st.success(
                    "✅ No fraudulent transactions detected."
                )

            # -----------------------
            # Download
            # -----------------------

            csv = result.to_csv(index=False).encode("utf-8")

            st.download_button(
                "📥 Download Results",
                csv,
                "fraud_prediction_results.csv",
                "text/csv",
                use_container_width=True
            )

    except Exception as e:

        st.error(f"Error: {e}")

else:

    st.info("👆 Upload a CSV file to begin fraud detection.")

# -------------------------------
# Footer
# -------------------------------

st.markdown("---")

st.caption("© 2026 | Credit Card Fraud Detection | Built with Streamlit & Scikit-Learn")