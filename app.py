import streamlit as st
import pandas as pd
import joblib

# Load trained model (trained without scaler)
model = joblib.load('fraud_detection_model.pkl')

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection App")
st.write("""
Upload credit card transaction data, select the time frame, and detect fraudulent transactions using a machine learning model.
""")

# CSV Upload
uploaded_file = st.file_uploader("Upload CSV File (Kaggle Credit Card Data Format)", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success(f"Uploaded `{uploaded_file.name}` successfully!")
    st.subheader("Sample of Uploaded Data")
    st.dataframe(data.head())

    if 'Time' in data.columns:
        min_time = int(data['Time'].min())
        max_time = int(data['Time'].max())

        st.sidebar.header("⏱️ Select Time Frame (in seconds)")
        start_time = st.sidebar.slider("Start Time", min_value=min_time, max_value=max_time, value=min_time)
        end_time = st.sidebar.slider("End Time", min_value=min_time, max_value=max_time, value=max_time)

        if start_time > end_time:
            st.sidebar.error("⚠️ Start time must be less than or equal to end time.")
        else:
            if st.sidebar.button("🚨 Detect Frauds"):
                filtered_data = data[(data['Time'] >= start_time) & (data['Time'] <= end_time)].copy()

                if filtered_data.empty:
                    st.warning("No transactions found in the selected time frame.")
                else:
                    st.subheader(f"Transactions in Selected Time Frame: {len(filtered_data)}")
                    X_filtered = filtered_data.drop(['Time', 'Class'], axis=1, errors='ignore')

                    # Predict frauds
                    predictions = model.predict(X_filtered)
                    filtered_data['Prediction'] = predictions

                    # Extract detected frauds
                    frauds = filtered_data[filtered_data['Prediction'] == 1]
                    total_frauds = len(frauds)
                    total_fraud_amount = frauds['Amount'].sum()

                    st.success(f"🚩 Total Frauds Detected: {total_frauds}")
                    st.success(f"💰 Total Fraudulent Amount: {total_fraud_amount:.2f}")

                    if not frauds.empty:
                        st.subheader("📝 Detected Fraud Transactions")
                        st.dataframe(frauds[['Time', 'Amount']].reset_index(drop=True))

                        # Download Button
                        csv = frauds.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Fraud Transactions CSV",
                            data=csv,
                            file_name='fraud_transactions.csv',
                            mime='text/csv',
                        )
    else:
        st.error("❗ Uploaded CSV does not contain a 'Time' column. Please upload the correct dataset format.")
