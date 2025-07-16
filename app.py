import streamlit as st
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load('fraud_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("💳 Credit Card Fraud Detection App")
st.write("Upload transaction data, select time frame, and detect frauds.")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV File", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success(f"Uploaded {uploaded_file.name} successfully!")
    st.write("Sample of Uploaded Data:")
    st.dataframe(data.head())

    if 'Time' in data.columns:
        min_time = int(data['Time'].min())
        max_time = int(data['Time'].max())

        st.sidebar.write("### Select Time Frame (seconds)")
        start_time = st.sidebar.slider("Start Time", min_value=min_time, max_value=max_time, value=min_time)
        end_time = st.sidebar.slider("End Time", min_value=min_time, max_value=max_time, value=max_time)

        if start_time > end_time:
            st.error("Start time must be less than or equal to end time.")
        else:
            if st.button("Detect Frauds"):
                filtered_data = data[(data['Time'] >= start_time) & (data['Time'] <= end_time)].copy()

                if filtered_data.empty:
                    st.warning("No transactions in the selected time frame.")
                else:
                    st.write(f"Transactions in selected time frame: {len(filtered_data)}")

                    # Preprocessing
                    filtered_data['Amount'] = scaler.transform(filtered_data[['Amount']])
                    X_filtered = filtered_data.drop(['Time', 'Class'], axis=1, errors='ignore')

                    predictions = model.predict(X_filtered)
                    filtered_data['Prediction'] = predictions

                    frauds = filtered_data[filtered_data['Prediction'] == 1]
                    total_frauds = len(frauds)
                    total_fraud_amount = frauds['Amount'].sum()

                    st.success(f"🚨 Total Frauds Detected: {total_frauds}")
                    st.success(f"💰 Total Fraudulent Amount: {total_fraud_amount:.2f}")

                    if not frauds.empty:
                        st.write("📝 Fraud Transactions Details:")
                        st.dataframe(frauds[['Time', 'Amount']].reset_index(drop=True))
    else:
        st.error("Uploaded CSV does not contain a 'Time' column.")
