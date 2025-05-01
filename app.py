import streamlit as st
import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from io import BytesIO
from streamlit_autorefresh import st_autorefresh

# Load environment variables
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")
db_name = os.getenv("DB_NAME")
collection_name = os.getenv("COLLECTION_NAME")
groq_api_key = os.getenv("GROQ_API_KEY")

# MongoDB connection
def connect_mongo():
    client = MongoClient(mongo_uri)
    db = client[db_name]
    return db[collection_name]

# Fetch data from MongoDB
def get_data(collection):
    df = pd.DataFrame(list(collection.find()))
    if '_id' in df.columns:
        df.drop(columns=['_id'], inplace=True)
    return df

# Train the regression model
def train_model(X, y):
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    return model

# Generate AI Report using LangChain + Groq
def generate_report(feature_impact, predicted_wqi, location, timestamp, selected):
    param_info = "\n".join([f"- {param}: {selected[param]}" for param in feature_impact.keys()])

    prompt = PromptTemplate.from_template(
        """You are an expert environmental analyst.

The predicted Water Quality Index (WQI) is {predicted_wqi} at location \"{location}\" on {timestamp}.
The top contributing parameters with their actual sensor values are:
{param_info}

Write a report that includes:
1. Likely causes for this WQI
2. Why these parameters are significant
3. Practical recommendations to improve WQI"""
    )

    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
    chain = LLMChain(llm=llm, prompt=prompt)

    report = chain.run(
        predicted_wqi=predicted_wqi,
        location=location,
        timestamp=timestamp,
        param_info=param_info
    )

    report_cleaned = report.replace("**", "")
    return report_cleaned

# Function to save report as TXT
def save_report_as_txt(text: str, filename: str) -> BytesIO:
    buffer = BytesIO()
    buffer.write(text.encode("utf-8"))
    buffer.seek(0)
    return buffer

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Water Quality AI Analyzer", layout="wide")
st.title("💧 Water Quality Index Prediction & AI-Powered Report")

# Add auto-refresh using Streamlit timer
st_autorefresh(interval=60 * 1000, key="datarefresh")
st.markdown("⏰ Auto-refreshing every 60 seconds to fetch latest data...")

# Real-time data load from MongoDB
collection = connect_mongo()
df = get_data(collection)

if df.empty:
    st.warning("No data found in MongoDB.")
    st.stop()

st.success("✅ Data successfully loaded from MongoDB")
st.dataframe(df.head())

# Define features and target
feature_cols = ['pH', 'turbidity', 'dissolved_oxygen', 'conductivity', 'temperature']
target_col = 'wqi'

if not all(col in df.columns for col in feature_cols + [target_col]):
    st.error("❌ Required columns are missing from the dataset.")
    st.stop()

# Train model
X = df[feature_cols]
y = df[target_col]
model = train_model(X, y)

# SHAP Explainer
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# Display SHAP feature importance with smaller size
st.subheader("📊 Feature Impact on WQI (SHAP Values)")
fig, ax = plt.subplots(figsize=(6, 4))
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
st.pyplot(fig)

# Select record
st.subheader("🔍 Select a Data Record for Detailed Analysis")
record_options = [f"{i}: {row.get('location', 'Unknown')} @ {row.get('timestamp', 'N/A')}" for i, row in df.iterrows()]
selected_label = st.selectbox("📋 Select a Record by Location & Time", options=record_options)
selected_index = int(selected_label.split(":")[0])
selected = df.iloc[selected_index]

# Show selected record details
st.markdown(f"🔢 Selected Index: `{selected_index}`")
st.markdown(f"📍 Location: `{selected.get('location', 'N/A')}`")
st.markdown(f"⏰ Timestamp: `{selected.get('timestamp', 'N/A')}`")

input_data = selected[feature_cols].to_frame().T
predicted_wqi = model.predict(input_data)[0]

# Display chosen parameter values
st.markdown("### 🧪 Selected Sensor Parameters Used for WQI Prediction")
for param in feature_cols:
    st.markdown(f"- **{param}**: `{selected[param]}`")

# SHAP for selected row
individual_shap = explainer(input_data)
impact = pd.Series(individual_shap.values[0], index=feature_cols).abs().sort_values(ascending=False)
top_impact = impact.head(3).to_dict()

# Show prediction
st.markdown(f"### 🤖 Predicted WQI: `{predicted_wqi:.2f}`")

# Generate AI report and download
if st.button("📝 Generate AI Report"):
    location = selected.get("location", "Unknown")
    timestamp = selected.get("timestamp", "Unknown")
    report = generate_report(top_impact, predicted_wqi, location, timestamp, selected)

    st.subheader("📝 AI-Generated Water Quality Report")
    st.markdown(report)

    # Save as TXT
    txt_file_name = f"water_quality_report_{location.replace(' ', '_')}_{timestamp[:10]}.txt"
    report_txt = save_report_as_txt(report, txt_file_name)

    st.download_button(
        label="📄 Download Report (TXT)",
        data=report_txt,
        file_name=txt_file_name,
        mime="text/plain"
    )
