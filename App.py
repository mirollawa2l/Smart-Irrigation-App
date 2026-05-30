import streamlit as st
import pandas as pd
import joblib
import requests

# 1. Page Configuration
st.set_page_config(page_title="Irrigation Recommender", page_icon="🌾", layout="wide")

# Mapping to decode XGBoost predictions
INVERSE_TARGET_MAPPING = {0: "Low", 1: "Medium", 2: "High"}

# 2. Load the trained model
@st.cache_resource
def load_model():
    try:
        return joblib.load('xgboost_irrigation_model.joblib')
    except Exception as e:
        st.error("Model not found. Please run train_model.py first!")
        st.stop()

model = load_model()

# Sidebar Context
st.sidebar.title("🌾 About the App")
st.sidebar.info(
    "This utility tool assists farmers in determining the irrigation needs "
    "of their fields based on soil characteristics, crop stage, and environmental conditions.\n\n"
    "**Model:** Extreme Gradient Boosting (XGBoost)\n"
    "**Accuracy Target:** ~90% Balanced Accuracy"
)

st.title("Smart Irrigation Utility")
st.markdown("Get real-time insights on your field's water requirements.")
st.markdown("---")

# 3. Create Tabs
tab1, tab2 = st.tabs(["💧 Single Field Prediction", "📂 Batch CSV Upload"])

with tab1:
    # --- LIVE WEATHER INTEGRATION ---
    st.subheader("⛅ Live Weather Integration")
    st.markdown("Select a city to automatically fill in the current environmental data.")
    
    city = st.selectbox("Select a City", ["None", "Alexandria", "Cairo", "Luxor"])
    
    live_temp, live_wind, live_humidity = 25.0, 10.0, 50.0
    
    if city != "None":
        coords = {"Alexandria": (31.2, 29.9), "Cairo": (30.0, 31.2), "Luxor": (25.7, 32.6)}
        lat, lon = coords[city]
        try:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
            res = requests.get(url).json()
            live_temp = res['current']['temperature_2m']
            live_humidity = res['current']['relative_humidity_2m']
            live_wind = res['current']['wind_speed_10m']
            st.success(f"✅ Fetched live weather for {city}: Temp {live_temp}°C, Humidity {live_humidity}%, Wind {live_wind} km/h")
        except:
            st.warning("Could not fetch live weather. Please enter values manually.")

    st.subheader("🌱 Field Details")
    
    # 4. User Inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Soil Characteristics**")
        soil_type = st.selectbox("Soil Type", ["Loamy", "Clay", "Sandy", "Silty", "Peaty", "Chalky"])
        soil_ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
        soil_moisture = st.number_input("Soil Moisture (%)", min_value=0.0, max_value=100.0, value=30.0)
        organic_carbon = st.number_input("Organic Carbon (%)", min_value=0.0, max_value=5.0, value=1.0)
        ec = st.number_input("Electrical Conductivity", min_value=0.0, max_value=10.0, value=1.5)
        
    with col2:
        st.markdown("**Environmental Factors**")
        temp = st.number_input("Temperature (°C)", min_value=-20.0, max_value=60.0, value=float(live_temp))
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=float(live_humidity))
        wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=200.0, value=float(live_wind))
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=100.0)
        sunlight = st.number_input("Sunlight Hours", min_value=0.0, max_value=24.0, value=7.0)
        
    with col3:
        st.markdown("**Crop & Field Info**")
        crop_type = st.selectbox("Crop Type", ["Sugarcane", "Wheat", "Rice", "Cotton", "Maize", "Soybeans"])
        crop_stage = st.selectbox("Crop Growth Stage", ["Sowing", "Vegetative", "Flowering", "Harvesting"])
        season = st.selectbox("Season", ["Kharif", "Rabi", "Zaid"])
        irrigation_type = st.selectbox("Irrigation Type", ["Drip", "Rainfed", "Sprinkler", "Canal"])
        water_source = st.selectbox("Water Source", ["Rainwater", "River", "Reservoir", "Well", "Canal"])
        mulching = st.selectbox("Mulching Used", ["Yes", "No"])
        region = st.selectbox("Region", ["East", "South", "North", "West"])
        field_area = st.number_input("Field Area (hectares)", min_value=0.1, value=5.0)
        prev_irrigation = st.number_input("Previous Irrigation (mm)", min_value=0.0, value=50.0)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # 5. Execution
    if st.button("🔄 Get Recommendation", type="primary", use_container_width=True):
        input_dict = {
            "Soil_Type": [soil_type], "Soil_pH": [soil_ph], "Soil_Moisture": [soil_moisture],
            "Organic_Carbon": [organic_carbon], "Electrical_Conductivity": [ec], 
            "Temperature_C": [temp], "Humidity": [humidity], "Rainfall_mm": [rainfall],
            "Sunlight_Hours": [sunlight], "Wind_Speed_kmh": [wind_speed],
            "Crop_Type": [crop_type], "Crop_Growth_Stage": [crop_stage], "Season": [season],
            "Irrigation_Type": [irrigation_type], "Water_Source": [water_source],
            "Field_Area_hectare": [field_area], "Mulching_Used": [mulching],
            "Previous_Irrigation_mm": [prev_irrigation], "Region": [region]
        }
        
        input_df = pd.DataFrame(input_dict)
        
        # Predict and decode from numeric back to string
        numeric_pred = model.predict(input_df)[0]
        prediction = INVERSE_TARGET_MAPPING[numeric_pred]
        
        st.markdown("---")
        if prediction == "Low":
            st.success("### 🟢 Recommendation: LOW Irrigation Need")
            st.write("Conditions are optimal. Avoid over-watering the field.")
        elif prediction == "Medium":
            st.warning("### 🟡 Recommendation: MEDIUM Irrigation Need")
            st.write("Monitor the soil moisture closely. A moderate irrigation cycle is recommended.")
        else:
            st.error("### 🔴 Recommendation: HIGH Irrigation Need")
            st.write("Field is highly stressed. Immediate water supply is required.")

# 6. Batch Upload Feature
with tab2:
    st.subheader("📂 Batch Prediction from CSV")
    st.write("Upload a CSV file containing rows of field data.")
    
    uploaded_file = st.file_uploader("Upload your field data CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            
            if "id" in batch_df.columns:
                batch_df = batch_df.drop(columns=["id"])
            if "Irrigation_Need" in batch_df.columns:
                batch_df = batch_df.drop(columns=["Irrigation_Need"])
                
            with st.spinner("Processing predictions..."):
                preds = model.predict(batch_df)
                # Decode all predictions in the batch
                batch_df['Predicted_Irrigation_Need'] = [INVERSE_TARGET_MAPPING[p] for p in preds]
            
            st.success("✅ Batch processing complete!")
            st.dataframe(batch_df.head(10)) 
            
            csv_data = batch_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download Recommendation Report",
                data=csv_data,
                file_name='xgboost_batch_recommendations.csv',
                mime='text/csv',
                type="primary"
            )
        except Exception as e:
            st.error(f"Error processing CSV. Please ensure columns match the training format. \n\n Details: {e}")