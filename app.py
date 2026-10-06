import streamlit as st
import joblib
import pandas as pd

# Page setup
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")


# @st.cache_resource
def load_all():
    pipeline = joblib.load("full_pipeline.pkl")
    model = joblib.load("random_forest_car_model.pkl")
    options = joblib.load("car_options.pkl")
    
    return pipeline, model, options

full_pipeline, model, options = load_all()

st.title("🚗 Used Car Price Predictor")
st.caption("Machine Learning powered car valuation app")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Vehicle Specifications")
    vehicle_age = st.number_input("Vehicle Age (in years)", min_value=0, max_value=35, value=5, step=1)
    km_driven = st.number_input("Kilometers Driven", min_value=100, max_value=600000, value=45000, step=1000)
    mileage = st.number_input("Mileage (kmpl)", min_value=5.0, max_value=40.0, value=19.5, step=0.5)
    engine = st.number_input("Engine Capacity (CC)", min_value=600, max_value=6000, value=1197, step=50)
    max_power = st.number_input("Max Power (bhp)", min_value=20.0, max_value=700.0, value=82.0, step=1.0)
    seats = st.number_input("Seats", min_value=2, max_value=10, value=5, step=1)

with col2:
    st.subheader("Car Identification & Attributes")
    brand = st.selectbox("Brand", options["brand"])
    car_model = st.text_input("Car Model (e.g. Swift, i20, City)", value="Swift")
    car_name = st.text_input("Full Car Name / Variant", value="Maruti Swift VXI")
    
    seller_type = st.selectbox("Seller Type", options["seller_type"])
    fuel_type = st.selectbox("Fuel Type", options["fuel_type"])
    transmission_type = st.selectbox("Transmission Type", options["transmission_type"])

st.divider()

if st.button("Predict Selling Price", type="primary", use_container_width=True):
    # Exactly aapke notebook wale columns ke names[cite: 2]
    input_data = pd.DataFrame([{
        # Numerical
        "vehicle_age": vehicle_age,
        "km_driven": km_driven,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats,
        # Categorical
        "car_name": car_name,
        "brand": brand,
        "model": car_model,
        "seller_type": seller_type,
        "fuel_type": fuel_type,
        "transmission_type": transmission_type
    }])
    
    try:
        # Preprocessing through ColumnTransformer
        transformed_input = full_pipeline.transform(input_data)
        
        # Predict with Random Forest
        predicted_price = model.predict(transformed_input)[0]
        
        # Display Result
        st.success(f"### Estimated Market Price: ₹ {predicted_price:,.2f}")
        st.balloons()
    except Exception as e:
        st.error(f"Error during prediction: {e}")