"""Streamlit app - SuperKart sales forecast.

Loads the model committed to the repository (model/best_model.joblib), collects
product and store details from the user, puts them in a DataFrame and shows the
predicted Product_Store_Sales_Total.

Run locally:   streamlit run app.py
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "model" / "best_model.joblib"

st.set_page_config(page_title="SuperKart Sales Forecast", page_icon="🛒")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("🛒 SuperKart Sales Forecast")
st.write("Enter the product and store details to predict the total sales for that product in that store.")

model = load_model()

PRODUCT_TYPES = [
    "Baking Goods", "Breads", "Breakfast", "Canned", "Dairy", "Frozen Foods",
    "Fruits and Vegetables", "Hard Drinks", "Health and Hygiene", "Household",
    "Meat", "Others", "Seafood", "Snack Foods", "Soft Drinks", "Starchy Foods",
]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Product")
    product_family = st.selectbox(
        "Product family", ["FD", "NC", "DR"],
        format_func=lambda x: {"FD": "FD - Food", "NC": "NC - Non-consumable", "DR": "DR - Drink"}[x],
    )
    product_type = st.selectbox("Product type", PRODUCT_TYPES, index=5)
    sugar = st.selectbox("Sugar content", ["Low Sugar", "Regular", "No Sugar"])
    weight = st.number_input("Product weight", 1.0, 30.0, 12.5, 0.1)
    area = st.number_input("Allocated display area (ratio)", 0.001, 0.400, 0.060, 0.001, format="%.3f")
    mrp = st.number_input("Product MRP", 20.0, 300.0, 147.0, 0.5)

with col2:
    st.subheader("Store")
    store_type = st.selectbox(
        "Store type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"]
    )
    store_size = st.selectbox("Store size", ["High", "Medium", "Small"], index=1)
    city = st.selectbox("City tier", ["Tier 1", "Tier 2", "Tier 3"], index=1)
    year = st.number_input("Store establishment year", 1980, 2026, 2009, 1)

if st.button("Predict sales", type="primary"):
    # Save the inputs into a dataframe, using the same columns as training
    input_df = pd.DataFrame(
        [{
            "Product_Weight": weight,
            "Product_Sugar_Content": sugar,
            "Product_Allocated_Area": area,
            "Product_Type": product_type,
            "Product_MRP": mrp,
            "Store_Establishment_Year": int(year),
            "Store_Size": store_size,
            "Store_Location_City_Type": city,
            "Store_Type": store_type,
            "Product_Family": product_family,
        }]
    )
    prediction = float(model.predict(input_df)[0])

    st.success(f"Predicted total sales: **{prediction:,.2f}**")
    with st.expander("Input data sent to the model"):
        st.dataframe(input_df)
