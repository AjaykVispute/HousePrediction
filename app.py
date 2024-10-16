import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Step 1: Load the California Housing dataset
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Step 2: Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Train a RandomForestRegressor (using a simple model here for demonstration)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_scaled, y)

# Step 4: Define Streamlit App layout
st.title('California Housing Price Prediction')

st.write("""
### Enter the features below to predict the median house price (in $100,000s):
""")

# Step 5: Create input sliders for features
longitude = st.slider('Longitude', float(X['Longitude'].min()), float(X['Longitude'].max()), float(X['Longitude'].mean()))
latitude = st.slider('Latitude', float(X['Latitude'].min()), float(X['Latitude'].max()), float(X['Latitude'].mean()))
housing_median_age = st.slider('Housing Median Age', float(X['HouseAge'].min()), float(X['HouseAge'].max()), float(X['HouseAge'].mean()))  # Correct column name
total_rooms = st.slider('Total Rooms', float(X['AveRooms'].min()), float(X['AveRooms'].max()), float(X['AveRooms'].mean()))
total_bedrooms = st.slider('Total Bedrooms', float(X['AveBedrms'].min()), float(X['AveBedrms'].max()), float(X['AveBedrms'].mean()))
population = st.slider('Population', float(X['Population'].min()), float(X['Population'].max()), float(X['Population'].mean()))
households = st.slider('Households', float(X['AveOccup'].min()), float(X['AveOccup'].max()), float(X['AveOccup'].mean()))
median_income = st.slider('Median Income', float(X['MedInc'].min()), float(X['MedInc'].max()), float(X['MedInc'].mean()))

# Step 6: Store inputs into a dataframe for prediction with column names
input_data = pd.DataFrame([[longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income]],
                          columns=X.columns)

# Step 7: Scale input data
input_data_scaled = scaler.transform(input_data)

# Step 8: Make the prediction
prediction = rf_model.predict(input_data_scaled)

# Step 9: Display the predicted price
st.write(f"### Predicted Median House Price: ${prediction[0] * 100000:,.2f}")

# Optional: Add a feature importance plot
if st.checkbox('Show Feature Importance'):
    importances = rf_model.feature_importances_
    features = data.feature_names
    feature_df = pd.DataFrame({'Features': features, 'Importance': importances})
    feature_df = feature_df.sort_values(by='Importance', ascending=False)
    
    st.bar_chart(feature_df.set_index('Features'))
