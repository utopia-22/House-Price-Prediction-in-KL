import streamlit as st
import joblib	
import numpy as np
import pandas as pd
import gdown
import os
from sklearn.preprocessing import LabelEncoder

# Load trained model and scaler
# model = joblib.load("model.pkl")  

# Google Drive File ID (Replace with your actual ID)
file_id = "1ah68ZFFL1frG8uAsREeFOfoqCL-GutLl"
output = "model.pkl" 

# Download the file
gdrive_url = "https://drive.google.com/uc?id={file_id}"

# Check if model exists
if not os.path.exists(output):
    gdown.download(gdrive_url, output, quiet=False)
else:
    pass

# Load the model using Joblib
model = joblib.load(output)

# Load your dataset (Ensure df1 is already loaded)
df = pd.read_csv("original_data.csv")
df1 = pd.read_csv("cleaned_data.csv")  

# Initialize Label Encoder
label_encoder = LabelEncoder()
df["Property Type Encoded"] = label_encoder.fit_transform(df["Property Type"])

# Create a mapping of categories to their encoded values
property_type_mapping = dict(zip(df["Property Type"], df["Property Type Encoded"]))


st.title("House Price Prediction App")

st.divider()

st.write("This app uses machine learning for predicting house price")

st.divider()

rooms = st.number_input("Rooms", min_value=1, value=3)
bathrooms = st.number_input("Bathrooms", min_value=1, value=2)
car_parks = st.number_input("Car Parks", min_value=0, value=1)
selected_property_type = st.selectbox("Property Type", sorted(property_type_mapping.keys()) )
size = st.slider("Size (sq. ft.)", min_value=400, max_value=5000, value=(1000, 3000))
st.write(f"Selected size range: {size[0]} - {size[1]} sq. ft.")


# Furnishing options (One-Hot Encoding)
furnishing_options = ["Fully Furnished", "Partly Furnished", "Unfurnished"]
furnishing = st.selectbox("Furnishing", furnishing_options)

# Convert furnishing to one-hot encoding
furnishing_fully = 1 if furnishing == "Fully Furnished" else 0
furnishing_partly = 1 if furnishing == "Partly Furnished" else 0
furnishing_unfurnished = 1 if furnishing == "Unfurnished" else 0

# Convert selected category to its encoded value
property_type_encoded = property_type_mapping[selected_property_type]

# Compute Derived Features (Hidden from User)
property_type_furnishing = f"{selected_property_type}_{furnishing}"
property_type_furnishing_encoded = df1["PropertyType_Furnishing"].map(df1["PropertyType_Furnishing"].value_counts()).get(property_type_furnishing, 0)

# Compute Price per Sq.Ft. (Hidden from User)
price_per_sqft = df1.set_index("PropertyType_Furnishing")["Price_per_sqft"].get(property_type_furnishing, 0)

# Apply custom CSS to make the dropdown cursor a hand (pointer)
st.markdown(
    """
    <style>
        /* Change cursor to pointer when hovering over selectbox */
        .stSelectbox div[data-baseweb="select"] > div { 
            cursor: pointer !important; 
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.divider()

predictbutton = st.button("Predict!")


if predictbutton:
    
	st.balloons()
    
	# Combine into a single input array
	user_input = np.array([[rooms, bathrooms, car_parks, property_type_encoded, size,
                 furnishing_fully, furnishing_partly, furnishing_unfurnished]])

	user_input_dict = {
    "Rooms": [rooms],
    "Bathrooms": [bathrooms],
    "Car Parks": [car_parks],
    "Price_per_sqft": [price_per_sqft],
    "Property Type": [property_type_encoded],
    "Size (sq. ft.)": [size],
    "Furnishing_Fully Furnished": [1 if furnishing_fully else 0],  
    "Furnishing_Partly Furnished": [1 if furnishing_partly else 0],  
    "Furnishing_Unfurnished": [1 if furnishing_unfurnished else 0],
    "PropertyType_Furnishing": [property_type_furnishing_encoded]
}
	
	# Convert dictionary to DataFrame
	user_input = pd.DataFrame(user_input_dict)

	# Reorder columns to match X_train (used in training)
	user_input = user_input[model.feature_names_in_]  # Ensures correct order

	# Predict price
	prediction = model.predict(user_input)
 	
	# Compute `Price_per_sqft`
	price_per_sqft = prediction[0] / size if size > 0 else 0
 
	st.write(f"Predicted Price: RM {prediction[0]:,.2f}")

else:
    
    st.write("Please insert value first before click predict button")
    