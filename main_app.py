import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import base64

st.markdown('<h1 class="title">CROPCURE AI - Detect Early, Protect Always</h1>', unsafe_allow_html=True)


# ✅ Encode Image to Base64 for Background
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        st.error("Background image file not found. Please check the path.")
        return None

# ✅ Background Image Path
background_image_path = "C:/Users/hp/Downloads/Plant_Disease/male-farmer-with-beard-check-tea-farm_1150-14747.jpg"
background_image_base64 = get_base64_of_bin_file(background_image_path)

# ✅ Custom CSS for Styling
if background_image_base64:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{background_image_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .title {{
            color: white;
            font-size: 2.5em;
            font-weight: bold;
            text-align: center;
        }}
        .prediction {{
            font-size: 1.5em;
            color: blue;
            background-color: #FFFFFF;
            padding: 10px;
            border-radius: 10px;
            text-align: center;
        }}
        .analysis {{
            font-size: 1.2em;
            color: black;
            background-color: #F0F0F0;
            padding: 8px;
            border-radius: 8px;
            text-align: center;
        }}
        .visualization-title {{
            font-size: 1.8em;
            font-weight: bold;
            text-align: center;
            color: #00000;
            background-color: #F0F0F0;
            padding: 10px;
            border-radius: 4px;
            margin-top: 20px;
        }}
        .preventive-measures {{
            font-size: 1.2em;
            color: black;
            background-color: #F9F9F9;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            border: 1px solid #DDDDDD;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ Load the Model
model_path = "C:/Users/hp/Downloads/Plant_Disease/plant_disease.h5"
model = load_model(model_path)

# ✅ Class Names and Disease Management Dictionary
CLASS_NAMES = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']
DISEASE_MANAGEMENT = {
    'Corn-Common_rust': {
        'Description': 'A fungal disease that causes rust-colored spots on leaves.',
        'Preventive Measures': [
            'Use disease-resistant corn varieties.',
            'Avoid overhead irrigation.',
            'Rotate crops annually.'
        ],
        'Pesticides': [
            'Fungicide A',
            'Fungicide B'
        ]
    },
    'Potato-Early_blight': {
        'Description': 'A fungal disease that affects potato leaves, causing brown spots.',
        'Preventive Measures': [
            'Plant certified disease-free seeds.',
            'Remove infected plant debris.',
            'Apply fungicides early.'
        ],
        'Pesticides': [
            'Fungicide X',
            'Fungicide Y'
        ]
    },
    'Tomato-Bacterial_spot': {
        'Description': 'Bacterial infection causing dark spots on tomato leaves and fruits.',
        'Preventive Measures': [
            'Use pathogen-free seeds.',
            'Ensure proper air circulation.',
            'Apply copper-based bactericides.'
        ],
        'Pesticides': [
            'Bactericide M',
            'Bactericide N'
        ]
    }
}
# ✅ Upload Image
plant_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# ✅ Prediction Button
submit = st.button('Predict & Analyze')

# ✅ Infection Analysis Function
def analyze_infection(image):
    """
    Analyze the infection levels in a plant image.
    Args:
        image (numpy array): The input plant leaf image in OpenCV format.
    Returns:
        infected_percentage (float): Percentage of infected regions.
        healthy_percentage (float): Percentage of healthy regions.
    """
    # Convert image to HSV for better segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define mask for infected regions (e.g., brown, yellowish areas)
    infected_mask = cv2.inRange(hsv, (10, 40, 20), (35, 255, 255))
    healthy_mask = cv2.inRange(hsv, (36, 50, 70), (89, 255, 255))
    
    # Calculate pixel counts
    total_pixels = image.shape[0] * image.shape[1]
    infected_pixels = cv2.countNonZero(infected_mask)
    healthy_pixels = cv2.countNonZero(healthy_mask)
    
    # Calculate percentages
    infected_percentage = (infected_pixels / total_pixels) * 100
    healthy_percentage = (healthy_pixels / total_pixels) * 100
    
    return infected_percentage, healthy_percentage

# ✅ On Prediction Button Click
if submit:
    if plant_image is not None:
        # Read and display the image
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        st.image(opencv_image, channels="BGR", caption="Uploaded Plant Image", use_container_width=True)
        
        # Resize image for prediction
        resized_image = cv2.resize(opencv_image, (256, 256))
        image_array = np.expand_dims(resized_image, axis=0)
        
        # ✅ Make Prediction
        Y_pred = model.predict(image_array)
        result = CLASS_NAMES[np.argmax(Y_pred)]
        st.markdown(f'<p class="prediction">This is a {result.split("-")[0]} leaf with {result.split("-")[1]}</p>', unsafe_allow_html=True)
        
        # ✅ Analyze Infection
        infected_percentage, healthy_percentage = analyze_infection(opencv_image)
        st.markdown(
            f"""
            <div class="analysis">
                <p><strong>Infection Analysis:</strong></p>
                <p>Infected Area: <strong>{infected_percentage:.2f}%</strong></p>
                <p>Healthy Area: <strong>{healthy_percentage:.2f}%</strong></p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # ✅ Fetch Management Info
    disease_key = result.split("-")[0] + '-' + result.split("-")[1]
    if disease_key in DISEASE_MANAGEMENT:
        management_info = DISEASE_MANAGEMENT[disease_key]
        st.markdown(
            f"""
           <div class="preventive-measures">
               <p><strong>Disease:</strong> {management_info['Description']}</p>
               <p><strong>Preventive Measures:</strong></p>
               <ul>
                   {''.join([f'<li>{measure}</li>' for measure in management_info['Preventive Measures']])}
               </ul>
               <p><strong>Recommended Pesticides:</strong></p>
               <ul>
                   {''.join([f'<li>{pesticide}</li>' for pesticide in management_info['Pesticides']])}
               </ul>
           </div>
            """,
           unsafe_allow_html=True
      )
    else:  # Ensure this line is followed immediately by the code or block
        st.warning("No management information found for this disease.")


        # ✅ Display Masked Images
    hsv = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2HSV)
    infected_mask = cv2.inRange(hsv, (10, 40, 20), (35, 255, 255))
    healthy_mask = cv2.inRange(hsv, (36, 50, 70), (89, 255, 255))

    infected_display = cv2.bitwise_and(opencv_image, opencv_image, mask=infected_mask)
    healthy_display = cv2.bitwise_and(opencv_image, opencv_image, mask=healthy_mask)

    st.markdown(
        '<p class="visualization-title">Infected and Healthy Regions Visualization</p>',
        unsafe_allow_html=True
        )
    col1, col2 = st.columns(2)
    with col1:
        st.image(infected_display, caption="Infected Region", use_column_width=True)
    with col2:
        st.image(healthy_display, caption="Healthy Region", use_column_width=True)
else:
     st.warning("No management information found for this disease.")

