import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import requests
import pandas as pd
import numpy as np
from PIL import Image
import io
import json
import time
from datetime import datetime
import os
from dotenv import load_dotenv
import folium
from streamlit_folium import folium_static
import geocoder
from geopy.geocoders import Nominatim
import random
import tempfile
import base64

# Load environment variables
load_dotenv()

# Configure the page
st.set_page_config(
    page_title="AI Public Health Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("🏥 AI-Driven Public Health Chatbot")
st.markdown("""
Welcome to your AI-powered health assistant! This chatbot can:
- Provide information about diseases and prevention
- Analyze symptoms from text descriptions
- Examine skin conditions from images
- Locate nearby hospitals, pharmacies, and emergency services
- Offer general health advice
""")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Service selection
    st.subheader("Services Needed")
    hospital_finder = st.checkbox("Find Hospitals", value=True)
    pharmacy_finder = st.checkbox("Find Pharmacies", value=True)
    ambulance_service = st.checkbox("Ambulance Services", value=True)
    
    # Location input
    st.subheader("Your Location")
    location_method = st.radio("Location method:", ("Auto-detect", "Manual Entry"))
    
    user_location = None
    user_lat_lng = None
    
    if location_method == "Manual Entry":
        address = st.text_input("Enter your address or city:")
        if address:
            user_location = address
            try:
                geolocator = Nominatim(user_agent="health_chatbot")
                location = geolocator.geocode(address)
                if location:
                    user_lat_lng = (location.latitude, location.longitude)
                    st.success(f"Coordinates found: {user_lat_lng}")
            except Exception as e:
                st.error(f"Geocoding error: {str(e)}")
    else:
        if st.button("Detect My Location"):
            try:
                g = geocoder.ip('me')
                if g.ok:
                    user_lat_lng = (g.lat, g.lng)
                    user_location = f"{g.city}, {g.country}"
                    st.success(f"Location detected: {user_location}")
                else:
                    st.error("Could not detect location automatically")
            except Exception as e:
                st.error(f"Location detection failed: {str(e)}")
    
    # Disclaimer
    st.divider()
    st.warning("""
    **Disclaimer:** This AI assistant provides health information for educational purposes only. 
    It is not a substitute for professional medical advice, diagnosis, or treatment. 
    Always seek the advice of qualified healthcare providers with questions about medical conditions.
    """)

# Initialize Gemini model using environment variables
def setup_gemini():
    """Configure the Gemini model with safety settings using environment variables"""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        st.error("""
        GEMINI_API_KEY not found in environment variables.
        Please set it in your .env file or environment variables.
        Get your API key from https://aistudio.google.com/app/apikey
        """)
        st.stop()
    
    try:
        genai.configure(api_key=api_key)
        
        # Set up the model
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
        }

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        return model
    except Exception as e:
        st.error(f"Error configuring Gemini: {str(e)}")
        st.stop()

# Function to find nearby hospitals with realistic data
def find_nearby_hospitals(lat, lng, radius_km=10):
    """Find nearby hospitals with realistic data (simulated)"""
    try:
        # Sample hospital data with phone numbers
        hospitals = [
            {
                "name": "City General Hospital",
                "address": "123 Main St, City Center",
                "distance": round(random.uniform(0.5, 5.0), 1),
                "phone": "+1-555-0123",
                "type": "General Hospital",
                "rating": round(random.uniform(3.5, 5.0), 1),
                "latitude": lat + random.uniform(-0.05, 0.05),
                "longitude": lng + random.uniform(-0.05, 0.05)
            },
            {
                "name": "Community Medical Center",
                "address": "456 Oak Ave, West District",
                "distance": round(random.uniform(1.0, 6.0), 1),
                "phone": "+1-555-0456",
                "type": "Multi-specialty",
                "rating": round(random.uniform(3.0, 4.8), 1),
                "latitude": lat + random.uniform(-0.05, 0.05),
                "longitude": lng + random.uniform(-0.05, 0.05)
            },
            {
                "name": "Urgent Care Clinic",
                "address": "789 Pine Rd, East Side",
                "distance": round(random.uniform(0.3, 4.0), 1),
                "phone": "+1-555-0789",
                "type": "Urgent Care",
                "rating": round(random.uniform(3.2, 4.5), 1),
                "latitude": lat + random.uniform(-0.05, 0.05),
                "longitude": lng + random.uniform(-0.05, 0.05)
            },
            {
                "name": "Children's Hospital",
                "address": "101 Maple St, North Area",
                "distance": round(random.uniform(2.0, 7.0), 1),
                "phone": "+1-555-0321",
                "type": "Pediatric",
                "rating": round(random.uniform(4.0, 5.0), 1),
                "latitude": lat + random.uniform(-0.05, 0.05),
                "longitude": lng + random.uniform(-0.05, 0.05)
            },
            {
                "name": "Regional Medical Center",
                "address": "202 Elm Blvd, South Region",
                "distance": round(random.uniform(3.0, 8.0), 1),
                "phone": "+1-555-0654",
                "type": "Tertiary Care",
                "rating": round(random.uniform(3.8, 4.9), 1),
                "latitude": lat + random.uniform(-0.05, 0.05),
                "longitude": lng + random.uniform(-0.05, 0.05)
            }
        ]
        
        # Sort by distance
        hospitals.sort(key=lambda x: x["distance"])
        return hospitals
    except Exception as e:
        st.error(f"Error finding hospitals: {str(e)}")
        return []

# Function to find nearby pharmacies
def find_nearby_pharmacies(lat, lng, radius_km=5):
    """Find nearby pharmacies with realistic data (simulated)"""
    try:
        # Sample pharmacy data
        pharmacies = [
            {
                "name": "QuickScript Pharmacy",
                "address": "321 Elm St, Downtown",
                "distance": round(random.uniform(0.3, 3.0), 1),
                "phone": "+1-555-0987",
                "type": "Retail Pharmacy",
                "hours": "8:00 AM - 10:00 PM",
                "latitude": lat + random.uniform(-0.03, 0.03),
                "longitude": lng + random.uniform(-0.03, 0.03)
            },
            {
                "name": "24-Hour Drugstore",
                "address": "654 Maple Ave, City Center",
                "distance": round(random.uniform(0.5, 4.0), 1),
                "phone": "+1-555-0765",
                "type": "24/7 Pharmacy",
                "hours": "Open 24 hours",
                "latitude": lat + random.uniform(-0.03, 0.03),
                "longitude": lng + random.uniform(-0.03, 0.03)
            },
            {
                "name": "HealthMart Pharmacy",
                "address": "987 Birch Rd, West District",
                "distance": round(random.uniform(1.0, 5.0), 1),
                "phone": "+1-555-0432",
                "type": "Community Pharmacy",
                "hours": "9:00 AM - 9:00 PM",
                "latitude": lat + random.uniform(-0.03, 0.03),
                "longitude": lng + random.uniform(-0.03, 0.03)
            }
        ]
        
        # Sort by distance
        pharmacies.sort(key=lambda x: x["distance"])
        return pharmacies
    except Exception as e:
        st.error(f"Error finding pharmacies: {str(e)}")
        return []

# Function to call ambulance services
def emergency_services(location=None, phone_number=None):
    """Simulate calling emergency services with more details"""
    if not location:
        return "Please provide your location to call emergency services."
    
    # In a real application, this would integrate with emergency service APIs
    emergency_info = {
        "message": f"Emergency services have been notified and are on their way to {location}.",
        "estimated_time": f"{random.randint(5, 15)} minutes",
        "advice": "Please stay on the line for further instructions. Clear a path for emergency personnel.",
        "emergency_numbers": {
            "local_emergency": "911 (or your local emergency number)",
            "poison_control": "+1-800-222-1222",
            "mental_health_crisis": "+1-800-273-8255"
        }
    }
    
    return emergency_info

# Function to analyze image with Gemini - FIXED VERSION
def analyze_image(uploaded_file, prompt):
    """Analyze an uploaded image with Gemini - Fixed implementation"""
    if st.session_state.gemini_model is None:
        return "Please configure the Gemini API key first."
    
    try:
        # Read the image file
        image = Image.open(uploaded_file)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Create the prompt parts - FIXED FORMAT
        full_prompt = f"""
        You are a medical assistant AI. Analyze this image and provide information about potential health concerns.
        Focus on visible symptoms, possible conditions, and when to seek medical attention.
        Be cautious and always recommend consulting healthcare professionals for proper diagnosis.
        
        {prompt}
        """
        
        # Generate content using the correct format
        # Create a dictionary with the image data and mime type
        image_part = {
            "mime_type": "image/jpeg",
            "data": img_byte_arr
        }
        
        # Generate content
        response = st.session_state.gemini_model.generate_content([full_prompt, image_part])
        
        return response.text
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

# Function to get Gemini response
def get_gemini_response(prompt):
    """Get response from Gemini model"""
    if st.session_state.gemini_model is None:
        return "Please configure the Gemini API key first."
    
    try:
        response = st.session_state.gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error getting response: {str(e)}"

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = None
if "location" not in st.session_state:
    st.session_state.location = user_location
if "lat_lng" not in st.session_state:
    st.session_state.lat_lng = user_lat_lng

# Set up the model using environment variables
if st.session_state.gemini_model is None:
    st.session_state.gemini_model = setup_gemini()

# Main app interface
tab1, tab2, tab3, tab4 = st.tabs(["Chat", "Image Analysis", "Nearby Services", "Emergency Help"])

# Chat tab
with tab1:
    st.header("Health Chat Assistant")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about health concerns, symptoms, or prevention tips..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Add context to prompt for better medical responses
                medical_prompt = f"""
                You are a medical assistant AI. Provide helpful information about health concerns, 
                diseases, symptoms, prevention, and general wellness. Be accurate, cautious, and 
                always recommend consulting healthcare professionals for medical advice.
                
                Important: Do not provide diagnoses. Instead, suggest possible conditions and 
                emphasize the need for professional medical consultation.
                
                User question: {prompt}
                """
                
                response = get_gemini_response(medical_prompt)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Image Analysis tab - FIXED VERSION
with tab2:
    st.header("Symptom Image Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload an image of a symptom (e.g., skin condition, injury)", 
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Options for analysis
        analysis_type = st.radio(
            "What would you like to focus on?",
            ["General Analysis", "Skin Conditions", "Injury Assessment", "Other"]
        )
        
        custom_prompt = st.text_area("Or provide specific instructions for analysis:")
        
        if st.button("Analyze Image"):
            with st.spinner("Analyzing image..."):
                # Prepare prompt based on selection
                if analysis_type == "Skin Conditions":
                    prompt = "Focus on skin conditions. Describe what you see, potential causes, and when to see a dermatologist."
                elif analysis_type == "Injury Assessment":
                    prompt = "Focus on injuries. Describe what you see, potential severity, and first aid recommendations."
                elif analysis_type == "Other" and custom_prompt:
                    prompt = custom_prompt
                else:
                    prompt = "Provide a general analysis of any visible symptoms or concerns."
                
                # Analyze the image using the fixed function
                analysis_result = analyze_image(uploaded_file, prompt)
                
                # Display results
                st.subheader("Analysis Results")
                st.write(analysis_result)
                
                # Disclaimer
                st.warning("""
                **Important:** This AI analysis is not a medical diagnosis. 
                Always consult a healthcare professional for proper medical evaluation and treatment.
                """)

# Nearby Services tab
with tab3:
    st.header("Find Nearby Health Services")
    
    if not user_lat_lng:
        st.info("Please enable location access or enter your address in the sidebar to find nearby services.")
    else:
        st.success(f"Searching for services near: {user_location or f'Lat: {user_lat_lng[0]}, Lng: {user_lat_lng[1]}'}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if hospital_finder:
                st.subheader("🏥 Nearby Hospitals")
                hospitals = find_nearby_hospitals(user_lat_lng[0], user_lat_lng[1])
                
                if hospitals:
                    for hospital in hospitals:
                        with st.expander(f"{hospital['name']} ({hospital['distance']} km)"):
                            st.write(f"**Type:** {hospital['type']}")
                            st.write(f"**Address:** {hospital['address']}")
                            st.write(f"**Distance:** {hospital['distance']} km")
                            st.write(f"**Phone:** {hospital['phone']}")
                            st.write(f"**Rating:** {hospital['rating']:.1f}/5.0")
                            
                            # Button to call
                            if st.button(f"Call {hospital['name']}", key=f"call_{hospital['name']}"):
                                st.info(f"Calling {hospital['phone']}...")
                else:
                    st.info("No hospitals found nearby.")
        
        with col2:
            if pharmacy_finder:
                st.subheader("💊 Nearby Pharmacies")
                pharmacies = find_nearby_pharmacies(user_lat_lng[0], user_lat_lng[1])
                
                if pharmacies:
                    for pharmacy in pharmacies:
                        with st.expander(f"{pharmacy['name']} ({pharmacy['distance']} km)"):
                            st.write(f"**Type:** {pharmacy['type']}")
                            st.write(f"**Address:** {pharmacy['address']}")
                            st.write(f"**Distance:** {pharmacy['distance']} km")
                            st.write(f"**Phone:** {pharmacy['phone']}")
                            st.write(f"**Hours:** {pharmacy['hours']}")
                            
                            # Button to call
                            if st.button(f"Call {pharmacy['name']}", key=f"call_{pharmacy['name']}"):
                                st.info(f"Calling {pharmacy['phone']}...")
                else:
                    st.info("No pharmacies found nearby.")
        
        # Map visualization
        st.subheader("📍 Service Locations on Map")
        try:
            # Create a map centered on user's location
            m = folium.Map(location=[user_lat_lng[0], user_lat_lng[1]], zoom_start=13)
            
            # Add user location marker
            folium.Marker(
                [user_lat_lng[0], user_lat_lng[1]], 
                popup="Your Location", 
                tooltip="You are here",
                icon=folium.Icon(color='blue', icon='user')
            ).add_to(m)
            
            # Add hospital markers
            for hospital in hospitals:
                folium.Marker(
                    [hospital['latitude'], hospital['longitude']],
                    popup=f"{hospital['name']}<br>{hospital['address']}<br>Phone: {hospital['phone']}",
                    tooltip=hospital['name'],
                    icon=folium.Icon(color='red', icon='plus-sign')
                ).add_to(m)
            
            # Add pharmacy markers
            for pharmacy in pharmacies:
                folium.Marker(
                    [pharmacy['latitude'], pharmacy['longitude']],
                    popup=f"{pharmacy['name']}<br>{pharmacy['address']}<br>Phone: {pharmacy['phone']}",
                    tooltip=pharmacy['name'],
                    icon=folium.Icon(color='green', icon='shopping-cart')
                ).add_to(m)
            
            # Display the map
            folium_static(m, width=700, height=400)
        except Exception as e:
            st.error(f"Map error: {str(e)}")
            st.info("Map visualization is not available with the current location data.")

# Emergency Help tab
with tab4:
    st.header("🆘 Emergency Assistance")
    
    st.warning("""
    **For life-threatening emergencies, please call your local emergency number immediately!**
    """)
    
    st.subheader("Emergency Services")
    
    if ambulance_service and user_location:
        if st.button("🚑 Call Ambulance / Emergency Services", type="secondary", help="Simulated emergency call"):
            emergency_info = emergency_services(user_location)
            
            st.error(emergency_info["message"])
            st.info(f"**Estimated arrival time:** {emergency_info['estimated_time']}")
            st.info(f"**Advice:** {emergency_info['advice']}")
            
            st.subheader("Important Emergency Numbers")
            for service, number in emergency_info["emergency_numbers"].items():
                st.write(f"**{service.replace('_', ' ').title()}:** {number}")
    else:
        st.info("Please enable location services to access emergency features.")
    
    st.subheader("Emergency Health Information")
    
    emergency_topics = st.selectbox(
        "Select emergency topic for guidance:",
        ["Heart Attack", "Stroke", "Choking", "Severe Allergic Reaction", "Heavy Bleeding", "Poisoning", "Burn", "Seizure"]
    )
    
    if st.button("Get Emergency Instructions"):
        with st.spinner("Getting emergency instructions..."):
            prompt = f"""
            Provide concise first aid instructions for {emergency_topics}. 
            Include signs/symptoms to recognize, immediate actions to take, 
            and when to call emergency services. Format as bullet points.
            """
            
            response = get_gemini_response(prompt)
            st.markdown(response)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center'>
    <p>This AI health assistant is for informational purposes only and not a substitute for professional medical advice.</p>
    <p>Always consult healthcare professionals for medical concerns and emergencies.</p>
</div>
""", unsafe_allow_html=True)