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
import re
import speech_recognition as sr
from gtts import gTTS
import googletrans
from googletrans import Translator
import string

# Load environment variables
load_dotenv()

# Configure the page
st.set_page_config(
    page_title="HEALTHRI, Your AI Public Health Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("🏥 AI-Driven Public Health Chatbot")
st.markdown("""
Welcome to your AI-powered health assistant! This chatbot can:
- Provide information about diseases and prevention in multiple languages
- Analyze symptoms from text descriptions or voice input
- Examine skin conditions from images
- Locate nearby hospitals, pharmacies, and emergency services
- Offer general health advice
- Send health alerts via SMS (simulated)
- Support voice input and output in multiple languages
""")

# Initialize translator
translator = Translator()

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Language selection
    st.subheader("Language Settings")
    languages = {
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Hindi": "hi",
        "Chinese": "zh-cn",
        "Arabic": "ar",
        "Portuguese": "pt",
        "Russian": "ru",
        "Japanese": "ja",
        "Bengali": "bn",
        "Odia": "or",
        "Telugu": "te",
        "Tamil": "ta"
    }
    selected_language = st.selectbox(
        "Select Interface Language:",
        list(languages.keys()),
        index=0
    )
    lang_code = languages[selected_language]
    
    # Voice settings
    st.subheader("Voice Settings")
    voice_input = st.checkbox("Enable Voice Input", value=False)
    voice_output = st.checkbox("Enable Voice Output", value=False)
    
    # Service selection
    st.subheader("Services Needed")
    hospital_finder = st.checkbox("Find Hospitals", value=True)
    pharmacy_finder = st.checkbox("Find Pharmacies", value=True)
    ambulance_service = st.checkbox("Ambulance Services", value=True)
    sms_alerts = st.checkbox("SMS Health Alerts", value=True)
    
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
    
    # SMS configuration
    if sms_alerts:
        st.subheader("SMS Settings")
        phone_number = st.text_input("Phone number for alerts:", placeholder="+1234567890")
        alert_types = st.multiselect(
            "Select alert types:",
            ["Medication Reminders", "Appointment Alerts", "Health Tips", "Emergency Alerts"]
        )
    
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
            "local_emergery": "911 (or your local emergency number)",
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
        
        # Create the prompt parts - FIXED FORCT
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

# SMS Simulation Functions
def validate_phone_number(phone):
    """Validate phone number format"""
    # Simple validation - can be enhanced based on requirements
    pattern = r'^\+?[1-9]\d{1,14}$'
    return re.match(pattern, phone) is not None

def simulate_sms_send(phone_number, message):
    """Simulate sending an SMS message"""
    if not validate_phone_number(phone_number):
        return False, "Invalid phone number format"
    
    # In a real implementation, this would connect to an SMS gateway
    # For simulation, we'll just log the message
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sms_log = {
        "timestamp": timestamp,
        "to": phone_number,
        "message": message,
        "status": "SIMULATED_DELIVERY"
    }
    
    # Store in session state
    if "sms_log" not in st.session_state:
        st.session_state.sms_log = []
    st.session_state.sms_log.append(sms_log)
    
    return True, f"SMS simulated to {phone_number} at {timestamp}"

def send_health_alert(phone_number, alert_type, details=None):
    """Send health alert via SMS"""
    if not phone_number:
        return False, "Phone number not provided"
    
    # Generate appropriate message based on alert type
    if alert_type == "Medication Reminder":
        message = f"Health Alert: Remember to take your medication. {details if details else ''}"
    elif alert_type == "Appointment Alert":
        message = f"Health Alert: You have an upcoming appointment. {details if details else ''}"
    elif alert_type == "Health Tip":
        message = f"Health Tip: {details if details else 'Stay hydrated and maintain a balanced diet.'}"
    elif alert_type == "Emergency Alert":
        message = f"EMERGENCY: {details if details else 'Please seek immediate medical attention.'}"
    else:
        message = f"Health Notification: {details if details else 'This is a health notification.'}"
    
    # Add disclaimer
    message += " Reply STOP to unsubscribe. This is not a substitute for professional medical advice."
    
    return simulate_sms_send(phone_number, message)

def process_incoming_sms(phone_number, message):
    """Process incoming SMS messages and generate responses"""
    # Convert message to lowercase for easier processing
    msg_lower = message.lower().strip()
    
    # Simple command processing
    if msg_lower in ["hello", "hi", "start"]:
        return "Welcome to Health Assistant! Text: INFO for health info, LOCATE for nearby services, ALERTS for health alerts, HELP for emergency info."
    
    elif msg_lower == "info":
        return "Health Information: Text a symptom or health question (e.g. 'fever', 'headache') for information."
    
    elif msg_lower == "locate":
        return "Location Services: Please share your location or text your address to find nearby hospitals and pharmacies."
    
    elif msg_lower == "alerts":
        return "Health Alerts: Text 'REMINDER' for medication reminders, 'APPOINTMENT' for appointment alerts, 'TIPS' for health tips."
    
    elif msg_lower == "help":
        return "Emergency Help: In case of emergency, call your local emergency number. Text 'EMERGENCY' for first aid instructions."
    
    elif msg_lower == "reminder":
        return "Medication Reminder: Please text your medication name and time (e.g. 'Aspirin at 8AM') to set a reminder."
    
    elif msg_lower == "appointment":
        return "Appointment Alert: Please text your appointment details (e.g. 'Doctor on Monday 10AM') to set a reminder."
    
    elif msg_lower == "tips":
        return "Health Tip: Regular exercise, balanced diet, and adequate sleep are key to good health. Text a specific topic for more tips."
    
    elif msg_lower == "emergency":
        return "Emergency Info: Text: HEART for heart attack, STROKE for stroke, CHOKING for choking, BLEEDING for heavy bleeding, BURN for burns."
    
    elif msg_lower in ["heart", "stroke", "choking", "bleeding", "burn"]:
        # Get emergency instructions from Gemini
        prompt = f"""
        Provide concise first aid instructions for {msg_lower.upper()}. 
        Include signs/symptoms to recognize, immediate actions to take, 
        and when to call emergency services. Format as bullet points.
        Keep response under 160 characters for SMS.
        """
        try:
            response = get_gemini_response(prompt)
            # Truncate to SMS length if needed
            if len(response) > 160:
                response = response[:157] + "..."
            return response
        except:
            return "Please seek immediate medical help. Call emergency services for assistance."
    
    elif "stop" in msg_lower:
        return "You have unsubscribed from health alerts. Text START to resubscribe."
    
    else:
        # Assume it's a health question for Gemini
        prompt = f"""
        You are a medical assistant AI. Provide helpful information about health concerns in a concise manner.
        Be accurate, cautious, and always recommend consulting healthcare professionals for medical advice.
        Keep response under 160 characters for SMS.
        
        User question: {message}
        """
        try:
            response = get_gemini_response(prompt)
            # Truncate to SMS length if needed
            if len(response) > 160:
                response = response[:157] + "..."
            return response
        except:
            return "I'm sorry, I couldn't process your request. Please try again or contact a healthcare provider."

# Voice input function
def speech_to_text(language_code="en-US"):
    """Convert speech to text using microphone input"""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now")
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
            text = r.recognize_google(audio, language=language_code)
            return text
        except sr.WaitTimeoutError:
            return "No speech detected within timeout"
        except sr.UnknownValueError:
            return "Could not understand the audio"
        except sr.RequestError as e:
            return f"Error with speech recognition service: {e}"
        except Exception as e:
            return f"Unexpected error: {e}"

# Clean text for speech output
def clean_text_for_speech(text):
    """Remove or replace punctuation that shouldn't be spoken aloud"""
    # Remove bullet points and markers
    text = re.sub(r'[•\-*]\s*', '', text)
    
    # Replace certain punctuation with pauses
    text = text.replace(';', ',')
    text = text.replace(':', ',')
    
    # Remove parentheses and their content
    text = re.sub(r'\([^)]*\)', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Text to speech function
def text_to_speech(text, language_code="en"):
    """Convert text to speech and return audio data"""
    try:
        # Clean the text for better speech output
        cleaned_text = clean_text_for_speech(text)
        
        tts = gTTS(text=cleaned_text, lang=language_code, slow=False)
        audio_file = io.BytesIO()
        tts.write_to_fp(audio_file)
        audio_file.seek(0)
        return audio_file
    except Exception as e:
        st.error(f"Error in text-to-speech: {str(e)}")
        return None

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = None
if "location" not in st.session_state:
    st.session_state.location = user_location
if "lat_lng" not in st.session_state:
    st.session_state.lat_lng = user_lat_lng
if "sms_log" not in st.session_state:
    st.session_state.sms_log = []
if "current_language" not in st.session_state:
    st.session_state.current_language = lang_code

# Set up the model using environment variables
if st.session_state.gemini_model is None:
    st.session_state.gemini_model = setup_gemini()

# Main app interface
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Chat", "Image Analysis", "Nearby Services", "Emergency Help", "SMS Interface"])

# Chat tab
with tab1:
    st.header("Health Chat Assistant")
    
    # Voice input section
    if voice_input:
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🎤 Start Voice Input", use_container_width=True):
                # Map language code for speech recognition
                sr_lang_map = {
                    "en": "en-US", "es": "es-ES", "fr": "fr-FR", 
                    "de": "de-DE", "hi": "hi-IN", "zh-cn": "zh-CN",
                    "ar": "ar-SA", "pt": "pt-BR", "ru": "ru-RU", "ja": "ja-JP",
                    "bn": "bn-IN", "or": "or-IN", "te": "te-IN", "ta": "ta-IN"
                }
                sr_lang = sr_lang_map.get(lang_code, "en-US")
                
                voice_text = speech_to_text(sr_lang)
                if voice_text and voice_text not in ["No speech detected within timeout", 
                                                   "Could not understand the audio"]:
                    # Add to chat
                    st.session_state.messages.append({"role": "user", "content": voice_text})
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Add voice output for assistant messages if enabled
            if message["role"] == "assistant" and voice_output:
                audio_data = text_to_speech(message["content"], lang_code)
                if audio_data:
                    st.audio(audio_data, format="audio/mp3")
    
    # Chat input
    chat_input_placeholder = st.empty()
    prompt = chat_input_placeholder.chat_input("Ask about health concerns, symptoms, or prevention tips...")
    
    if prompt:
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
                
                Please respond in {selected_language} language.
                
                User question: {prompt}
                """
                
                response = get_gemini_response(medical_prompt)
                st.markdown(response)
                
                # Add voice output if enabled
                if voice_output:
                    audio_data = text_to_speech(response, lang_code)
                    if audio_data:
                        st.audio(audio_data, format="audio/mp3")
        
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
                
                # Add language instruction
                prompt += f"\nPlease respond in {selected_language} language."
                
                # Analyze the image using the fixed function
                analysis_result = analyze_image(uploaded_file, prompt)
                
                # Display results
                st.subheader("Analysis Results")
                st.write(analysis_result)
                
                # Add voice output if enabled
                if voice_output:
                    audio_data = text_to_speech(analysis_result, lang_code)
                    if audio_data:
                        st.audio(audio_data, format="audio/mp3")
                
                # Send as SMS if requested
                if sms_alerts and phone_number:
                    if st.button("Send Analysis via SMS"):
                        # Truncate if too long for SMS
                        if len(analysis_result) > 150:
                            sms_message = analysis_result[:147] + "..."
                        else:
                            sms_message = analysis_result
                        
                        success, result_msg = simulate_sms_send(phone_number, sms_message)
                        if success:
                            st.success(f"Analysis sent to {phone_number}")
                        else:
                            st.error(f"Failed to send SMS: {result_msg}")
                
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
                            
                            # Button to send details via SMS
                            if sms_alerts and phone_number:
                                if st.button(f"Send via SMS", key=f"sms_{hospital['name']}"):
                                    message = f"{hospital['name']} - {hospital['address']} - Phone: {hospital['phone']} - {hospital['distance']}km away"
                                    success, result_msg = simulate_sms_send(phone_number, message)
                                    if success:
                                        st.success(f"Hospital details sent to {phone_number}")
                                    else:
                                        st.error(f"Failed to send SMS: {result_msg}")
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
                            
                            # Button to send details via SMS
                            if sms_alerts and phone_number:
                                if st.button(f"Send via SMS", key=f"sms_{pharmacy['name']}"):
                                    message = f"{pharmacy['name']} - {pharmacy['address']} - Phone: {pharmacy['phone']} - {pharmacy['distance']}km away"
                                    success, result_msg = simulate_sms_send(phone_number, message)
                                    if success:
                                        st.success(f"Pharmacy details sent to {phone_number}")
                                    else:
                                        st.error(f"Failed to send SMS: {result_msg}")
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
            
            # Send emergency alert via SMS
            if sms_alerts and phone_number:
                message = f"EMERGENCY: Assistance requested at {user_location}. Estimated arrival: {emergedy_info['estimated_time']}"
                success, result_msg = send_health_alert(phone_number, "Emergency Alert", message)
                if success:
                    st.success(f"Emergency alert sent to {phone_number}")
                else:
                    st.error(f"Failed to send alert: {result_msg}")
            
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
            Please respond in {selected_language} language.
            """
            
            response = get_gemini_response(prompt)
            st.markdown(response)
            
            # Add voice output if enabled
            if voice_output:
                audio_data = text_to_speech(response, lang_code)
                if audio_data:
                    st.audio(audio_data, format="audio/mp3")
            
            # Option to send via SMS
            if sms_alerts and phone_number:
                if st.button("Send Instructions via SMS"):
                    # Truncate if too long for SMS
                    if len(response) > 150:
                        sms_message = response[:147] + "..."
                    else:
                        sms_message = response
                    
                    success, result_msg = simulate_sms_send(phone_number, sms_message)
                    if success:
                        st.success(f"Emergency instructions sent to {phone_number}")
                    else:
                        st.error(f"Failed to send SMS: {result_msg}")

# SMS Interface tab
with tab5:
    st.header("SMS Health Assistant")
    st.info("""
    This simulates an SMS-based health assistant. In a real implementation, 
    this would connect to an SMS gateway service to send and receive messages.
    """)
    
    # SMS Simulation Interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Send Health Alert")
        
        alert_type = st.selectbox(
            "Alert Type:",
            ["Medication Reminder", "Appointment Alert", "Health Tip", "Emergency Alert"]
        )
        
        alert_details = st.text_area("Alert Details:")
        
        if st.button("Send Health Alert via SMS"):
            if phone_number and validate_phone_number(phone_number):
                success, result_msg = send_health_alert(phone_number, alert_type, alert_details)
                if success:
                    st.success(f"Alert sent to {phone_number}")
                else:
                    st.error(f"Failed to send alert: {result_msg}")
            else:
                st.error("Please provide a valid phone number in the sidebar")
    
    with col2:
        st.subheader("Simulate Incoming SMS")
        
        simulated_number = st.text_input("From number:", placeholder="+1234567890")
        simulated_message = st.text_area("Message content:")
        
        if st.button("Process Incoming SMS"):
            if simulated_number and simulated_message:
                response = process_incoming_sms(simulated_number, simulated_message)
                st.info(f"Response: {response}")
                
                # Log the interaction
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                sms_log = {
                    "timestamp": timestamp,
                    "from": simulated_number,
                    "message": simulated_message,
                    "response": response,
                    "direction": "INBOUND"
                }
                st.session_state.sms_log.append(sms_log)
            else:
                st.error("Please provide both a number and message")
    
    # SMS Log
    st.subheader("SMS Message Log")
    if st.session_state.sms_log:
        for i, log in enumerate(reversed(st.session_state.sms_log)):
            with st.expander(f"{log.get('timestamp', 'Unknown time')} - {log.get('direction', 'Unknown')}"):
                if log.get('direction') == 'INBOUND':
                    st.write(f"**From:** {log.get('from', 'Unknown')}")
                    st.write(f"**Message:** {log.get('message', 'No message')}")
                    st.write(f"**Response:** {log.get('response', 'No response')}")
                else:
                    st.write(f"**To:** {log.get('to', 'Unknown')}")
                    st.write(f"**Message:** {log.get('message', 'No message')}")
                    st.write(f"**Status:** {log.get('status', 'Unknown')}")
    else:
        st.info("No SMS messages yet. Send or receive a message to see them here.")
    
    # Clear log button
    if st.button("Clear SMS Log"):
        st.session_state.sms_log = []
        st.rerun()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center'>
    <p>This AI health assistant is for informational purposes only and not a substitute for professional medical advice.</p>
    <p>Always consult healthcare professionals for medical concerns and emergencies.</p>
    <p>SMS functionality is simulated. Real implementation requires an SMS gateway service.</p>
</div>
""", unsafe_allow_html=True)
