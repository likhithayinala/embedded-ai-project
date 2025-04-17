
import cv2
import numpy as np
import speech_recognition as sr
import sounddevice as sd
from faster_whisper import WhisperModel
import time
import pyttsx3
import requests
import google.generativeai as genai
import os
from PIL import Image

# Retrieve API keys from environment
OPENWEATHER_API_KEY = "f34b5df8a4a390837dcfce87c604f68f"  # For the weather API
GEMINI_FLASH_API_KEY = "AIzaSyCuGmoh6C5dZjsO0io0nWxFRqab5KsYE3w"  # For the Gemini Flash 2.0 API

def keyword_detection(text, keywords):
    """
    Checks if any activation keyword from the list is found in the recognized text.
    Returns True if one is detected.
    """
    for keyword in keywords:
        if keyword.lower() in text.lower():
            return True
    return True

def start_listening(model):
    """
    Initializes the microphone and listens for speech input.
    Returns the recognized text.
    """
    duration = 2  # 2 seconds of recording
    fs = 16000  # 16kHz sampling rate
    keywords = ["weather", "outfit", "recommendation", "fashion", "start"]
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  
    try:
        # Pass the numpy array recording directly to the model
        segments,_ = model.transcribe(recording.flatten().astype(np.float32) / 32768.0, beam_size=5)
        text = [segment.text for segment in segments if segment.text]
        text = " ".join(text)
        print("Recognized Speech (offline):", text)
        if keyword_detection(text, keywords):
            print("Activation keyword detected.")
            return True, text
        else:
            print("No activation keyword detected.")
            return False, ""
    except sr.RequestError as e:
        print("Pocketsphinx error: {0}".format(e))
    except sr.WaitTimeoutError:
        print("Listening timed out while waiting for phrase to start")
    except sr.UnknownValueError:
        print("Pocketsphinx could not understand audio")
    except sr.RequestError as e:
        print("Pocketsphinx error: {0}".format(e))
    return ""

def speech_to_text_offline(model,time=5):
    """
    Listens for speech input via the microphone and performs offline speech recognition
    using pocketsphinx. Returns the recognized text.
    """
    # Define duration (in seconds) and sampling frequency
    duration = time  # 2 seconds of recording
    fs = 16000  # 16kHz sampling rate
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  
    try:
        # Pass the numpy array recording directly to the model
        segments,_ = model.transcribe(recording.flatten().astype(np.float32) / 32768.0, beam_size=5)
        text = [segment.text for segment in segments if segment.text]
        text = " ".join(text)
        print("Recognized Speech (offline):", text)
        return text
    except sr.RequestError as e:
        print("Pocketsphinx error: {0}".format(e))
    except sr.WaitTimeoutError:
        print("Listening timed out while waiting for phrase to start")
    except sr.UnknownValueError:
        print("Pocketsphinx could not understand audio")
    except sr.RequestError as e:
        print("Pocketsphinx error: {0}".format(e))
    return ""



def capture_user_image():
    """
    Captures a single frame from the default camera.
    Returns the image (in BGR format) if successful.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera could not be opened")
        return None

    ret, frame = cap.read()
    cap.release()
    if ret:
        print("User image captured")
        return frame
    else:
        print("Failed to capture image")
        return None

def blur_faces(image):
    """
    Detects faces using Haar cascades and applies Gaussian blur on each detected face.
    Returns the processed image.
    """
    # Load pre-trained face detection model from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    print(f"Detected {len(faces)} face(s) for blurring")
    
    for (x, y, w, h) in faces:
        face_region = image[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
        image[y:y+h, x:x+w] = blurred_face
    return image

def extract_location(text):
    """
    A simple heuristic to extract a location from the speech input.
    For example, if the text contains "in London", this returns "London".
    Defaults to 'New York' if no location is found.
    """
    tokens = text.split()
    if "in" in tokens:
        idx = tokens.index("in")
        if idx + 1 < len(tokens):
            location = tokens[idx+1].strip(",.?!")
            print("Extracted location:", location)
            return location
    print("No location found in input. Defaulting to New York.")
    return "New York"

def get_weather_data(location):
    """
    Retrieves current weather data from a traditional weather API (e.g., OpenWeatherMap)
    for the provided location. Returns a dictionary with temperature and description.
    """
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        temp = data['main']['temp']
        description = data['weather'][0]['description']
        print(f"Retrieved weather for {location}: {temp}°C, {description}")
        return {"temperature": temp, "description": description}
    else:
        print("Weather API error:", response.status_code)
        return None

def get_outfit_recommendation_with_image(input_text, weather, image_path):
    """
    Sends the blurred image along with textual details (user input and weather information)
    to the Gemini Vision API to obtain an outfit recommendation.
    """
    
    # Configure the Gemini API with the API key
    genai.configure(api_key=GEMINI_FLASH_API_KEY)
    model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")

    # Construct a prompt that incorporates the spoken text and weather details
    prompt_text = (
        f"User speech: '{input_text}'. The current weather is {weather['description']} with "
        f"a temperature of {weather['temperature']}°C. Based on the attached image, please provide a detailed outfit recommendation."
    )

    try:
        # Load image using PIL (Gemini expects PIL.Image)
        image = Image.open(image_path)

        print("Sending request to Gemini Vision API...")
        response = model.generate_content([prompt_text, image])
        
        if hasattr(response, 'text'):
            recommendation = response.text
            print("Outfit Recommendation:", recommendation)
            return recommendation
        else:
            return "I'm sorry, I couldn't retrieve an outfit recommendation."
    except Exception as e:
        print("Error calling Gemini API:", e)
        return "I'm sorry, I couldn't retrieve an outfit recommendation."

def text_to_speech(text):
    """
    Converts text to speech using the pyttsx3 offline engine.
    """
    engine = pyttsx3.init()
    engine.setProperty('volume', 1.0)
    engine.setProperty('voice', 'en-us') 
    engine.say(text)
    engine.runAndWait()
    time.sleep(1)  # Wait for the speech to finish
    # Explicitly stop and dispose of the engine
    engine.stop()
    print("Spoken output delivered")

def main():
    """
    Main pipeline:
      1. Capture offline speech-to-text.
      2. Check for activation keywords.
      3. Capture and blur user image.
      4. Extract location and get weather data.
      5. Send the blurred image and textual context to Gemini Flash 2.0 for an outfit recommendation.
      6. Convert the recommendation to speech.
    """
    # Step 1: Offline speech recognition
    model = WhisperModel("small", compute_type="int8")

    print("Listening for speech input...")
    start = False
    start_time = time.time()
    while not start and time.time() - start_time < 40:
        start, input_text = start_listening(model)
        if not start:
            print("No activation keyword detected. Listening again...")
            time.sleep(1)

    if not start:
        print("40 seconds elapsed with no activation. Exiting.")
        return
    print("Started Listening to User's voice")
    text = speech_to_text_offline(model,time=10)
    print("Recognized Speech (offline):", text)
    # Ask user if they want to take a picture
    text_to_speech("Could I take a picture of you? Please say yes or no.")
    user_response = speech_to_text_offline(model,time=1)
    if "yes" in user_response.lower():
        print("User agreed to take a picture.")
        # Step 3: Capture image and blur faces
        user_image = capture_user_image()
        if user_image is not None:
            user_image_blurred = blur_faces(user_image)
            image_path = "user_image_blurred.jpg"
            cv2.imwrite(image_path, user_image_blurred)
            print(f"User image with blurred faces saved as {image_path}")
        else:
            print("Skipping image processing due to capture error.")
            return
    else:
        print("User declined to take a picture.")
        image_path = None
    # Step 4: Extract location and get weather data
    location = extract_location(input_text)
    weather = get_weather_data(location)
    if not weather:
        print("Weather information unavailable. Exiting.")
        return

    # Step 5: Get outfit recommendation via Gemini Flash 2.0 using the blurred image
    recommendation = get_outfit_recommendation_with_image(input_text, weather, image_path)
    print("Final Recommendation:", recommendation)

    # Step 6: Output recommendation via TTS
    text_to_speech(recommendation)

if __name__ == "__main__":
    main()
