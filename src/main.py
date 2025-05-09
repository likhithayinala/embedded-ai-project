from picamera2 import Picamera2
import cv2
from audio import ESP32Recorder
import numpy as np
import speech_recognition as sr
import sounddevice as sd
from faster_whisper import WhisperModel
import time
import pyttsx3
import requests
import google.generativeai as genai
from kwt.models.kwt import kwt_from_name, KWT
import json
import os
import datetime
import logging
from PIL import Image
from fine_tune import fine_tune_model
import torch

def setup_results_directory():
    """
    Creates a results directory with a timestamp-based subfolder for the current run.
    Returns the path to the current run directory.
    """
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Create a timestamped folder for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("results", timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    return run_dir

def setup_logging(run_dir):
    """
    Sets up logging to save all console output to a log file.
    """
    log_file = os.path.join(run_dir, "session.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also output to console
        ]
    )
    
    return logging.getLogger()

# Set up results directory and logging
run_dir = setup_results_directory()
logger = setup_logging(run_dir)
logger.info(f"Session started. Results will be saved to: {run_dir}")

# Retrieve API keys from environment 
# Read .txt file for API keys
with open("data/api_keys.txt", "r") as f:
    lines = f.readlines()
    # Remove any leading/trailing whitespace characters
    lines = [line.strip() for line in lines]
    # Assign the keys to variables
    OPENWEATHER_API_KEY = lines[0]
    GEMINI_FLASH_API_KEY = lines[1]

def keyword_detection(text, keywords):
    """
    Checks if any activation keyword from the list is found in the recognized text.
    Returns True if one is detected.
    """
    for keyword in keywords:
        if keyword.lower() in text.lower():
            return True
    return True

def pre_process_audio(recording):
    """
    Preprocesses audio recording to create a mel spectrogram.
    Returns the processed audio features.
    """
    import torch.nn.functional as F
    import torchaudio.transforms as T
    
    # Convert numpy array to tensor and normalize to float between -1 and 1
    wav = torch.tensor(recording.flatten(), dtype=torch.float32) / 32768.0
    wav = wav.unsqueeze(0)  # Add channel dimension: (1, samples)
    
    SAMPLE_RATE = 16000  # Using the same sample rate as in recording
    
    # Create mel spectrogram
    spec = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, win_length=640,
        hop_length=160, n_mels=40
    )(wav)  # (1, 40, T)
    
    log_mel = torch.log(spec + 1e-6)
    
    # Pad/crop to 98 frames
    if log_mel.shape[-1] < 98:
        pad = 98 - log_mel.shape[-1]
        log_mel = F.pad(log_mel, (0, pad))
    log_mel = log_mel[:, :, :98]
    
    return log_mel

def start_listening(model, rec):
    """
    Initializes the microphone and listens for speech input.
    Returns the recognized text.
    """
    duration = 2  # 2 seconds of recording
    fs = 16000  # 16kHz sampling rate
    recording = rec.record_chunk(duration)
    sd.wait()  
    try:
        processed_audio = pre_process_audio(recording)
        logits = model(processed_audio.unsqueeze(0))  # Move to GPU if available
        probs = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()
        if predicted_class == 1:  # Assuming '1' is the class for "yes"
            logger.info("Activation keyword detected")
        return True

    except sr.RequestError as e:
        logger.error(f"Pocketsphinx error: {e}")
    except sr.WaitTimeoutError:
        logger.error("Listening timed out while waiting for phrase to start")
    except sr.UnknownValueError:
        logger.error("Pocketsphinx could not understand audio")
    except sr.RequestError as e:
        logger.error(f"Pocketsphinx error: {e}")
    return ""

def speech_to_text_offline(model, rec, time=5):
    """
    Listens for speech input via the microphone and performs offline speech recognition
    using pocketsphinx. Returns the recognized text.
    """
    # Define duration (in seconds) and sampling frequency
    duration = time  # 2 seconds of recording
    fs = 16000  # 16kHz sampling rate
    recording = rec.record_chunk(duration)
    sd.wait()  
    try:
        # Pass the numpy array recording directly to the model
        segments, _ = model.transcribe(recording.flatten().astype(np.float32) / 32768.0, beam_size=5)
        text = [segment.text for segment in segments if segment.text]
        text = " ".join(text)
        logger.info(f"Recognized Speech (offline): {text}")
        return text
    except sr.RequestError as e:
        logger.error(f"Pocketsphinx error: {e}")
    except sr.WaitTimeoutError:
        logger.error("Listening timed out while waiting for phrase to start")
    except sr.UnknownValueError:
        logger.error("Pocketsphinx could not understand audio")
    except sr.RequestError as e:
        logger.error(f"Pocketsphinx error: {e}")
    return ""

def capture_user_image():
    """
    Captures a single frame from the default camera.
    Returns the image (in BGR format) if successful.
    """
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(
        main={"size": (640, 480), "format": "XRGB8888"}
    ))
    picam2.start()
    
    # Give camera time to adjust to lighting conditions
    logger.info("Allowing camera to adjust...")
    for _ in range(10):  # Capture and discard frames to let camera adjust
        time.sleep(0.1)  # Short delay between frames
    
    # Now capture the actual frame we want to use
    frame = picam2.capture_array()
    # Rotate the image 90 degrees clockwise
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if frame.any():
        logger.info("User image captured")
        # Save the image to the run directory
        image_path = os.path.join(run_dir, "user_image.jpg")
        cv2.imwrite(image_path, frame)
        return frame
    else:
        logger.error("Failed to capture image")
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
    logger.info(f"Detected {len(faces)} face(s) for blurring")
    
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
            logger.info(f"Extracted location: {location}")
            return location
    logger.info("No location found in input. Defaulting to New York.")
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
        logger.info(f"Retrieved weather for {location}: {temp}°C, {description}")
        return {"temperature": temp, "description": description}
    else:
        logger.error(f"Weather API error: {response.status_code}")
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
        f"You are a fashion assistant. Your job is to help me pick a good outfits for the day. I am going to provide the speech and the image. Based on both give me a suggestion. Do not exceed 50 words. "
        f"User speech: '{input_text}'. The current weather is {weather['description']} with "
        f"a temperature of {weather['temperature']}°C. Based on the attached image, please provide a detailed outfit recommendation. Follow the instructions carefully."
    )

    try:
        # Load image using PIL (Gemini expects PIL.Image)
        if image_path is None:
            response = model.generate_content([prompt_text])
            return response.text
        else:
            # Open the image using PIL
            logger.info("Opening image for Gemini API...")
        image = Image.open(image_path)

        logger.info("Sending request to Gemini Vision API...")
        response = model.generate_content([prompt_text, image])
        
        if hasattr(response, 'text'):
            recommendation = response.text
            logger.info(f"Outfit Recommendation: {recommendation}")
            return recommendation
        else:
            return "I'm sorry, I couldn't retrieve an outfit recommendation."
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        return "I'm sorry, I couldn't retrieve an outfit recommendation."

def text_to_speech(text, save=False):
    """
    Converts text to speech using the pyttsx3 offline engine.
    """
    if not save: 
        engine = pyttsx3.init()
        engine.setProperty('volume', 1.0)
        engine.setProperty('voice', 'en-uk') 
        engine.say(text)
        engine.runAndWait()
        time.sleep(1)  # Wait for the speech to finish
    # Save the audio to a file
    if save:
        # Save the audio to a file
        output_path = os.path.join(run_dir, "output.mp3")
        engine = pyttsx3.init()
        engine.setProperty('volume', 1.0)
        engine.setProperty('voice', 'en-uk') 
        engine.say(text)
        engine.save_to_file(text, output_path)
        engine.runAndWait()
        engine.stop()
    # Explicitly stop and dispose of the engine
    engine.stop()
    logger.info("Spoken output delivered")

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
    # Step 2: Audio Recorder 
    rec = ESP32Recorder('/dev/ttyACM0', 115200)

    # if trained model doesn't exist then train it
    if not os.path.exists("yesno_trained_model.pth"):
        logger.info("Training the keyword detection model...")
        text_to_speech("Training the keyword detection model. Please say yes 5 times and then no 5 times.")
        # Load the pre-trained model
        base_model = fine_tune_model(rec)
        # Ask user for city and wardrobe information

        # Define path for storing user data
        user_data_path = os.path.join(run_dir, "user_info.json")

        # Initialize user data dictionary
        user_data = {}

        # Ask for city
        text_to_speech("Please tell me which city you live in.")
        city_response = speech_to_text_offline(model, rec, time=2)
        # Remove any duplicate words
        city_response = " ".join(dict.fromkeys(city_response.split()))
        # Check if the response is empty
        if city_response:
            user_data["city"] = city_response
            logger.info(f"Recorded city: {city_response}")
        else:
            user_data["city"] = "New York"
            logger.info("Could not record city information. Defaulting to New York.")

        # Ask for wardrobe collection
        text_to_speech("Please describe your wardrobe collection briefly.")
        wardrobe_response = speech_to_text_offline(model, rec, time=25)
        if wardrobe_response:
            user_data["wardrobe"] = wardrobe_response
            logger.info(f"Recorded wardrobe: {wardrobe_response}")
        else:
            user_data["wardrobe"] = "Unknown"
            logger.info("Could not record wardrobe information")

        # Save to JSON file
        with open(user_data_path, "w") as f:
            json.dump(user_data, f, indent=2)
        logger.info(f"User information saved to {user_data_path}")
    else:
        # Load existing user data
        try:
            user_data_path = "data/user_info.json"  # Try to load from original location
            with open(user_data_path, "r") as f:
                user_data = json.load(f)
            # Copy the user data to the current run directory
            new_user_data_path = os.path.join(run_dir, "user_info.json")
            with open(new_user_data_path, "w") as f:
                json.dump(user_data, f, indent=2)
        except FileNotFoundError:
            logger.error("User data file not found. Creating default values.")
            user_data = {"city": "New York", "wardrobe": "Unknown"}

    city = user_data.get("city", "Unknown")
    wardrobe = user_data.get("wardrobe", "Unknown")

    cfg = {
        "input_res":[40,98],"patch_res":[40,1],"num_classes":2,
        "mlp_dim":256,"dim":64,"heads":1,"depth":12,
        "dropout":0.0,"emb_dropout":0.1,"pre_norm":False
    }
    detection_model = KWT(**cfg)
    detection_model_weights = torch.load("yesno_trained_model.pth", map_location="cpu")
    detection_model.load_state_dict(detection_model_weights, strict=False)
    detection_model.eval()

    logger.info("Listening for speech input...")
    start = False
    start_time = time.time()
    while not start and time.time() - start_time < 40:
        start = start_listening(detection_model, rec)
        if not start:
            logger.info("No activation keyword detected. Listening again...")
            time.sleep(1)

    if not start:
        logger.info("40 seconds elapsed with no activation. Exiting.")
        return
    logger.info("Started Listening to User's voice")
    text_to_speech("Hello! How can I help you today?")
    text = speech_to_text_offline(model, rec, time=6)
    logger.info(f"Recognized Speech (offline): {text}")
    # Ask user if they want to take a picture
    text_to_speech("Could I take a picture of you? Please say yes or no.")
    user_response = speech_to_text_offline(model, rec, time=2)
    if "yes" in user_response.lower():
        logger.info("User agreed to take a picture.")
        # Step 3: Capture image and blur faces
        user_image = capture_user_image()
        if user_image is not None:
            user_image_blurred = blur_faces(user_image)
            image_path = os.path.join(run_dir, "user_image_blurred.jpg")
            cv2.imwrite(image_path, user_image_blurred)
            logger.info(f"User image with blurred faces saved as {image_path}")
        else:
            logger.error("Skipping image processing due to capture error.")
            return
    else:
        logger.info("User declined to take a picture.")
        image_path = None
    # Step 4: Extract location and get weather data
    location = user_data["city"]
    weather = get_weather_data(location)

    # Step 5: Get outfit recommendation via Gemini Flash 2.0 using the blurred image
    text = text + str(user_data["wardrobe"])
    recommendation = get_outfit_recommendation_with_image(text, weather, image_path)
    logger.info(f"Final Recommendation: {recommendation}")

    # Step 6: Output recommendation via TTS
    recommendation = recommendation.replace("*", "")
    text_to_speech(recommendation, save=True)
    rec.close()

if __name__ == "__main__":
    main()
