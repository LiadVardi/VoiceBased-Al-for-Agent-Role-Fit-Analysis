import numpy as np
import librosa
import pickle
from tensorflow.keras.models import load_model

# Load the model and tools
model = load_model('final_emotion_model.keras')
scaler = pickle.load(open('scaler.pickle', 'rb'))
encoder = pickle.load(open('encoder.pickle', 'rb'))

def extract_features(file_path):
    """
    Extracts MFCC features from an audio file.
    Settings must match the training process exactly.
    """
    #Load audio file 
    data, sr = librosa.load(file_path, duration=2.5, offset=0.6)
    #Extract 30 MFCC features
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=30)
    #Calculate the mean across time for each MFCC coefficient
    result = np.mean(mfcc.T, axis=0)
    
    return result

def predict_emotion(audio_path):
    """
    Processes the audio and returns the predicted emotion and confidence.
    """
    # Step A: Feature extraction
    features = extract_features(audio_path)
    
    # Step B: Scaling (reshape to 2D for the scaler)
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    # Step C: Reshaping for 1D CNN (Add 3rd dimension: [batch, features, channel])
    features_cnn = np.expand_dims(features_scaled, axis=2)
    
    # Step D: Model prediction (probabilities)
    predictions = model.predict(features_cnn, verbose=0)
    
    # Step E: Inverse transform the label and get confidence score
    emotion_label = encoder.inverse_transform(predictions)[0][0]
    confidence_score = np.max(predictions)
    
    return emotion_label, confidence_score

# --- Main Execution ---
# Replace this with the path to the audio file you want to test
test_audio = "path_to_your_file.wav" 

try:
    label, score = predict_emotion(test_audio)
    print("-" * 30)
    print(f"File: {test_audio}")
    print(f"Predicted Emotion: {label}")
    print(f"Confidence: {score * 100:.2f}%")
    print("-" * 30)
except Exception as e:
    print(f"An error occurred: {e}")