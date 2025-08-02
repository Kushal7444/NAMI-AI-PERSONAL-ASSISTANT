import json
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load intents file
with open("intents.json") as file:
    data = json.load(file)

# Load trained model
model = load_model("chat_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load label encoder
with open("label_encoder.pkl", "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Chat loop
while True:
    input_text = input("Enter your command -> ")

    # Optional exit condition
    if input_text.lower() in ["quit", "exit", "bye"]:
        print("Goodbye!")
        break

    # Convert text to padded sequence
    padded_sequences = pad_sequences(
        tokenizer.texts_to_sequences([input_text]),
        maxlen=20,
        truncating='post'
    )

    # Predict intent
    result = model.predict(padded_sequences)
    tag = label_encoder.inverse_transform([np.argmax(result)])

    # Find matching intent and respond
    for intent in data['intents']:
        if intent['tag'] == tag[0]:
            print(np.random.choice(intent['responses']))  # Fixed comment
