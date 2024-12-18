import json
import random
import pickle
import numpy as np
import tensorflow as tf
import os
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('punkt')

lemmatizer = WordNetLemmatizer()

# Load CSV data
def load_intents_from_csv(csv_path):
    data = pd.read_csv(csv_path)
    intents = {}
    for _, row in data.iterrows():
        tag = row['tag']
        if tag not in intents:
            intents[tag] = {
                "responses": [],
                "follow_up": [],
                "questions": []
            }
        intents[tag]["responses"].append(row['response'])
        if pd.notna(row['follow_up_question']):
            intents[tag]["questions"].append({
                'question': row['follow_up_question'],
                'key': row['follow_up_key']
            })
    return intents

# Construct the path to your CSV file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
intents_csv_path = os.path.join(BASE_DIR, 'MediGuide(bot) (2).csv')

# Load the intents from the CSV file
intents_data = load_intents_from_csv(intents_csv_path)

# Function to clean the user input
def clean_up_sentence(sentence):
    sentence_words = sentence.split()
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Function to create bag of words
def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

# Function to predict class (with basic example)
def predict_class(user_input):
    intents_list = []
    if "hello" in user_input.lower():
        intents_list.append({"intent": "greeting"})
    return intents_list

# Function to get the response based on intents
def get_response(intents_list, intents_data):
    global context, user_data
    
    if not intents_list:
        return "Sorry, I didn't understand that. Can you rephrase?"

    tag = intents_list[0]['intent']
    intent_data = intents_data.get(tag, None)
    
    if intent_data and 'responses' in intent_data:
        response_data = random.choice(intent_data['responses'])
        if 'questions' in intent_data and intent_data['questions']:
            question_data = intent_data['questions'][0]
            question = question_data['question']
            key = question_data['key']
            context = {"tag": tag, "follow_up": question_data, "questions": intent_data['questions']}
            return response_data + " " + question
        
        return response_data
    
    return "Sorry, I don't have an answer for that."

def handle_follow_up():
    global context, user_data

    if not context:
        return "Sorry, I lost track of the conversation. Can we start over?"

    questions = context.get('questions', [])
    if questions:
        for question_data in questions:
            key = question_data['key']
            if key not in user_data:
                question = question_data['question']
                print(question)
                user_response = input("Your response: ")
                user_data[key] = user_response
                if len(user_data) < len(questions):
                    continue
                else:
                    return suggest_remedy()

    context = None
    return suggest_remedy()

def suggest_remedy():
    global user_data

    symptom = context['tag']
    age = user_data.get('age', 'unknown')
    duration = user_data.get('duration', 'unknown')
    bp = user_data.get('bp', 'unknown')

    # Check for additional logic here if needed (e.g., age-based conditions)
    remedies = {
        "body_pain": "Try taking rest and applying a warm compress. If the pain persists, please consult a doctor.",
        "cold_cough": "Drink ginger tea and stay hydrated. If it persists, please see a doctor.",
    }

    remedy = remedies.get(symptom, "I'm not sure about the remedy for that symptom. Please consult a doctor.")
    return f"Thank you for sharing the details. {remedy}"

# MAIN CHAT LOOP
print("Chatbot is running! Type 'quit' or 'exit' to end the chat.")
while True:
    message = input("Enter your message: ")
    if message.lower() in ["quit", "exit"]:
        print("Goodbye! Take care!")
        break
    
    intents = predict_class(message)
    response = get_response(intents, intents_data)
    print(response)
