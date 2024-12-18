import json
import random
import pickle
import numpy as np
import tensorflow as tf
import os
import nltk

nltk.download('wordnet')
nltk.download('punkt')

from nltk.stem import WordNetLemmatizer

# import google.generativeai as genai

# # Configure Gemini API Key
# genai.configure(api_key="AIzaSyAM9kVJyXw0nxpkUGvOYpXeqYzXQhz69go")

# def get_response_gemini(user_input):
#     try:
#         model = genai.GenerativeModel("gemini-pro")  # Use "gemini-pro" model
#         response = model.generate_content(user_input)
#         return response.text.strip()
#     except Exception as e:
#         print("Error with Gemini API:", e)
#         return "Sorry, I am having some technical difficulties. Please try again later."

# def get_response(intents_list, intents_json, user_input):
#     if not intents_list:
#         return get_response_gemini(user_input)
    
#     # Fallback to Gemini if no matching intents are found
#     tag = intents_list[0]['intent']
#     intent_data = next((i for i in intents_json['intents'] if i['tag'] == tag), None)
    
#     if intent_data and 'responses' in intent_data:
#         return random.choice(intent_data['responses'])
#     return get_response_gemini(user_input)


lemmatizer = WordNetLemmatizer()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHATBOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

intents_path = os.path.join(CHATBOT_DIR, 'prepared', 'intents.json')
words_path = os.path.join(CHATBOT_DIR, 'prepared', 'words.pkl')
classes_path = os.path.join(CHATBOT_DIR, 'prepared', 'classes.pkl')
model_path = os.path.join(CHATBOT_DIR, 'prepared', 'chatbot_model.h5')

intents_json = json.loads(open(intents_path).read())
words = pickle.load(open(words_path, 'rb'))
classes = pickle.load(open(classes_path, 'rb'))
model = tf.keras.models.load_model(model_path)

context = None
user_data = {}

def clean_up_sentence(sentence):
    sentence_words = sentence.split()
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow_input = bow(sentence, words)
    res = model.predict(np.array([bow_input]), verbose=0)
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res[0]) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    global context

    if context:
        return handle_follow_up()

    if not intents_list:
        return "Sorry, I didn't understand that. Can you rephrase?"

    tag = intents_list[0]['intent']
    intent_data = next((i for i in intents_json['intents'] if i['tag'] == tag), None)

    if intent_data and 'responses' in intent_data:
        response_data = random.choice(intent_data['responses'])
        if isinstance(response_data, dict):
            follow_up = response_data.get('follow_up', {})
            if follow_up:
                context = {"tag": tag, "follow_up": follow_up, "questions": follow_up['questions']}
                return response_data['response']
        return response_data

    return "Sorry, I don't have an answer for that."

def handle_follow_up():
    global context, user_data

    if not context:
        return "Sorry, I lost track of the conversation. Can we start over?"

    questions = context.get('questions', [])

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
    
    if symptom == "body_pain" or symptom == "cold_cough":
        age = int(user_data.get('age', 0))
        bp = user_data.get('bp', 'unknown').lower()
        diabetes = user_data.get('diabetes', 'no').lower()

        age_conditions = next(
            (i['logic']['age_conditions'] for i in intents_json['intents'] if i['tag'] == symptom), 
            None
        )

        if age_conditions:
            for condition in age_conditions:
                age_range = condition['range']
                if age >= age_range[0] and age <= age_range[1]:
                    if 'bp_check' in condition:
                        bp_check = condition['bp_check']
                        if bp in bp_check:
                            return bp_check[bp]
                        elif bp == "normal" and 'normal' in bp_check:
                            diabetes_check = bp_check['normal']['diabetes']
                            return diabetes_check.get(diabetes, "Please consult a doctor for more advice.")

        return "I'm sorry, I couldn't determine. Please consult a doctor."
    
    # elif symptom == "cold_cough" or symptom == "":
    #     age = int(user_data.get('age', 0))
    #     bp = user_data.get('bp', 'unknown').lower()
    #     diabetes = user_data.get('diabetes', 'no').lower()

    #     age_conditions = next(
    #         (i['logic']['age_conditions'] for i in intents_json['intents'] if i['tag'] == symptom), 
    #         None
    #     )

    #     if age_conditions:
    #         for condition in age_conditions:
    #             age_range = condition['range']
    #             if age >= age_range[0] and age <= age_range[1]:
    #                 if 'bp_check' in condition:
    #                     bp_check = condition['bp_check']
    #                     if bp in bp_check:
    #                         return bp_check[bp]
    #                     elif bp == "normal" and 'normal' in bp_check:
    #                         diabetes_check = bp_check['normal']['diabetes']
    #                         return diabetes_check.get(diabetes, "Please consult a doctor for more advice.")

    #     return "I'm sorry, I couldn't determine a specific remedy for your cough and cold. Please consult a doctor."

    remedies = {
        "body_pain": "Based on your details, I suggest resting, staying hydrated, and applying a warm compress to the affected area. If the pain persists, please consult a doctor.",
        "fever": "For fever, I recommend extracting juice from tulsi leaves and bel flowers, adding honeyheylo, and taking it twice a day. Also, drink plenty of fluids.",
        "headache": "For headaches, try sniffing roasted ajwain or applying a paste of powdered cloves on your forehead.",
        "cold_cough": "For a cold, consider taking ginger decoction and staying hydrated. Honey can also help soothe your throat.",
        "anaemia": "For anemia, I suggest eating iron-rich foods like spinach, lentils, and beans. You can also consider taking iron supplements.",
        "stress_relief": "To relieve stress, try deep breathing exercises, meditation, or yoga. You can also consider taking a break and going for a walk.",
        "chest_congestion": "For chest congestion, try steam inhalation with eucalyptus oil. Drinking warm water can also help clear the congestion.",
        "dehydration": "For dehydration, drink plenty of fluids like water, coconut water, and ORS. Avoid caffeinated and alcoholic beverages.",
        "diabetes": "For diabetes, maintain a healthy diet, exercise regularly, and monitor your blood sugar levels. Consult a doctor for medication.",
        "acidity": "For acidity, avoid spicy and oily foods. Drink cold milk or water with a teaspoon of ghee to reduce acidity."
    }

    remedy = remedies.get(symptom, "I'm not sure about the remedy for that symptom. Please consult a doctor.")

    return f"Thank you for sharing the details. {remedy}"


print("Go! Bot is running!")

while True:
    message = input("Enter your message: ")
    if message.lower() in ["quit", "exit"]:
        print("Goodbye! Take care!")    
        break
    intents = predict_class(message)
    response = get_response(intents, intents_json)
    print(response)
