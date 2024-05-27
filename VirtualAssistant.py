import speech_recognition as sr
import pyttsx3
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Load intent classification data
data = pd.read_csv('intent_classification_data.csv')

# Create a Naive Bayes Classifier for intent classification
vectorized = CountVectorizer()
X = vectorized.fit_transform(data['text'])
y = data['intent']
clf = MultinomialNB()
clf.fit(X, y)


# Define a function to classify user input
def classify_intent(text):
    text_vector = vectorized.transform([text])
    intent = clf.predict(text_vector)[0]
    return intent


# Create a speech recognition object
r = sr.Recognizer()

# Create a text-to-speech object
engine = pyttsx3.init()
voices = engine.getProperty("voices")
engine.setProperty("voice", voices)
engine.say("Hello World!")
engine.runAndWait()


# Define a function to listen and respond
def listen():
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            intent = classify_intent(text)
            print("You said: " + text)
            print("With intent " + intent)
        except sr.UnknownValueError:
            print("Sorry, I didn't understand that.")


# Define a function to respond to user input based on intent
def respond(intent):
    engine.runAndWait()
    if intent == 'greeting':
        greeting = "Hello! How can I assist you today?"
        engine.say(greeting)
        print(greeting)
    elif intent == 'farewell':
        farewell = "Alright, see you next time!"
        engine.say(farewell)
        print(farewell)


# Start listening and responding
listen()
respond(classify_intent(listen()))
