import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from textblob import TextBlob  # For sentiment analysis
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define prompt for mental health support with empathy
prompt_template = """
You are a compassionate, empathetic, and supportive virtual assistant designed to provide mental health and emotional support to students.
Your responses should be kind, understanding, and helpful. 
If someone is feeling down, respond gently and acknowledge their feelings before offering guidance.

Here’s the student’s input:
{user_input}

Provide your response:
"""

# Initialize the Google Gemini AI model
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)

# Create a prompt template
prompt = PromptTemplate(template=prompt_template, input_variables=["user_input"])

# Define a chain using the model and the prompt
chain = LLMChain(llm=model, prompt=prompt)


# Function to analyze sentiment of the user's input
def analyze_sentiment(user_input):
    analysis = TextBlob(user_input)
    sentiment = analysis.sentiment.polarity
    return sentiment

# Function to provide supportive responses based on sentiment
def provide_supportive_response(user_input):
    # Analyze the sentiment of the input
    sentiment = analyze_sentiment(user_input)

    # Get the AI response
    ai_response = chain.run(user_input=user_input)
    
    # Modify response based on sentiment
    if sentiment < -0.3:  # Negative sentiment
        st.write("I sense you're feeling down. Remember, it's okay to feel this way, and I'm here to support you.")
    elif sentiment > 0.3:  # Positive sentiment
        st.write("It's great to hear that you're feeling positive! Keep it up!")
    
    return ai_response

# Main function to run the chatbot app
def main():
    st.set_page_config(page_title="Mental Health Support Assistant", page_icon=":brain:")
    st.header("Shanti.ai")
    st.header("Mental Health and Emotional Support Assistant 🤖 for Students")
    st.write("Welcome! I’m here to listen and provide emotional support. Feel free to share your thoughts.")

    # Get user input
    user_input = st.text_input("How are you feeling today?", "")
    
    if user_input:
        with st.spinner("I'm here for you..."):
            response = provide_supportive_response(user_input)
            st.write(response)
    st.write("---")
    st.markdown("""
        **Resources:**
        - [Breathing exercises](https://www.headspace.com/meditation/breathing-exercises)
        - [Mental health tips for students](https://www.nami.org)
        - [Guided Meditation](https://www.calm.com)
    """)

if __name__ == "__main__":
    main()