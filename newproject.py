import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from textblob import TextBlob
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Optimized prompt template
prompt_template = """
You are a supportive mental health assistant for students.
Provide concise and comforting responses with empathy.

User: {user_input}

Your Response:
"""

# Optimized model parameters
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, stream=True)

prompt = PromptTemplate(template=prompt_template, input_variables=["user_input"])
chain = LLMChain(llm=model, prompt=prompt)


def analyze_sentiment(user_input):
    """Performs sentiment analysis using TextBlob."""
    return TextBlob(user_input).sentiment.polarity


def provide_supportive_response(user_input):
    """Generates AI response and adds supportive messaging based on sentiment."""
    sentiment = analyze_sentiment(user_input)
    
    if sentiment < -0.3:
        st.write("💙 I sense you're feeling down. It's okay to feel this way, and I'm here to support you.")
    elif sentiment > 0.3:
        st.write("😊 It's great to hear you're feeling positive! Keep it up!")

    return chain.run(user_input=user_input)


def main():
    st.set_page_config(page_title="Shanti.ai", page_icon="🧘")
    st.title("Shanti.ai 🌿")
    st.subheader("Your AI Mental Health Companion for Students")
    st.write("Welcome! I'm here to listen and provide emotional support. Feel free to share your thoughts.")

    with st.form("chat_form"):
        user_input = st.text_input("How are you feeling today?", "")
        submit_button = st.form_submit_button("Send")

    if submit_button and user_input:
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            response = provide_supportive_response(user_input)
            st.write(response)

    st.write("---")
    st.markdown("""
        **Resources for Well-being:**
        - 🧘 [Breathing Exercises](https://www.headspace.com/meditation/breathing-exercises)
        - 💡 [Mental Health Tips](https://www.nami.org)
        - 🎵 [Guided Meditation](https://www.calm.com)
    """)


if __name__ == "__main__":
    main()
