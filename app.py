# import libraries
import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

# langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"] = 'true'
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with OpenAI"

# prompt template design
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please response to the user queries"),
        ("user", "Question: {question}")
    ]
)

# generate response
def generate_response(question, api_key, engine, temperature, max_tokens):
    llm = ChatOpenAI(
        model = engine,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key= api_key
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

# streamlit web app
## title of the app
st.title("Q&A Chatbot With OpenAI")

## sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Open AI key: ", type='password')

## select the openAI model
engine = st.sidebar.selectbox("Select open AI model", ["gpt-4o", "gpt-4-turbo", "gpt-4"])

## adjust response parameter
# temperature = st.sidebar("Temperature", min_value=0.0, max_value=1.0, value=0.7)
temperature = st.sidebar.number_input(
    "Temperature", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.7
)

max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

## main interface for user input
st.write("Ask any Question")
user_input = st.text_input("You: ")

if user_input and api_key:
    response = generate_response(user_input, api_key, engine, temperature, max_tokens)
    st.write(response)

elif user_input:
    st.write("Please enter the OpenAI api key in the sidebar")

else:
    st.write("Please provide the user input")