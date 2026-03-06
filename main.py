from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv

load_dotenv() #read .env variables

MODEL_NAME = os.getenv("MODEL_NAME", "glm-5:cloud")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TURNS = int(os.getenv("MAX_TURNS", "10"))

llm = ChatOllama(
    model = MODEL_NAME,
    temprature= TEMPERATURE
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
    ])

chain = prompt | llm | StrOutputParser()

chat_history = []   #memory store
# max_turns = 10 #10 messages (Human + AI)

def chat(question):
    current_turn = len(chat_history) // 2

    if current_turn >= MAX_TURNS:
        return(
            "Context Windos is full!"
            "The AI may not follow previous thread properly"
            "Please type 'clear' for new chat"
        )


    response = chain.invoke(
        {
            "question": question,
            "chat_history": chat_history
        }
    )
    
    chat_history.append(AIMessage(content="response"))
    chat_history.append(HumanMessage(content=question))

    remaining = max_turns - (current_turn + 1)
    if remaining <= 2:
        response += f"Warning: Only {remaining} turn(s) left"
    

    return response

def main():
    print("LangChain Chatbot Ready! (Type 'quit' for exit, 'clear' for reset chat history)")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "clear":
            chat_history.clear()
            print("History cleard, Starting freshi!")
            continue

        print(f"AI: {chat(user_input)}")

main()


#print(chat("What is Rag?"))
#print(chat("give me a Python example of it"))
#print(chat("Now explain the code you just gave"))




#for chunk in chain.stream({"question": "What is RAG?"}):
#    print(chunk, end="", flush=True)