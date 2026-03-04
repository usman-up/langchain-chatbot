from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(
    model ="glm-5:cloud",
    temprature=0.7
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
    ])

chain = prompt | llm | StrOutputParser()

chat_history = []

def chat(question):
    response = chain.invoke(
        {
            "question": question,
            "chat_history": chat_history
        }
    )
    
    chat_history.append(AIMessage(content="response"))
    chat_history.append(HumanMessage(content=question))
    
    return response


print(chat("What is Rag?"))
print(chat("give me a Python example of it"))
print(chat("Now explain the code you just gave"))




#for chunk in chain.stream({"question": "What is RAG?"}):
#    print(chunk, end="", flush=True)