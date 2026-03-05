from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(
    model = "qwen3.5:cloud",
    temperature = 0.5
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI Engineer and Teacher"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

chain = prompt | llm | StrOutputParser()

chat_history = []

def chat(question):
    response = chat.invoke(
        {"question": question,
         "chat_history", chat_history
    })

    chat_history.append(AIMessage.content("response"))
    chat_history.append(HumanMessage.content=question)

    return response

print(chat("what are skills?"))