from langchain_ollama import ChatOllama
llm = ChatOllama(
    model ="glm-5:cloud",
    temprature=0.7
)
response = llm.invoke("What is Rag?")
print(response.content)