from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

app = Flask(__name__)
CORS(app)

# Initialize LangChain chatbot
llm = ChatOllama(
    model="glm-5:cloud",
    temperature=0.7
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

chain = prompt | llm | StrOutputParser()

# Store chat histories per session
chat_sessions = {}
max_turns = 10


def get_chat_response(session_id, question):
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []

    chat_history = chat_sessions[session_id]
    current_turn = len(chat_history) // 2

    if current_turn >= max_turns:
        return (
            "Context Window is full! "
            "The AI may not follow previous thread properly. "
            "Please click 'Clear' to start a new chat."
        ), False

    response = chain.invoke({
        "question": question,
        "chat_history": chat_history
    })

    # Fix: Store messages correctly
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response))

    remaining = max_turns - (current_turn + 1)
    warning = False
    if remaining <= 2:
        response += f"\n\n⚠️ Warning: Only {remaining} turn(s) left before context window is full."
        warning = True

    return response, warning


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get('message', '').strip()
    session_id = data.get('session_id', 'default')

    if not question:
        return jsonify({'error': 'No message provided'}), 400

    response, warning = get_chat_response(session_id, question)

    return jsonify({
        'response': response,
        'warning': warning,
        'turns_remaining': max_turns - (len(chat_sessions.get(session_id, [])) // 2)
    })


@app.route('/api/clear', methods=['POST'])
def clear():
    session_id = request.get_json().get('session_id', 'default')
    if session_id in chat_sessions:
        chat_sessions[session_id] = []
    return jsonify({'status': 'cleared'})


@app.route('/api/history', methods=['GET'])
def get_history():
    session_id = request.args.get('session_id', 'default')
    history = chat_sessions.get(session_id, [])
    messages = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            messages.append({'role': 'user', 'content': msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({'role': 'assistant', 'content': msg.content})
    return jsonify({'history': messages})


if __name__ == '__main__':
    print("Starting chatbot interface at http://localhost:5000")
    app.run(debug=True, port=5000)
