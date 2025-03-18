from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("❌ ERROR: GROQ_API_KEY is missing! Add it to your .env file.")
    exit()

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# Initialize Llama 3 Model via LangChain
llm = ChatGroq(model="llama3-8b-8192", groq_api_key=GROQ_API_KEY)

# Chat Memory for Session-Based Conversations
session_histories = {}

def get_session_history(session_id):
    if session_id not in session_histories:
        session_histories[session_id] = ChatMessageHistory()
    return session_histories[session_id]

# Define Prompt Template (Ensures UBI-Specific Answers)
ubi_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are UVA, Union Bank of India’s Virtual Assistant."),
    ("system", "You only answer questions about Union Bank of India (UBI). If the user asks something unrelated, politely refuse."),
    ("system", "Provide clear and professional banking-related answers."),
    ("human", "{user_input}"),
])

# Setup Memory for Context Retention
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create Runnable Chain with Session-based Memory
conversation = RunnableWithMessageHistory(
    ubi_prompt | llm,
    get_session_history=get_session_history,
    input_messages_key="user_input",
)

# ✅ **Predefined UBI Responses Without Asterisks**
UBI_RESPONSES = {
    "services": "Union Bank of India offers Savings & Current Accounts, Fixed Deposits, Loans, Credit Cards, Net Banking, and Investment Plans.",
    "loans": """UBI provides the following loan products:

    - Home Loans – Interest starts from 8.50% per annum
    - Personal Loans – Interest starts from 10.50% per annum
    - Education Loans – Interest starts from 9.75% per annum
    - Car Loans – Interest starts from 8.90% per annum
    - Business Loans – Interest depends on business type and credit score
    - MSME Loans – Special schemes for small enterprises
    - Agricultural Loans – Special rates based on government subsidies""",
    "home loan": "Home Loan interest starts from 8.50% per annum, based on credit score and tenure.",
    "personal loan": "Personal Loan interest starts from 10.50% per annum, with flexible repayment options.",
    "education loan": "Education Loan interest starts from 9.75% per annum, covering tuition and student expenses.",
    "car loan": "Car Loan interest starts from 8.90% per annum, applicable to new and used cars.",
    "business loan": "Business Loan interest depends on business type, credit score, and loan amount.",
    "msme loan": "MSME Loans have special rates for small enterprises under government schemes.",
    "agriculture loan": "Agricultural Loan interest rates are based on government subsidies and financing programs.",
    "interest rates": """Current interest rates for various loans:

    - Home Loan: 8.50% per annum
    - Personal Loan: 10.50% per annum
    - Education Loan: 9.75% per annum
    - Car Loan: 8.90% per annum
    - Business Loan: Custom rates based on eligibility
    - MSME Loan: Special government-backed rates
    - Agriculture Loan: Government-subsidi zed rates available""",
    "eligibility": "You can check your loan eligibility using the loan eligibility calculator below.",
    "customer care": "For assistance, contact Union Bank of India Customer Support at 1800 22 2244.",
    "internet banking": "UBI provides secure Net Banking and Mobile Banking services for fund transfers, bill payments, and more.",
}

# **Function to get AI Response (Only if No Predefined Answer)**
def get_llama3_response(user_message, session_id="default"):
    try:
        return conversation.invoke({"user_input": user_message}, config={"configurable": {"session_id": session_id}}).strip()
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return None

# **Chat API**
@app.route("/chat", methods=["POST"])
def chatbot_response():
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "Missing 'message' field"}), 400

        user_message = data["message"].strip().lower()
        print(f"🔹 Received user query: {user_message}")

        # Restrict inappropriate queries
        restricted_keywords = ["sex", "violence", "politics", "abuse", "terrorism", "hack", "scam"]
        if any(keyword in user_message for keyword in restricted_keywords):
            return jsonify({"response": "Please ask a valid banking-related query."})

        # **Check predefined UBI responses first**
        for key in UBI_RESPONSES:
            if key in user_message:
                print(f"✅ Answered from predefined responses: {UBI_RESPONSES[key][:100]}...")
                return jsonify({"response": UBI_RESPONSES[key]})

        # **Generate AI Response if No Predefined Answer**
        ai_response = get_llama3_response(user_message)
        if not ai_response:
            return jsonify({"response": "I'm here to assist with Union Bank of India's services. Ask me about loans, credit cards, or online banking."})

        print(f"✅ AI Response: {ai_response[:100]}...")
        return jsonify({"response": ai_response})

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return jsonify({"error": "Something went wrong!"}), 500

# **Welcome Message API**
@app.route("/welcome", methods=["GET"])
def welcome_message():
    return jsonify({"response": "UBI: Welcome! I am UVA (Union Bank’s Virtual Assistant), here to help you with loan products and services of Union Bank of India."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
