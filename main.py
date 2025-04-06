from flask import Flask, redirect, url_for, render_template, jsonify, request
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import pymongo
from dotenv import load_dotenv
import os
from datetime import datetime
import json
from google import genai
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
import re

# chatbot_model_name = "microsoft/DialoGPT-medium"

# # Load tokenizer
# chatbot_tokenizer = AutoTokenizer.from_pretrained(chatbot_model_name)

# # Load model with specific parameters to avoid the error
# chatbot_model = AutoModelForCausalLM.from_pretrained(
#     chatbot_model_name,
#     device_map="auto",  # Automatically decide device placement
#     low_cpu_mem_usage=True,  # Optimize memory usage
#     # Add this if you still encounter issues
#     torch_dtype="auto"  # Use automatic type detection
# )

# def generate_chat_response(user_input, chat_history_ids=None):
#     new_user_input_ids = chatbot_tokenizer.encode(user_input + chatbot_tokenizer.eos_token, return_tensors="pt")
#     bot_input_ids = new_user_input_ids if chat_history_ids is None else torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
#     chat_history_ids = chatbot_model.generate(bot_input_ids, max_length=1000, pad_token_id=chatbot_tokenizer.eos_token_id)
#     output = chatbot_tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
#     return output, chat_history_ids

load_dotenv()

app = Flask(__name__)

app.secret_key = os.getenv("SECRET_KEY", "your_secret_key")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY) 
db = pymongo.MongoClient(os.getenv("MONGO_URI")).get_database("stock_data")

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
login_manager.session_protection = "strong"
login_manager.login_message = "Please log in to access this page."
login_manager.login_message_category = "info"

class User(UserMixin):
    def __init__(self, id):
        self.id = id

    def get_id(self):
        return self.id
        
@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        
        user_doc = db.users.find_one({"username": username})
        if user_doc and user_doc["password"] == password:
            user = User(username)
            login_user(user)
            return redirect(url_for("mystockchat"))
        else:
            return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        
        if db.users.find_one({"username": username}):
            return render_template("register.html", error="Username already exists")
        
        db.users.insert_one({"username": username, "password": password})
        return redirect(url_for("login"))
    
    return render_template("register.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))

@app.route("/chat")
@login_required
def mystockchat():
    return render_template("chat.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    if not current_user.is_authenticated:
        return jsonify({"error": "User not authenticated"}), 401

    data = request.get_json()
    user_message = data.get('message', '')
    chat_history_ids = data.get('chat_history_ids', None)

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    stock_keywords = ["stock", "market", "price", "shares", "company", "investment"]
    is_stock_related = any(keyword in user_message.lower() for keyword in stock_keywords)

    if is_stock_related:
        # Improved company name extraction
        # First check for simple patterns like "Show me [Company] stock"
        company_match = re.search(r"(?:show me|get|display|find|lookup)\s+([A-Za-z\s]+)(?:\s+stock|\s+price|\s+shares|\s+company)?", user_message, re.IGNORECASE)
        
        # If not found, check for other patterns
        if not company_match:
            company_match = re.search(r"(?:about|for|on)\s+([A-Za-z\s]+)(?:\s+stock|\s+price|\s+shares|\s+company)?", user_message, re.IGNORECASE)
        
        # If still not found, look for any capitalized words that might be company names
        if not company_match:
            words = re.findall(r'\b[A-Z][a-z]+\b', user_message)
            company_name = ' '.join(words) if words else None
        else:
            company_name = company_match.group(1).strip()
            
        # Fallback if no company detected
        if not company_name:
            return jsonify({
                "response": "I'd be happy to provide stock information. Could you specify which company you're interested in?",
                "chat_history_ids": None
            })

        prompt = f"You are a financial assistant.\n" \
             f"Provide precise and concise stock information for {company_name}:\n" \
             f"- Current price\n" \
             f"- Market cap\n" \
             f"- Recent performance\n" \
             f"- Key financial metrics\n" \
             f"- Analyst ratings\n" \
             f"- Major news headlines\n" \
             f"Keep the response short and to the point."

        try:
            gemini_response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[{"role": "user", "parts": [{"text": prompt}]}]
            )

            if gemini_response and gemini_response.candidates:
                reply_text = gemini_response.candidates[0].content.parts[0].text.strip()
                return jsonify({
                    "response": reply_text,
                    "chat_history_ids": None,
                    "suggestion": f"Want full data report on {company_name}?",
                    "suggestion_action": {
                        "type": "stock_data",
                        "company": company_name
                    }
                })
            else:
                return jsonify({"error": "No content generated by Gemini"}), 500

        except Exception as e:
            return jsonify({"error": f"Gemini API failed: {str(e)}"}), 500

    # # Else: Use DialoGPT for general questions
    # if chat_history_ids:
    #     chat_history_ids = torch.tensor(chat_history_ids)

    # response, history_ids = generate_chat_response(user_message, chat_history_ids)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[{"role": "user", "parts": [{"text": user_message}]}]
    )
    if not response:
        return jsonify({"error": "No response from the model"}), 500
    suggestion = generate_suggestion(user_message, response)

    return jsonify({
        "response": response,
        "chat_history_ids": None,
        "suggestion": suggestion["text"] if suggestion else None,
        "suggestion_action": suggestion["action"] if suggestion else None
    })

@app.route("/api/request/gemini")
def request_stock_data():
    if not current_user.is_authenticated:
        return jsonify({"error": "User not authenticated"}), 401
    company_name = request.args.get('company_name', '')
    
    if not company_name:
        return jsonify({"error": "Company name is required"}), 400
    
    current_year = datetime.now().year

    prompt = f"""
    Provide comprehensive stock market data for {company_name} for year {current_year} in a strictly formatted JSON structure. 
    Include the following details:
    1. Basic Company Information
    2. Current Stock Performance
    3. Financial Metrics
    4. Market Classification
    5. Historical Performance
    6. Analyst Ratings
    7. Recent News Articles
    8. Social Media Sentiment
    9. Insider Ownership
    10. Upcoming Festivals
    11. Upcoming Earnings Reports
    12. Upcoming Events
    13. Analyst Comments

    JSON Structure:
    {{
        "companyName": "Full Company Name",
        "stockSymbol": "Stock Exchange Symbol",
        "currentPrice": 0.00,
        "marketCap": "Market Capitalization",
        "priceChange": {{
            "amount": 0.00,
            "percentage": 0.00
        }},
        "industryCategory": "Primary Industry Category",
        "sector": "Broader Economic Sector",
        "exchange": "Primary Stock Exchange",
        "financialMetrics": {{
            "peRatio": 0.00,
            "dividendYield": 0.00,
            "earningsPerShare": 0.00,
            "52WeekHigh": 0.00,
            "52WeekLow": 0.00
        }},
        "historicalPerformance": {{
            "yearToDate": 0.00,
            "lastYear": 0.00
        }},
        "analystRatings": {{
            "buy": 0,
            "hold": 0,
            "sell": 0
        }},
        "news": [{{
            "headline": "Latest news headline",
            "url": "Link to article",
            "date": "YYYY-MM-DD"
        }}],
        "socialSentiment": {{
            "positive": 0.00,
            "neutral": 0.00,
            "negative": 0.00
        }},
        "insiderOwnership": 0.00,
        "upcomingFestivals": [{{
            "festivalName": "Name of the festival",
            "date": "YYYY-MM-DD",
            "impact": "Expected impact on stock"
        }}],
        "upcomingEarnings": {{
            "date": "YYYY-MM-DD",
            "estimatedEarnings": 0.00
        }},
        "upcomingEvents": [{{
            "eventName": "Name of the event",
            "date": "YYYY-MM-DD",
            "impact": "Expected impact on stock"
        }}],
        "analystComments": [{{
            "comment": "Analyst comment",
            "analystName": "Analyst name",
            "date": "YYYY-MM-DD"
        }}],
    }}

    Instructions:
    - Use most recent available data
    - Provide numeric values where applicable
    - Be precise and factual
    - If any data is unavailable, use null
    """

    try:
        response = client.models.generate_content(model="gemini-2.0-flash", contents=[{"role": "user", "parts": [{"text": prompt}]}])
        if not response:
            return jsonify({"error": "No response from the model"}), 500
        if hasattr(response, 'error') and response.error:
            return jsonify({"error": f"Model error: {response.error}"}), 500
        if hasattr(response, 'candidates') and not response.candidates:
            return jsonify({"error": "No candidates in response"}), 500
        if hasattr(response, 'candidates') and not response.candidates[0].content:
            return jsonify({"error": "No content in candidate"}), 500
        if hasattr(response.candidates[0], 'error') and response.candidates[0].error:
            return jsonify({"error": f"Candidate error: {response.candidates[0].error}"}), 500
        if response and response.candidates:
            response_text = response.candidates[0].content.parts[0].text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:-3].strip()  # Remove ```json wrapper
            stock_data = json.loads(response_text)
            return jsonify(stock_data)
        else:
            return jsonify({"error": "No content generated by the model"}), 500
        
    except KeyError as e:
        return jsonify({"error": f"Response parsing failed: {str(e)}"}), 500
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Invalid JSON in response: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

def generate_suggestion(user_message, bot_response):
    """Generate contextual suggestions based on conversation"""
    # Define topic patterns to recognize
    topics = {
        "travel": ["vacation", "trip", "travel", "flight", "hotel", "destination"],
        "food": ["recipe", "restaurant", "cooking", "meal", "food", "dish", "cuisine"],
        "tech": ["computer", "software", "hardware", "app", "technology", "device", "programming"],
        "health": ["exercise", "diet", "fitness", "health", "medical", "wellness"],
        "finance": ["money", "investment", "budget", "finance", "saving", "stock", "market"]
    }
    
    # Detect topics in conversation
    detected_topics = []
    combined_text = (user_message + " " + bot_response).lower()
    
    for topic, keywords in topics.items():
        if any(keyword in combined_text for keyword in keywords):
            detected_topics.append(topic)
    
    if not detected_topics:
        return None
    
    # Generate relevant suggestion based on topic
    primary_topic = detected_topics[0]
    
    suggestions = {
        "travel": {
            "text": "Would you like to explore popular destinations for your next trip?",
            "action": {"type": "explore", "category": "travel_destinations"}
        },
        "food": {
            "text": "Would you like to see some popular recipes you might enjoy?",
            "action": {"type": "explore", "category": "recipes"}
        },
        "tech": {
            "text": "Would you like to learn about the latest tech trends?",
            "action": {"type": "explore", "category": "tech_news"}
        },
        "health": {
            "text": "Would you like to see some wellness tips that might help you?",
            "action": {"type": "explore", "category": "wellness_tips"}
        },
        "finance": {
            "text": "Would you like to learn more about investment strategies?",
            "action": {"type": "explore", "category": "investment_guides"}
        }
    }
    
    return suggestions.get(primary_topic)

if __name__ == "__main__":
    app.run(debug=True, port=80)