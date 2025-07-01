from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from phi.agent import Agent, RunResponse
from phi.tools.googlesearch import GoogleSearch
from phi.tools.yfinance import YFinanceTools
from phi.model.groq import Groq
import os
import time
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from config import Config
from phi.utils.pprint import pprint_run_response

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)
CORS(app)  # Enable CORS for all routes

# Set rate limiter
limiter = Limiter(get_remote_address, app=app)

# Set Groq API key (ensure this is set in Render's environment variables)
os.environ["GROQ_API_KEY"] = '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'

# Function to create agents
def create_sentiment_agent():
    return Agent(
        name="Sentiment Agent",
        role="Search and interpret news articles.",
        model=Groq(id="llama-3.3-70b-versatile"),
        tools=[GoogleSearch()],
        instructions=[
            "Find relevant news articles for each company and analyze the sentiment.",
            "Provide sentiment scores from 1 (negative) to 10 (positive) with reasoning and sources.",
            "Cite your sources. Be specific and provide links.",
        ],
        show_tool_calls=True,
        markdown=True,
    )

def create_finance_agent():
    return Agent(
        name="Finance Agent",
        role="Get financial data and interpret trends.",
        model=Groq(id="llama-3.3-70b-versatile"),
        tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
        instructions=[
            "Retrieve stock prices, analyst recommendations, and key financial data.",
            "Focus on trends and present the data in tables with key insights.",
        ],
        show_tool_calls=True,
        markdown=True,
    )

def create_analyst_agent():
    return Agent(
        name="Analyst Agent",
        role="Ensure thoroughness and draw conclusions.",
        model=Groq(id="llama-3.3-70b-versatile"),
        instructions=[
            "Check outputs for accuracy and completeness.",
            "Synthesize data to provide a final sentiment score (1-10) with justification.",
            "Include graphs and charts to justify responses.",
        ],
        show_tool_calls=True,
        markdown=True,
    )

def create_agent_team():
    sentiment_agent = create_sentiment_agent()
    finance_agent = create_finance_agent()
    analyst_agent = create_analyst_agent()
    return Agent(
        model=Groq(id="llama-3.3-70b-versatile"),
        team=[sentiment_agent, finance_agent, analyst_agent],
        instructions=[
            "Combine the expertise of all agents to provide a cohesive, well-supported response.",
            "Always include references and dates for all data points and sources.",
            "Present all data in structured tables for clarity.",
            "Explain the methodology used to arrive at the sentiment scores.",
        ],
        show_tool_calls=True,
        markdown=True,
    )

# Retry logic for handling rate limits
def retry_with_backoff(agent_team, prompt, retries=5, backoff_factor=2):
    for attempt in range(retries):
        try:
            return agent_team.run(prompt, stream=True)
        except Exception as e:
            if "429 Too Many Requests" in str(e) and attempt < retries - 1:
                wait_time = backoff_factor ** attempt
                print(f"Rate limit reached. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
@limiter.limit("10 per minute")  # Increased limit for testing
def analyze():
    try:
        data = request.json
        if not data or not all(key in data for key in ['company1', 'company2', 'startDate', 'endDate']):
            return jsonify({
                'status': 'error',
                'message': 'Missing required fields: company1, company2, startDate, endDate'
            }), 400

        company1 = data.get('company1')
        company2 = data.get('company2')
        start_date = data.get('startDate')
        end_date = data.get('endDate')

        agent_team = create_agent_team()

        prompt = f"""
        Analyze the sentiment for the following companies during {start_date} to {end_date}: {company1}, {company2}.
        1. **Sentiment Analysis**: Search for relevant news and interpret the sentiment for each company. Provide sentiment scores on a scale of 1 to 10.
        2. **Financial Data**: Retrieve stock prices, analyst recommendations, and financial insights.
        3. **Consolidated Analysis**: Combine the insights from sentiment analysis and financial data to assign a final sentiment score (1-10) for each company. Justify the scores with reasoning and provide references.
        """

        response_stream = retry_with_backoff(agent_team, prompt)
        full_response = []

        for resp in response_stream:
            if isinstance(resp, RunResponse) and isinstance(resp.content, str):
                full_response.append(resp.content)
            else:
                print(f"Unexpected response part: {resp}")

        complete_response = ''.join(full_response)
        if not complete_response:
            return jsonify({
                'status': 'error',
                'message': 'No response content received from agent'
            }), 500

        return jsonify({
            'status': 'success',
            'response': complete_response
        })

    except Exception as e:
        print(f"Error in /analyze: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }), 500

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)  # Debug mode for local testing only
