from flask import Flask, render_template, request, jsonify
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

# Set rate limiter
limiter = Limiter(get_remote_address, app=app)

# Set Groq API key
os.environ["GROQ_API_KEY"] = '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'

# Custom error handlers to return JSON instead of HTML
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

@app.errorhandler(429)
def rate_limit_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Rate limit exceeded. Please try again later.'
    }), 429

# Handle rate limit exceeded from flask-limiter
@limiter.error_handler
def rate_limit_handler(e):
    return jsonify({
        'status': 'error',
        'message': f'Rate limit exceeded: {e.description}'
    }), 429

# Function to create agents
def create_sentiment_agent():
    return Agent(
        name="Sentiment Agent",
        role="Search and interpret news articles.",
        model=Groq(id="llama3-70b-8192"),
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
        model=Groq(id="llama3-70b-8192"),
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
        model=Groq(id="llama3-70b-8192"),
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
        model=Groq(id="llama3-70b-8192"),
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
def retry_with_backoff(agent_team, prompt, retries=3, backoff_factor=2):
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
@limiter.limit("5 per minute")  # Limit requests to 5 per minute per IP
def analyze():
    try:
        # Validate request content type
        if not request.is_json:
            return jsonify({
                'status': 'error',
                'message': 'Request must be JSON'
            }), 400
        
        data = request.json
        
        # Validate required fields
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400
        
        company1 = data.get('company1')
        company2 = data.get('company2')
        start_date = data.get('startDate')
        end_date = data.get('endDate')
        
        # Validate required fields
        if not all([company1, company2, start_date, end_date]):
            return jsonify({
                'status': 'error',
                'message': 'Missing required fields: company1, company2, startDate, endDate'
            }), 400

        agent_team = create_agent_team()

        prompt = f"""
        Analyze the sentiment for the following companies during {start_date} to {end_date}: {company1}, {company2}.

        1. **Sentiment Analysis**: Search for relevant news and interpret the sentiment for each company. Provide sentiment scores on a scale of 1 to 10.

        2. **Financial Data**: Retrieve stock prices, analyst recommendations, and financial insights.

        3. **Consolidated Analysis**: Combine the insights from sentiment analysis and financial data to assign a final sentiment score (1-10) for each company. Justify the scores with reasoning and provide references.
        """

        response_stream = retry_with_backoff(agent_team, prompt)
        full_response = []

        # Collect streamed response
        for resp in response_stream:
            if isinstance(resp, RunResponse) and isinstance(resp.content, str):
                full_response.append(resp.content)

        # Join all response parts
        complete_response = ''.join(full_response)
        
        # Ensure we have a response
        if not complete_response:
            return jsonify({
                'status': 'error',
                'message': 'No response generated from agents'
            }), 500

        return jsonify({
            'status': 'success',
            'response': complete_response
        })

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        # Always return JSON, never let Flask return HTML error pages
        return jsonify({
            'status': 'error',
            'message': f'An error occurred: {str(e)}'
        }), 500

# Additional route to handle OPTIONS requests (for CORS preflight)
@app.route('/analyze', methods=['OPTIONS'])
def analyze_options():
    return jsonify({'status': 'ok'}), 200

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)  # Added debug=True for development
