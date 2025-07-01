from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from phi.agent import Agent, RunResponse
from phi.tools.googlesearch import GoogleSearch
from phi.tools.yfinance import YFinanceTools
from phi.model.groq import Groq
import os
import time
import logging
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from config import Config

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins for debugging

# Set rate limiter
limiter = Limiter(get_remote_address, app=app)

# Verify Groq API key
if not os.environ.get("GROQ_API_KEY"):
    logger.error("GROQ_API_KEY is not set")
    raise EnvironmentError("GROQ_API_KEY environment variable is missing")

# Function to create agents
def create_sentiment_agent():
    logger.debug("Creating Sentiment Agent")
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
    logger.debug("Creating Finance Agent")
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
    logger.debug("Creating Analyst Agent")
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
    logger.debug("Creating Agent Team")
    try:
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
    except Exception as e:
        logger.error(f"Error creating agent team: {str(e)}")
        raise

# Retry logic for handling rate limits
def retry_with_backoff(agent_team, prompt, retries=5, backoff_factor=2):
    logger.debug(f"Attempting to run agent with prompt: {prompt[:50]}...")
    for attempt in range(retries):
        try:
            return agent_team.run(prompt, stream=True)
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if "429 Too Many Requests" in str(e) and attempt < retries - 1:
                wait_time = backoff_factor ** attempt
                logger.info(f"Rate limit reached. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise

# Health check route
@app.route('/health')
def health():
    logger.debug("Health check endpoint called")
    return jsonify({'status': 'healthy', 'message': 'Server is running'})

# Index route
@app.route('/')
def index():
    logger.debug("Index endpoint called")
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index.html: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Failed to load index page'}), 500

# Analyze route
@app.route('/analyze', methods=['POST'])
@limiter.limit("10 per minute")
def analyze():
    logger.debug(f"Analyze endpoint called with request: {request.json}")
    try:
        # Validate request JSON
        if not request.is_json:
            logger.error("Request does not contain valid JSON")
            return jsonify({
                'status': 'error',
                'message': 'Request must contain valid JSON'
            }), 400

        data = request.json
        required_fields = ['company1', 'company2', 'startDate', 'endDate']
        if not data or not all(key in data for key in required_fields):
            logger.error(f"Missing required fields: {required_fields}")
            return jsonify({
                'status': 'error',
                'message': f'Missing required fields: {", ".join(required_fields)}'
            }), 400

        company1 = data.get('company1')
        company2 = data.get('company2')
        start_date = data.get('startDate')
        end_date = data.get('endDate')

        logger.debug(f"Processing companies: {company1}, {company2} from {start_date} to {end_date}")

        # Create agent team
        agent_team = create_agent_team()

        # Construct prompt
        prompt = f"""
        Analyze the sentiment for the following companies during {start_date} to {end_date}: {company1}, {company2}.
        1. **Sentiment Analysis**: Search for relevant news and interpret the sentiment for each company. Provide sentiment scores on a scale of 1 to 10.
        2. **Financial Data**: Retrieve stock prices, analyst recommendations, and financial insights.
        3. **Consolidated Analysis**: Combine the insights from sentiment analysis and financial data to assign a final sentiment score (1-10) for each company. Justify the scores with reasoning and provide references.
        """

        # Run agent with retry logic
        response_stream = retry_with_backoff(agent_team, prompt)
        full_response = []

        # Collect streamed response
        for resp in response_stream:
            if isinstance(resp, RunResponse) and isinstance(resp.content, str):
                full_response.append(resp.content)
            else:
                logger.warning(f"Unexpected response part: {resp}")
                full_response.append(str(resp))  # Fallback to string representation

        complete_response = ''.join(full_response)
        if not complete_response:
            logger.error("No response content received from agent")
            return jsonify({
                'status': 'error',
                'message': 'No response content received from agent'
            }), 500

        logger.debug("Successfully generated response")
        return jsonify({
            'status': 'success',
            'response': complete_response
        })

    except Exception as e:
        logger.error(f"Error in /analyze: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }), 500

# Error handler for 404
@app.errorhandler(404)
def not_found(e):
    logger.error(f"404 error: {str(e)}")
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404

# Error handler for 405
@app.errorhandler(405)
def method_not_allowed(e):
    logger.error(f"405 error: {str(e)}")
    return jsonify({
        'status': 'error',
        'message': 'Method not allowed'
    }), 405

if __name__ == '__main__':
    logger.info("Starting Flask app in debug mode")
    app.run(debug=True)  # Debug mode for local testing only
