from flask import Flask, render_template, request, jsonify
from phi.agent import Agent, RunResponse
from phi.tools.googlesearch import GoogleSearch
from phi.tools.yfinance import YFinanceTools
from phi.model.groq import Groq
import os
from config import Config
from phi.utils.pprint import pprint_run_response

app = Flask(__name__)
app.config.from_object(Config)
os.environ["GROQ_API_KEY"] = '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'

# Initialize agents
def create_sentiment_agent():
    return Agent(
        name="Sentiment Agent",
        role="Search and interpret news articles.",
        model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
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
        model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
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
        model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
        instructions=[
            "Check outputs for accuracy and completeness.",
            "Synthesize data to provide a final sentiment score (1-10) with justification",
            "Give also graph and chart by which the response will look justified.",
        ],
        show_tool_calls=True,
        markdown=True,
    )

def create_agent_team():
    sentiment_agent = create_sentiment_agent()
    finance_agent = create_finance_agent()
    analyst_agent = create_analyst_agent()
    
    return Agent(
        model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
        team=[sentiment_agent, finance_agent, analyst_agent],
        instructions=[
            "Combine the expertise of all agents to provide a cohesive, well-supported response.",
            "Always include references and dates for all data points and sources.",
            "Present all data in structured tables for clarity.",
            "Explain the methodology used to arrive at the sentiment scores",
            "Give also graph and chart by which the response will look justified.",
        ],
        show_tool_calls=True,
        markdown=True,
    )
from flask import Flask, render_template, request, jsonify
from phi.agent import Agent
from phi.tools.googlesearch import GoogleSearch
from phi.tools.yfinance import YFinanceTools
from phi.model.groq import Groq
import os
from config import Config
from phi.utils.pprint import pprint_run_response

app = Flask(__name__)
app.config.from_object(Config)
os.environ["GROQ_API_KEY"] = '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'

# Initialize agents
def create_sentiment_agent():
    return Agent(
        name="Sentiment Agent",
        role="Search and interpret news articles.",
        model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
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
        model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
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
        model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
        instructions=[
            "Check outputs for accuracy and completeness.",
            "Synthesize data to provide a final sentiment score (1-10) with justification",
            "Give also graph and chart by which the response will look justified.",
        ],
        show_tool_calls=True,
        markdown=True,
    )

def create_agent_team():
    sentiment_agent = create_sentiment_agent()
    finance_agent = create_finance_agent()
    analyst_agent = create_analyst_agent()
    
    return Agent(
        model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
        team=[sentiment_agent, finance_agent, analyst_agent],
        instructions=[
            "Combine the expertise of all agents to provide a cohesive, well-supported response.",
            "Always include references and dates for all data points and sources.",
            "Present all data in structured tables for clarity.",
            "Explain the methodology used to arrive at the sentiment scores",
            "Give also graph and chart by which the response will look justified.",
        ],
        show_tool_calls=True,
        markdown=True,
    )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    company1 = data.get('company1')
    company2 = data.get('company2')
    start_date = data.get('startDate')
    end_date = data.get('endDate')
    
    agent_team = create_agent_team()
    
    prompt = f"""Analyze the sentiment for the following companies during {start_date} to {end_date}: {company1}, {company2}.

1. **Sentiment Analysis**: Search for relevant news articles and interpret the sentiment for each company. Provide sentiment scores on a scale of 1 to 10, explain your reasoning, and cite your sources.

2. **Financial Data**: Analyze stock price movements, analyst recommendations, and any notable financial data. Highlight key trends or events, and present the data in tables.

3. **Consolidated Analysis**: Combine the insights from sentiment analysis and financial data to assign a final sentiment score (1-10) for each company. Justify the scores and provide a summary of the most important findings.

Tell me also that in which company I have to buy share for earning more and more profit.
Give also graph and chart by which the response will look justified."""

    try:
        response_stream = agent_team.run(prompt, stream=True)
        full_response = []
        
        # Collect streamed response
        for resp in response_stream:
            if isinstance(resp, RunResponse) and isinstance(resp.content, str):
                full_response.append(resp.content)
        
        # Join all response parts
        complete_response = ''.join(full_response)
        print("Complete response:", complete_response)  # Debug print
        
        return jsonify({
            'status': 'success',
            'response': complete_response
        })
         
    except Exception as e:
        print(f"Error occurred: {str(e)}")  # Debug print
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run()