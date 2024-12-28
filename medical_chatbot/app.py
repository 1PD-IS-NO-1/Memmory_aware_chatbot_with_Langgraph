from flask import Flask, render_template, request, jsonify
from chatbot import ChatBot
import os

app = Flask(__name__)

# Initialize chatbot
chatbot = ChatBot()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        query = data.get('query')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        response = chatbot.ask(query)
        return jsonify({'response': response})
    
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)