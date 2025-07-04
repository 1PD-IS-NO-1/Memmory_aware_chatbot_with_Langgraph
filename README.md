# Flask Chatbot with Memory
flask_chatbot/
│
├── app.py                 # Main Flask application
├── templates/
│   ├── index.html         # Frontend HTML file
├── static/
│   ├── css/
│   │   ├── style.css      # CSS for styling
│   ├── js/
│       ├── chatbot.js     # JavaScript for handling UI interactions
├── requirements.txt       # Python dependencies
└── README.md              # Project description

## Setup

1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Run the Flask app: `python app.py`.
4. Open the browser and navigate to `http://127.0.0.1:5000`.

Enjoy chatting with your AI-powered chatbot!


## chatbot UI

![As we can see that chatbot is using memory](image.png)

## Access the chatbot with memory.

https://memmory-aware-chatbot-with-langgraph-24.onrender.com

# AGENTIC CHATBOT

## 1.Project structure

```
agentic_chatbot
├── README.md
├── app.py
├── config.py
├── static-|
|          |-script.js
|          |-style.css
├── model.py
├── requirements.txt
├── templates|-index.html

## 2. Setup Virtual Environment

Create a virtual environment (without conda):

```bash
python3 -m venv venv
```

Activate the virtual environment:

```bash
# On macOS/Linux
source venv/bin/activate

# On Windows
.\venv\Scripts\activate
```

## 3. Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

## 4. Configure Groq API

1. Generate your free Groq API key at: [https://console.groq.com/keys](https://console.groq.com/keys)
2. Set the API key in `config.py`:
```python
GROQ_API_KEY = "your-api-key-here"
```

## 5. Run the Application

Start the Flask server:

```bash
python app.py
```

Access the application at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

## 6. Interface

### Chatbot Interface
![CHATBOT UI](./image-2.png)
*Modern interface for stock analysis queries*

### Analysis Response
![Analysis Response](./image-1.png)
*Well-structured and fully analyzed data output*

## 7. Features

- Real-time stock data analysis
- Comparative analysis of multiple stocks
- Historical data visualization
- Responsive modern UI
- Markdown support for formatted output
- Code syntax highlighting

```bash
Chatbot Deployment
```

## 8. Deployed chatbot Link

#### you can access the deployed chatbot here.(Deployed on render cloud)
https://memmory-aware-chatbot-with-langgraph-12.onrender.com


# 3. Medical Chatbot with RAG and Tavily

![Chatbot UI](image-3.png)
*flask app*

![Chatbot Response UI](image-4.png)
*Well-structured and fully analyzed data output*
