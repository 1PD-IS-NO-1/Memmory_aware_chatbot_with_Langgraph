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
![CHATBOT UI](./images-1.png)
*Modern interface for stock analysis queries*

### Analysis Response
![Analysis Response](./images.png)
*Well-structured and fully analyzed data output*

## 7. Features

- Real-time stock data analysis
- Comparative analysis of multiple stocks
- Historical data visualization
- Responsive modern UI
- Markdown support for formatted output
- Code syntax highlighting