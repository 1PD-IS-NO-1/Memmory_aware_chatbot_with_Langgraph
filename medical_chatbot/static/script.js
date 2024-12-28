document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chat-form');
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');

    // Configure marked options
    marked.setOptions({
        breaks: true,  // Convert \n to <br>
        gfm: true,     // Enable GitHub Flavored Markdown
        sanitize: false // Allow HTML in the markdown
    });

    chatForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        const query = userInput.value.trim();
        if (!query) return;

        // Add user message
        addMessage('user', query);
        userInput.value = '';

        // Add loading message
        const loadingDiv = addLoadingMessage();

        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: query })
            });

            const data = await response.json();
            
            // Remove loading message
            loadingDiv.remove();

            if (data.error) {
                addMessage('bot', 'Sorry, I encountered an error. Please try again.');
            } else {
                addMessage('bot', data.response, true); // Added isMarkdown flag
            }
        } catch (error) {
            console.error('Error:', error);
            loadingDiv.remove();
            addMessage('bot', 'Sorry, something went wrong. Please try again.');
        }

        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    });

    function addMessage(type, content, isMarkdown = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
        
        const iconDiv = document.createElement('div');
        iconDiv.className = 'message-icon';
        iconDiv.innerHTML = type === 'user' ? 
            '<i class="fas fa-user"></i>' : 
            '<i class="fas fa-robot"></i>';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content markdown-content';
        
        // Parse markdown if it's a bot message
        if (type === 'bot' && isMarkdown) {
            contentDiv.innerHTML = marked.parse(content);
        } else {
            contentDiv.textContent = content;
        }

        messageDiv.appendChild(iconDiv);
        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);

        // Initialize code highlighting if there are code blocks
        if (type === 'bot' && isMarkdown) {
            messageDiv.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightBlock(block);
            });
        }
    }

    function addLoadingMessage() {
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message bot-message loading';
        loadingDiv.innerHTML = `
            <div class="message-icon">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-content">
                <div class="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;
        chatMessages.appendChild(loadingDiv);
        return loadingDiv;
    }

    function clearChat() {
        chatMessages.innerHTML = '';
    }

    // Make clearChat available globally
    window.clearChat = clearChat;
});