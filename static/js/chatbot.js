document.getElementById('send-btn').addEventListener('click', function() {
    sendMessage();
});

document.getElementById('user-input').addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
});

function sendMessage() {
    const userInput = document.getElementById('user-input').value.trim();
    if (!userInput) return;

    const chatBox = document.getElementById('chat-box');
    chatBox.innerHTML += `
        <div class="chat-message user">
            <p>${userInput}</p>
        </div>
    `;
    document.getElementById('user-input').value = '';
    scrollToBottom(chatBox);

    fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userInput })
    })
    .then(response => response.json())
    .then(data => {
        if (data.response) {
            chatBox.innerHTML += `
                <div class="chat-message assistant">
                    <p>${data.response}</p>
                </div>
            `;
        } else if (data.error) {
            chatBox.innerHTML += `
                <div class="chat-message assistant">
                    <p>Error: ${data.error}</p>
                </div>
            `;
        }
        scrollToBottom(chatBox);
    })
    .catch(error => {
        chatBox.innerHTML += `
            <div class="chat-message assistant">
                <p>Error: ${error.message}</p>
            </div>
        `;
        scrollToBottom(chatBox);
    });
}

function scrollToBottom(chatBox) {
    chatBox.scrollTop = chatBox.scrollHeight;
}
