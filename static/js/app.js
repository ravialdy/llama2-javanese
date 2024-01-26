// app.js
function addMessageToChat(message) {
    var chatBox = document.getElementById('chatBox');
    var newMessage = document.createElement('div');
    newMessage.textContent = message;
    chatBox.appendChild(newMessage);
}

function sendMessage() {
    var userInput = document.getElementById('userInput');
    var message = userInput.value.trim();
    var loadingIndicator = document.getElementById('loadingIndicator');
    if (message) {
        addMessageToChat("You: " + message);
        loadingIndicator.style.display = 'block'; // Show loading indicator
        fetch('/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({'message': message})
        })
        .then(response => response.json())
        .then(data => {
            loadingIndicator.style.display = 'none'; // Hide loading indicator
            addMessageToChat("Bot: " + data.response);
        })
        .catch(error => {
            loadingIndicator.style.display = 'none'; // Hide loading indicator
            console.error('Error:', error);
        });
    }
    userInput.value = '';  // Clear input after sending
}