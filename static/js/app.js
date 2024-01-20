// chat.js
function sendMessage() {
    var userInput = document.getElementById('userInput');
    var message = userInput.value.trim();
    if (message) {
        addMessageToChat("You: " + message);
        fetch('/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({'message': message})
        })
        .then(response => response.json())
        .then(data => {
            addMessageToChat("Bot: " + data.response);
        });
    }
    userInput.value = '';  // Clear input after sending
}

function addMessageToChat(message) {
    var chatBox = document.getElementById('chatBox');
    var newMessage = document.createElement('div');
    newMessage.textContent = message;
    chatBox.appendChild(newMessage);
}