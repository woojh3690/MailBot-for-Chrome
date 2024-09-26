const messagesDiv = document.getElementById('messages');
const userInput = document.getElementById('user-input');

userInput.addEventListener('keypress', function (e) {
    if (e.key === 'Enter' && userInput.value.trim() !== '') {
        const query = userInput.value.trim();
        userInput.value = '';

        // Display user's message
        const userMessage = document.createElement('div');
        userMessage.className = 'message user';
        userMessage.textContent = query;
        messagesDiv.appendChild(userMessage);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;

        // Send query to the backend server
        fetch('http://localhost:5000/chatbot', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query })
        })
        .then(response => response.json())
        .then(data => {
            // Display bot's answer
            const botMessage = document.createElement('div');
            botMessage.className = 'message bot';
            botMessage.textContent = data.answer;
            messagesDiv.appendChild(botMessage);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
});
