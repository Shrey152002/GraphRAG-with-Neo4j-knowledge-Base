document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chat-messages');
    const questionInput = document.getElementById('question-input');
    const submitButton = document.getElementById('submit-btn');
    const loadingIndicator = document.getElementById('loading');

    function addMessage(content, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        messageDiv.classList.add(isUser ? 'user-message' : 'bot-message');
        messageDiv.textContent = content;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    async function askQuestion() {
        const question = questionInput.value.trim();
        if (!question) return;

        // Disable input and button while processing
        questionInput.disabled = true;
        submitButton.disabled = true;
        loadingIndicator.classList.remove('hidden');

        // Add user's question to chat
        addMessage(question, true);

        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question }),
            });

            const data = await response.json();

            if (data.status === 'success') {
                addMessage(data.answer);
            } else {
                addMessage('Sorry, I encountered an error while processing your question.');
            }
        } catch (error) {
            console.error('Error:', error);
            addMessage('Sorry, something went wrong. Please try again.');
        } finally {
            // Re-enable input and button
            questionInput.value = '';
            questionInput.disabled = false;
            submitButton.disabled = false;
            loadingIndicator.classList.add('hidden');
            questionInput.focus();
        }
    }

    // Event listeners
    submitButton.addEventListener('click', askQuestion);

    questionInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            askQuestion();
        }
    });
});