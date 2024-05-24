from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the pre-trained model and vectorizer
vectorizer = joblib.load('models/vectorizer.pkl')
clf = joblib.load('models/classifier.pkl')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        user_input = request.form['text']
        user_input_counts = vectorizer.transform([user_input])
        prediction = clf.predict(user_input_counts)
        sentiment = prediction[0]
        emoji = get_sentiment_emoji(sentiment)
        message = get_sentiment_message(sentiment)
        print("Input Text:", user_input)
        print("Predicted Sentiment:", sentiment)
        print("Generated Emoji:", emoji)
        print("Generated Message:", message)
        return render_template('index.html', sentiment=sentiment, emoji=emoji, message=message)

def get_sentiment_emoji(sentiment):
    emojis = {
        'Positive': '😊',
        'Negative': '😔',
        'neutral': '😐'
    }
    emoji = emojis.get(sentiment, '')
    print("Sentiment:", sentiment)
    print("Emoji:", emoji)
    return emoji

def get_sentiment_message(sentiment):
    messages = {
        'Positive': 'Looks like you are feeling positive today!',
        'Negative': 'I hope things get better for you soon.',
        'neutral': 'Seems like a neutral state of mind.'
    }
    message = messages.get(sentiment, '')
    print("Sentiment:", sentiment)
    print("Message:", message)
    return message

if __name__ == '__main__':
    app.run(debug=True)
