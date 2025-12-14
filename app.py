from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

# Initialize text processing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load the trained model and vectorizer
models_dir = 'models'

try:
    with open(f'{models_dir}/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open(f'{models_dir}/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    
    with open(f'{models_dir}/model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    print("✓ Model loaded successfully!")
    print(f"Model: {metadata['model_name']}")
    print(f"Accuracy: {metadata['accuracy']:.4f}")
    print(f"F1-Score: {metadata['f1_score']:.4f}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please run the training notebook first to generate the model files.")
    model = None
    tfidf = None
    metadata = None


def clean_text(text):
    """Clean and normalize text"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def tokenize_and_lemmatize(text):
    """Tokenize and lemmatize text"""
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    
    return ' '.join(tokens)


def predict_sentiment(review_text):
    """
    Predict sentiment score for a review
    
    Parameters:
    - review_text: string, the review to predict
    
    Returns:
    - prediction: int, predicted score (1-5)
    - probabilities: dict, confidence scores for each class
    """
    if model is None or tfidf is None:
        raise Exception("Model not loaded. Please train the model first.")
    
    # Preprocess the text
    cleaned = clean_text(review_text)
    processed = tokenize_and_lemmatize(cleaned)
    
    # Handle empty processed text
    if not processed:
        return 3, None 
    
    # Transform using TF-IDF vectorizer
    vectorized = tfidf.transform([processed])
    
    # Make prediction
    prediction = int(model.predict(vectorized)[0])
    
    # Get probability scores if available
    probabilities = None
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(vectorized)[0]
        probabilities = {int(score): float(prob) for score, prob in zip(model.classes_, probs)}
    
    return prediction, probabilities


@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Candy Crush Review Sentiment Analysis API',
        'status': 'running',
        'model': metadata['model_name'] if metadata else 'Not loaded',
        'endpoints': {
            '/predict': 'POST - Predict sentiment score for a review',
            '/health': 'GET - Check API health status'
        }
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_name': metadata['model_name'] if metadata else None,
        'accuracy': metadata['accuracy'] if metadata else None,
        'f1_score': metadata['f1_score'] if metadata else None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict sentiment score for a review
    
    Expected JSON body:
    {
        "review": "Your review text here"
    }
    
    Returns:
    {
        "review": "original review text",
        "predicted_score": 5,
        "confidence": {
            "1": 0.05,
            "2": 0.10,
            "3": 0.15,
            "4": 0.20,
            "5": 0.50
        },
        "model": "Logistic Regression"
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate input
        if not data or 'review' not in data:
            return jsonify({
                'error': 'Missing "review" field in request body',
                'example': {'review': 'This game is amazing!'}
            }), 400
        
        review_text = data['review']
        
        # Validate review text
        if not review_text or not isinstance(review_text, str):
            return jsonify({
                'error': 'Review must be a non-empty string'
            }), 400
        
        if len(review_text.strip()) == 0:
            return jsonify({
                'error': 'Review cannot be empty or only whitespace'
            }), 400
        
        # Make prediction
        predicted_score, confidence = predict_sentiment(review_text)
        
        # Prepare response
        response = {
            'review': review_text,
            'predicted_score': predicted_score,
            'confidence': confidence,
            'model': metadata['model_name'] if metadata else 'Unknown',
            'sentiment': get_sentiment_label(predicted_score)
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


def get_sentiment_label(score):
    """Convert score to sentiment label"""
    if score <= 2:
        return 'Negative'
    elif score == 3:
        return 'Neutral'
    else:
        return 'Positive'


if __name__ == '__main__':
    print("\n" + "="*80)
    print("CANDY CRUSH REVIEW SENTIMENT ANALYSIS API")
    print("="*80)
    if model is not None:
        print(f"✓ Model loaded: {metadata['model_name']}")
        print(f"✓ Accuracy: {metadata['accuracy']:.4f}")
        print(f"✓ F1-Score: {metadata['f1_score']:.4f}")
    else:
        print("✗ Model not loaded - please run the training notebook first")
    print("="*80)
    print("\nStarting Flask server...")
    print("API will be available at: http://localhost:5000")
    print("\nEndpoints:")
    print("  GET  /          - API information")
    print("  GET  /health    - Health check")
    print("  POST /predict   - Predict sentiment")
    print("\nPress CTRL+C to stop the server")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
