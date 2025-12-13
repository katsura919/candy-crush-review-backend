"""
Test script for the Sentiment Analysis API
Run this after starting the Flask server to test the endpoints
"""

import requests
import json

# Base URL
BASE_URL = "http://localhost:5000"

def test_home():
    """Test the home endpoint"""
    print("\n" + "="*80)
    print("TEST 1: Home Endpoint (GET /)")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")

def test_health():
    """Test the health endpoint"""
    print("\n" + "="*80)
    print("TEST 2: Health Check (GET /health)")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")

def test_predict_positive():
    """Test prediction with a positive review"""
    print("\n" + "="*80)
    print("TEST 3: Positive Review Prediction")
    print("="*80)
    
    data = {
        "review": "This game is absolutely amazing! I love playing it every day!"
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"Input: {data['review']}")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")

def test_predict_negative():
    """Test prediction with a negative review"""
    print("\n" + "="*80)
    print("TEST 4: Negative Review Prediction")
    print("="*80)
    
    data = {
        "review": "Worst game ever! Too many ads and constant crashes."
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"Input: {data['review']}")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")

def test_predict_neutral():
    """Test prediction with a neutral review"""
    print("\n" + "="*80)
    print("TEST 5: Neutral Review Prediction")
    print("="*80)
    
    data = {
        "review": "It's okay, nothing special but it passes the time."
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"Input: {data['review']}")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")

def test_predict_missing_field():
    """Test prediction with missing review field"""
    print("\n" + "="*80)
    print("TEST 6: Missing Review Field (Error Test)")
    print("="*80)
    
    data = {
        "text": "This should fail"
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"Input: {data}")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")

def test_predict_empty_review():
    """Test prediction with empty review"""
    print("\n" + "="*80)
    print("TEST 7: Empty Review (Error Test)")
    print("="*80)
    
    data = {
        "review": ""
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"Input: {data['review']}")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")

def test_batch_predictions():
    """Test multiple predictions"""
    print("\n" + "="*80)
    print("TEST 8: Batch Predictions")
    print("="*80)
    
    reviews = [
        "Excellent game!",
        "Terrible experience",
        "It's fine",
        "Best game ever created!",
        "Don't waste your time"
    ]
    
    results = []
    for review in reviews:
        response = requests.post(f"{BASE_URL}/predict", json={"review": review})
        if response.status_code == 200:
            data = response.json()
            results.append({
                'review': review,
                'score': data['predicted_score'],
                'sentiment': data['sentiment']
            })
    
    print("\nResults:")
    for i, result in enumerate(results, 1):
        print(f"{i}. \"{result['review']}\"")
        print(f"   Score: {result['score']}/5 ({result['sentiment']})\n")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("CANDY CRUSH SENTIMENT API - TEST SUITE")
    print("="*80)
    print("Make sure the Flask server is running on http://localhost:5000")
    print("="*80)
    
    try:
        # Run all tests
        test_home()
        test_health()
        test_predict_positive()
        test_predict_negative()
        test_predict_neutral()
        test_predict_missing_field()
        test_predict_empty_review()
        test_batch_predictions()
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED!")
        print("="*80 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to the API server.")
        print("Please make sure the Flask server is running:")
        print("  python app.py")
        print("\n")
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
