import requests
import json
import time
import subprocess
import sys

def test_api():
    base_url = "http://127.0.0.1:5000"
    
    # Start the server in a subprocess
    print("Starting server...")
    server = subprocess.Popen([sys.executable, "app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(3) # Wait for server to start
    
    try:
        # Test 1: Calculate Dissonance
        print("Testing /api/calculate_dissonance...")
        payload = {
            "partials": [
                {"ratio": 1.0, "amplitude": 1.0},
                {"ratio": 2.0, "amplitude": 0.5}
            ]
        }
        response = requests.post(f"{base_url}/api/calculate_dissonance", json=payload)
        if response.status_code == 200:
            data = response.json()
            if "alphas" in data and "dissonance" in data:
                print("PASS: Dissonance calculation returned valid data.")
            else:
                print("FAIL: Dissonance calculation returned invalid structure.")
        else:
            print(f"FAIL: Dissonance calculation returned {response.status_code}")
            
        # Test 2: Generate Audio
        print("Testing /api/generate_audio...")
        payload = {
            "partials": [
                {"ratio": 1.0, "amplitude": 1.0}
            ],
            "freqs": [440.0],
            "duration": 0.5
        }
        response = requests.post(f"{base_url}/api/generate_audio", json=payload)
        if response.status_code == 200 and response.headers['Content-Type'] == 'audio/wav':
            print("PASS: Audio generation returned WAV file.")
        else:
            print(f"FAIL: Audio generation returned {response.status_code} or wrong content type.")
            
    except Exception as e:
        print(f"FAIL: Exception occurred: {e}")
    finally:
        print("Stopping server...")
        server.terminate()
        server.wait()

if __name__ == "__main__":
    test_api()
