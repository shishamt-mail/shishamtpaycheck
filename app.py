import os
import requests
import base64
import json
import time
from flask import Flask, request, jsonify, render_template_string

# ----------------------------------------------------------------------
# ⚠️ IMPORTANT: PLACE YOUR GEMINI API KEY HERE
# ----------------------------------------------------------------------
# If this key is invalid or missing, you will receive an error.
GEMINI_API_KEY = "AIzaSyBN1AemjQffM8QtV9ykQoHk1x2FZqlBruI" # Your API Key goes here

# Set up the Flask application
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB file size

# The model to use for multimodal analysis
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"

# Define the JSON schema for the structured response
# The model MUST adhere to this structure.
RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "is_payment_screenshot": {
            "type": "BOOLEAN",
            "description": "True if the image appears to be a valid, successful payment transfer receipt/screenshot, False otherwise."
        },
        "error_reason": {
            "type": "STRING",
            "description": "A descriptive reason if 'is_payment_screenshot' is False, or if any compulsory item is missing (e.g., 'Not a payment screenshot', 'Transaction ID not found'). If all checks pass, set to 'N/A'."
        },
        "extracted_details": {
            "type": "OBJECT",
            "properties": {
                "amount": { "type": "STRING", "description": "The payment amount (compulsory). If not found, must be 'Amount not found'." },
                "payment_time": { "type": "STRING", "description": "The exact time and date of payment (compulsory). If not found, must be 'Payment time not found'." },
                "transaction_id": { "type": "STRING", "description": "The unique transaction ID/reference number (compulsory). If not found, must be 'Transaction ID not found'." },
                "sender_name": { "type": "STRING", "description": "The sender's name (not compulsory). If not found, must be 'Not found'." },
                "sender_info": {
                    "type": "OBJECT",
                    "description": "A nested object containing other possible sender details. Includes UPI ID, phone number, or remarks.",
                    "properties": {
                        "upi_id": { "type": "STRING", "description": "The sender's UPI ID, if available. If not found, set to 'Not found'." },
                        "remarks": { "type": "STRING", "description": "Any remark or note attached by the sender. If not found, set to 'Not found'." }
                    },
                    "required": [] # Ensures these nested fields are optional
                }
            },
            "propertyOrdering": ["amount", "payment_time", "transaction_id", "sender_name", "sender_info"]
        }
    },
    "propertyOrdering": ["is_payment_screenshot", "error_reason", "extracted_details"]
}

# The main prompt instructing the model on the task and rules
SYSTEM_PROMPT = """
You are an expert payment receipt analyzer. Your task is to analyze the uploaded image.
1. Determine if the image is a screenshot of a successful digital payment transfer.
2. Extract the following details into the provided JSON schema.
3. CRITICAL RULE: The fields 'amount', 'payment_time', and 'transaction_id' are compulsory. If you cannot find any of these, you MUST populate that specific field with the string 'Amount not found', 'Payment time not found', or 'Transaction ID not found' respectively, AND set 'is_payment_screenshot' to False and provide a reason in 'error_reason'.
4. If the image is clearly not a payment screenshot, set 'is_payment_screenshot' to False and explain why in 'error_reason'.
5. If the image is a payment screenshot and all compulsory fields are found, set 'is_payment_screenshot' to True and 'error_reason' to 'N/A'.
"""

def retry_fetch(payload, max_retries=3):
    """Handles API request with exponential backoff."""
    delay = 1  # Initial delay in seconds
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, json=payload)
            
            # 429 is Too Many Requests, 500/503 are server errors. These are retriable.
            if response.status_code in [429, 500, 503]:
                if attempt < max_retries - 1:
                    # print(f"Retryable error {response.status_code}. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                    continue
            
            return response

        except requests.exceptions.RequestException:
            # Catch network errors (like DNS failure, connection timeout)
            if attempt < max_retries - 1:
                # print(f"Network error. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
                continue
            
            raise # Re-raise the exception on final failure

    return None # Should be unreachable if max_retries > 0

def call_gemini_api(base64_image_data, mime_type):
    """Makes the API call to Gemini for structured analysis."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_API_KEY_HERE":
        return {"status": "error", "message": "API Key is missing or invalid. Please check GEMINI_API_KEY in app.py."}
    
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "Analyze this payment screenshot and extract the details according to the required JSON schema. Pay close attention to the compulsory fields."},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": base64_image_data
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA
        },
        "systemInstruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        }
    }

    try:
        response = retry_fetch(payload)
        
        if response is None:
            return {"status": "network_error", "message": "API connection failed after multiple retries."}

        if response.status_code != 200:
            # Attempt to extract detailed error message from the response body
            error_message = f"HTTP Error {response.status_code}. Status: {response.reason}."
            try:
                error_data = response.json()
                # The Gemini API usually wraps errors in an 'error' object
                detail_message = error_data.get('error', {}).get('message', error_data.get('message'))
                if detail_message:
                    # Provide the specific API error message
                    error_message = f"Gemini API Request Failed: {detail_message}"
            except json.JSONDecodeError:
                error_message = f"API Error ({response.status_code}). Response body was not JSON: {response.text[:100]}..."

            return {"status": "api_error", "message": error_message}
        
        # If 200, proceed with success logic
        
        # Extract the JSON response text from the model
        result = response.json()
        model_response_text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '{}')
        
        # Parse the structured JSON response
        structured_data = json.loads(model_response_text)
        
        return {"status": "success", "data": structured_data}

    except requests.exceptions.RequestException as e:
        return {"status": "network_error", "message": f"API or network error after all retries: {e}"}
    except json.JSONDecodeError:
        return {"status": "parsing_error", "message": "Failed to parse JSON response from the API (Model output may be malformed)."}
    except Exception as e:
        return {"status": "server_error", "message": f"An unexpected server error occurred: {e}"}

@app.route('/', methods=['GET'])
def index():
    """Renders the HTML upload form."""
    return render_template_string(open('index.html').read())

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles the file upload and analysis request."""
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part in the request"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected for uploading"}), 400

    if file:
        # Check MIME type
        mime_type = file.mimetype
        # Added a stricter check for common image types
        if not mime_type or mime_type not in ['image/jpeg', 'image/png', 'image/webp']:
            return jsonify({"status": "error", "message": f"Invalid file type. Please upload a JPEG, PNG, or WebP image. Found: {mime_type}"}), 400
        
        try:
            # Read file and encode to base64
            file_bytes = file.read()
            base64_encoded_data = base64.b64encode(file_bytes).decode('utf-8')
            
            # Call the Gemini API function
            api_result = call_gemini_api(base64_encoded_data, mime_type)
            
            if api_result['status'] == 'success':
                # The 'data' field contains the structured JSON from the model
                return jsonify(api_result['data'])
            else:
                # Return the detailed error message from the API or network failure
                return jsonify(api_result), 500

        except Exception as e:
            return jsonify({"status": "server_error", "message": f"Server processing error during file handling: {e}"}), 500

if __name__ == '__main__':
    # Set GEMINI_API_KEY before running.
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
