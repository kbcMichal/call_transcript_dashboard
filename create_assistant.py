import os
import argparse
import json
import requests
from dotenv import load_dotenv

# Load environment variables (if using .env)
load_dotenv()

# Set up argument parser
parser = argparse.ArgumentParser(description='Create an OpenAI Assistant')
parser.add_argument('--api-key', dest='api_key', help='OpenAI API Key (or set OPENAI_API_KEY env var)')
args = parser.parse_args()

# Determine API key to use
api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("No API key provided. Set OPENAI_API_KEY environment variable or pass --api-key")

# Instructions for the assistant
instructions = """You are an expert assistant specialized in analyzing call transcript data. 
You understand customer service and call center operations deeply.
You can help with:
1. Finding patterns in customer interactions
2. Analyzing sentiment across calls
3. Identifying common topics or issues
4. Providing insights on agent performance
5. Suggesting improvements for call handling

Always provide data-driven insights when possible, and be ready to create visualizations
when requested by the user."""

# Use the direct API request approach with v2
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
    "OpenAI-Beta": "assistants=v2"  # Updated to v2
}

payload = {
    "name": "Call Transcript Analyzer",
    "instructions": instructions,
    "model": "gpt-4o",
    "tools": [{"type": "code_interpreter"}]
}

response = requests.post(
    "https://api.openai.com/v1/assistants",
    headers=headers,
    data=json.dumps(payload)
)

if response.status_code == 200:
    assistant_data = response.json()
    print(f"Assistant created successfully!")
    print(f"Assistant ID: {assistant_data['id']}")
    print(f"Assistant Name: {assistant_data['name']}")
    print(f"Model: {assistant_data['model']}")
    print("\nIMPORTANT: Save this Assistant ID in your app's secrets as ASSISTANT_ID")
else:
    print(f"Failed to create assistant. Status code: {response.status_code}")
    print(f"Response: {response.text}") 