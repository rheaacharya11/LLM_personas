#!/bin/bash

# Define the API endpoint
URL="http://localhost:8000/v1/completions"

# Define the model path
MODEL_PATH="/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Llama-3.1-405B"

# Define the prompt
PROMPT="San Francisco is a"

# Define other parameters
MAX_TOKENS=7
TEMPERATURE=0

# Make the API request
curl "$URL" \
    -X POST \
    -H "Content-Type: application/json" \
    -d "{\
        \"model\": \"$MODEL_PATH\",\
        \"prompt\": \"$PROMPT\",\
        \"max_tokens\": $MAX_TOKENS,\
        \"temperature\": $TEMPERATURE\
    }"
