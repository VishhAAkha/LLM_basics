import requests
import json

# URL where the LLM model is running
url = 'http://localhost:5000/generate'

while True:

    user_input_instruction = input("Enter your prompt: ")
    # create data payload with user instruction
    data = {'instruction': user_input_instruction}
    # set headers for the POST request
    headers = {'Content-Type': 'application/json'}

    # send POST request to the model's URL
    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200: # 200 (OK)
        print(response.json()['generated_text'])
    else:
        print('Error:', response.json()['error']) # request failed
