from flask import Flask, request, jsonify
from huggingface_hub import login
from thai_llm_tester import LLM_Tester
import warnings

login()

# suppress warnings
warnings.filterwarnings("ignore")

# initialize flask app
app = Flask(__name__)

# load thai llm
llm_tester = LLM_Tester("raghav2005/thai_llm_lora")

@app.route('/generate', methods=['POST'])
def generate_text():
    try:
        data = request.get_json() # get request data
        instruction = data['instruction'] # extract instruction

        formatted_result = llm_tester.generate_output(instruction)

        return jsonify({'generated_text': formatted_result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500 # handle exceptions

# run the flask app
if __name__ == '__main__':
    app.run(debug = True)
