from django.shortcuts import render
from django.http import JsonResponse
from .models import Prompt
import requests
import json

def index(request):
    generated_text, error_message = "", ""

    if request.method == 'POST':
        input_prompt = request.POST.get('input_mssg')
        user_prompt = Prompt.objects.create(input_mssg=input_prompt)
        print("user prompt:", user_prompt)

        url = 'http://localhost:5000/generate'
        data = {'instruction': input_prompt}
        headers = {'Content-Type': 'application/json'}
        
        response = requests.post(url, headers=headers, data=json.dumps(data))
        
        if response.status_code == 200:
            generated_text = response.json().get('generated_text')
            print("generated text:", generated_text)
            return JsonResponse({'generated_text': generated_text})
        else:
            error_message = response.json().get('error', 'Unknown error')
            print("error message:", error_message)
            return JsonResponse({'error': error_message}, status=400)
    
    return render(request, 'index.html')
