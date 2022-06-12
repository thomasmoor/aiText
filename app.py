# https://huggingface.co/docs/transformers/model_doc/gpt_neo

from collections import OrderedDict
import glob
from flask import Flask, json, jsonify, redirect, render_template, request, session, make_response, url_for,Response
from flask_cors import CORS, cross_origin
from flask_session import Session
import json
import logging
import os
import re
import torch
from transformers import GPT2Tokenizer 
from transformers import GPTNeoForCausalLM
from transformers import TRANSFORMERS_CACHE
import openai
import config

# pip install flask flask_cors flask_session openai torch transformers

useGPT3=True
useNeo=True

# Set up logging
logging.basicConfig(
  filename='aiText.log',
  # encoding='utf-8',
  format='%(asctime)s %(levelname)s:%(message)s',
  level=logging.DEBUG
)
logging.debug("Logging activated")
logging.debug(f"TRANSFORMERS_CACHE: {TRANSFORMERS_CACHE}")

# Create the Flask instance
app = Flask(__name__)

# Enable Cross-Origin Resource Sharing for API use from another IP and/or port
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Use Flask server session to avoid a "Confirm Form Resubmission" pop-up:
# Redirect and pass form values from post to get method
app.config['SECRET_KEY'] = "your_secret_key" 
app.config['SESSION_TYPE'] = 'filesystem' 
app.config['SESSION_PERMANENT']= False
app.config.from_object(__name__)
Session(app)

# Initiate the GPT-3 model
# key=os.environ["OPENAI_API_KEY"]
if useGPT3:
  print(f"OpenAI key: {config.OPENAI_API_KEY}")
  openai.api_key=config.OPENAI_API_KEY

# Initiate the GPT-Neo Model
if useNeo:
  logging.debug("Loading modelCasualLM")
  print("Loading modelCasualLM")
  modelCasualLM = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
  logging.debug("Loading tokenizer")
  print("Loading tokenizer")
  tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
  logging.debug("Done")
  print("Done")

# API Endpoint
@app.route('/getaitext', methods=['POST'])
@cross_origin()
def api():
  logging.debug(f"aitext request - branch: api")
  data = json.loads(request.data)
  # logging.debug(data)
  max_length=data['maxLength']
  prompt=data['prompt']
  temperature=data['temperature']
  logging.debug(f"aitext request - branch: api - max_length: {max_length} temperature: {temperature} prompt: {prompt}")
  response=generateNeo(max_length,prompt,temperature)
  logging.debug(response)
  return jsonify(response)
# api

# HTML home page
@app.route('/', methods=['GET','POST'])
def slash():

  # The 'generate' button was pressed
  if 'generate' in request.form:
    max_length = request.form["maxLength"]
    prompt = request.form["prompt"]
    temperature = request.form["temperature"]
    temperature = float(temperature)
    if temperature > 1: temperature=temperature/ 100.0
    print(f"aiText - branch: generate - max_length: {max_length}  prompt: {prompt} temperature: {temperature}")
    logging.debug(f"aiText - branch: generate - max_length: {max_length}  prompt: {prompt} temperature: {temperature}")

    # OpenAI GPT-3 response
    echo=True
    resultsGPT3=generateGPT3(echo,max_length,prompt,temperature)
    # Neo response
    resultsNeo=generateNeo(max_length,prompt,temperature)    
    # Combined
    results={
      'GPT-3': resultsGPT3,
      'Neo': resultsNeo
    }
    
  # Download Option
  elif 'download' in request.form and 'results' in session:
    pass
  # Download

  # Redirect
  if request.method=='POST':
    print("aiText - branch: redirect")
    session['results'] = json.dumps(results)
    return redirect(url_for('slash'))

  # Render
  else:
    print("aiText - branch: render index.html")
    if 'results' in session:
      j=session['results']
      if j:
        results=json.loads(session['results'])
      else:
        results={}
    else:
      results={}
    return render_template("index.html",results=results)
  
# slash
  
def generateGPT3(echo,max_tokens,prompt,temperature):
  max_tokens=int(max_tokens)
  temperature=float(temperature)
  print(f"generateGPT3 - mt:{max_tokens} t:{temperature} p:{prompt}")
  response=openai.Completion.create(
    echo=echo,
    engine='text-davinci-002',
    prompt=prompt,
    temperature=temperature,
    max_tokens=max_tokens,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0.6,
    user="user123456"
    # stop="\n"
  )
  print(f"generateGPT3: {response}")
  print(f"generateGPT3: {response['choices'][0]['text']}")
  return response['choices'][0]['text']
# generateGPT3

def generateNeo(max_length,prompt,temperature):

  # return 'Please retry later'

  max_length=int(max_length)
  temperature=float(temperature)
  print(f"generateNeo - ml: {max_length} t:{temperature} p:{prompt}")
  logging.debug(f"generateNeo - ml: {max_length} t:{temperature} p:{prompt}")
  
  print("generateNeo - input_ids")
  logging.debug("generateNeo - input_ids")
  input_ids = tokenizer(prompt, return_tensors="pt").input_ids

  print("generateNeo - gen_tokens")
  logging.debug("generateNeo - gen_tokens")
  gen_tokens = modelCasualLM.generate(
    input_ids,
    do_sample=True,
    temperature=temperature,
    max_length=max_length,
  )

  print("generateNeo - decode")
  logging.debug("generateNeo - decode")
  response = tokenizer.batch_decode(gen_tokens)[0]

  logging.debug(f"generateNeo: {response}")
  print(f"generateNeo: {response}")
  return response
# generateNeo


prompt='Write a tagline for a sports shop'
# prompt='How many pounds are in a kilogram?'
# if useGPT3:
#   generateGPT3(True,100,prompt,0.5)
# if useNeo:
#   generateNeo(100,prompt,0.5)

app.run(host='0.0.0.0', port=5004, debug=True)
