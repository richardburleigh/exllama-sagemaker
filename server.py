from model import ExLlama, ExLlamaCache, ExLlamaConfig
from flask import Flask, request, Response
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
import os, glob, json

# Directory containing config.json, tokenizer.model and safetensors file for the model
model_directory = "/app/model/Llama-2-13B-chat-GPTQ"

tokenizer_path = os.path.join(model_directory, "tokenizer.model")
model_config_path = os.path.join(model_directory, "config.json")
st_pattern = os.path.join(model_directory, "*.safetensors")
model_path = glob.glob(st_pattern)[0]

config = ExLlamaConfig(model_config_path)               # create config from config.json
config.model_path = model_path                          # supply path to model weights file

model = ExLlama(config)                                 # create ExLlama instance and load the weights
print(f"Model loaded: {model_path}")

tokenizer = ExLlamaTokenizer(tokenizer_path)            # create tokenizer from tokenizer model file
cache = ExLlamaCache(model)                             # create cache for inference
generator = ExLlamaGenerator(model, tokenizer, cache)   # create generator

# Flask app

app = Flask(__name__)


# Inference with settings equivalent to the "precise" preset from the /r/LocalLLaMA wiki

@app.route('/invocations', methods=['POST'])
def inferContextP():
    data = json.loads(request.data)
    assert 'prompt' in data, "Missing 'prompt' key in input data"
    prompt = data['prompt']

    generator.settings.token_repetition_penalty_max = 1.176
    generator.settings.token_repetition_penalty_sustain = config.max_seq_len
    generator.settings.temperature = data.get('temperature', 0.7)
    generator.settings.top_p = 0.1
    generator.settings.top_k = data.get('top_k', 40)
    generator.settings.typical = 0.0    # Disabled

    outputs = generator.generate_simple(prompt, max_new_tokens = data.get('max_new_tokens', 1024))
    return json.dumps(outputs)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy."""
    return Response(response='\n', status=200, mimetype='application/json')


# Start Flask app

host = "0.0.0.0"
port = 8080
print(f"Starting server on address {host}:{port}")

if __name__ == '__main__':
    from waitress import serve
    serve(app, host = host, port = port)
