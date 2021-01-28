import datetime

from flask import Flask, render_template, request
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
from transformers import pipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_synopsis', methods=['POST'])
def generate_synopsis():
    choosen_category = request.form['category']
    generated_synopsis = get_new_synopsis(choosen_category)
    return render_template('generated_synopsis.html', category=choosen_category, generated_synopsis=generated_synopsis)


def load_model(category):
    model_path = f"../models/model_gpt2_{category}"
    tokenizer = GPT2Tokenizer.from_pretrained(model_path, local_files_only=True)
    model = TFGPT2LMHeadModel.from_pretrained(model_path, local_files_only=True)
    return model, tokenizer

def get_new_synopsis(category):
    model, tokenizer = load_model(category)
    nlp = pipeline("text-generation")
    text = nlp("", max_length=10)[0]["generated_text"]
    input = tokenizer.encode(text, return_tensors='tf')
    print("prediction_start", datetime.datetime.now())
    beam_output = model.generate(
        input_ids=input,
        min_length=500,
        max_length=2000,
        num_beams=1,
        temperature=0.97,
        no_repeat_ngram_size=2,
        num_return_sequences=1
    )
    generated_synopsis = tokenizer.decode(beam_output[0])
    print("prediction_end", datetime.datetime.now())
    return generated_synopsis
