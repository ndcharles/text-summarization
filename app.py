# import config
import torch
import flask
from flask import Flask, request, jsonify, render_template
import json
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

BART_PATH = 'facebook/bart-large'
T5_PATH = 't5-base'
# BART_PATH = 'model/bart'
# T5_PATH = 'model/t5'

'''
These are the steps to build the summarisation models (BART/T5)
1. Import necessary libraries and frameworks
2. Create model instances
3. Initialise the tokenizer
4. Get the article/text 
5. Input text and tokenize it
6. Apply the model to generate summary_id
7. Apply the tokenizer to decode the generated summary_ids
8. Return resulting summary
'''

app = Flask(__name__)
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')


t5_model = T5ForConditionalGeneration.from_pretrained(T5_PATH)
t5_tokenizer = T5Tokenizer.from_pretrained(T5_PATH)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def bart_summarize(input_text, num_beams=4, num_words=100):
    '''
    :input text: enter text to summarise
    :num_beams: num_beams refers to beam search, which is used for text generation.
    It returns the n most probable next words, rather than greedy search which returns the most probable next word.
    :num_words: number of words to output
    :n_gram: An ngram is a repeating phrase,
    where the 'n' stands for 'number' and the 'gram' stands for the words; e.g. a 'trigram' would be a three word ngram.
    :return: final summarized text
    '''
    input_text = str(input_text) # text input
    input_text = ' '.join(input_text.split()) # split along punctuation and join back
    input_tokenized = bart_tokenizer.encode(input_text, return_tensors='pt').to(device) # return_tensors=pt throws back a pytorch output
    summary_ids = bart_model.generate(input_tokenized,
                                      num_beams=int(num_beams),
                                      no_repeat_ngram_size=2,
                                      length_penalty=2.0,
                                      min_length=30,
                                      max_length=int(num_words),
                                      early_stopping=True)
    output = [bart_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    return output[0]


def t5_summarize(input_text, num_beams=4, num_words=100):
    input_text = str(input_text).replace('\n', '')
    input_text = ' '.join(input_text.split())
    input_tokenized = t5_tokenizer.encode(input_text, return_tensors="pt").to(device)
    summary_task = torch.tensor([[21603, 10]]).to(device) # more review needed
    input_tokenized = torch.cat([summary_task, input_tokenized], dim=-1).to(device)
    summary_ids = t5_model.generate(input_tokenized,
                                    num_beams=int(num_beams),
                                    no_repeat_ngram_size=4,
                                    length_penalty=2.0,
                                    min_length=30,
                                    max_length=int(num_words),
                                    early_stopping=True)
    output = [t5_tokenizer.decode(g, skip_special_tokens=False, clean_up_tokenization_spaces=False) for g in summary_ids]
    return output[0]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        sentence = request.json['input_text']
        num_words = request.json['num_words']
        num_beams = request.json['num_beams']
        model = request.json['model']
        if sentence != '':
            if model.lower() == 'bart':
                output = bart_summarize(sentence, num_beams, num_words)
            else:
                output = t5_summarize(sentence, num_beams, num_words)
            response = {}
            response['response'] = {
                'summary': str(output),
                'model': model.lower()
            }
            return flask.jsonify(response)
        else:
            res = dict({'message': 'Please add an input'})
            return app.response_class(response=json.dumps(res), status=500, mimetype='application/json')
    except Exception as ex:
        res = dict({'message': str(ex)})
        print(res)
        return app.response_class(response=json.dumps(res), status=500, mimetype='application/json')


if __name__ == '__main__':
    bart_model.to(device)
    bart_model.eval()
    t5_model.to(device)
    t5_model.eval()
    app.run(host='0.0.0.0', debug=True, port=8000, use_reloader=False)
