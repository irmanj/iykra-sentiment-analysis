import pickle

from os import environ
from os.path import join, dirname
from dotenv import load_dotenv

from flask import Flask, request, jsonify
from config import FEATURE_EXTRACTOR_FILEPATH, CLASSIFIER_FILEPATH, LABELS

from config import DATA_FILEPATH, FEATURE_EXTRACTOR_FILEPATH, CLASSIFIER_FILEPATH

app = Flask(__name__)

with open(FEATURE_EXTRACTOR_FILEPATH, 'rb') as infile:
    app.feature_extractor = pickle.load(infile)

with open(CLASSIFIER_FILEPATH, 'rb') as infile:
    app.classifier = pickle.load(infile)

def reply_success(data):
    response = jsonify({
        "data": data
    })

    response.headers['Access-Control-Allow-Origin'] = '*'

    return response

def reply_error(code, message):
    response = jsonify({
        "error": {
            "code": code,
            "message": message
        }
    })

    response.headers['Access-Control-Allow-Origin'] = '*'

    return response

# function to check duplicate text in positive.txt and negative.txt file
def check_duplicate(text):

    # get list positive tweets
    with open(DATA_FILEPATH + "/positive.txt", "r") as infile:
        positive_tweets = infile.readlines()

    # get list negative tweets
    with open(DATA_FILEPATH + "/negative.txt", "r") as infile:
        negative_tweets = infile.readlines()

    in_file = ''

    #looping for found duplicate text in positive.txt file 
    for phrase in positive_tweets:
        search = phrase.split('. ')
        for s in search:
            if text in s:
                in_file = 'positive'
                return "We have it already!"
    
    # looping for found duplicate text in negative.txt file 
    for phrase in negative_tweets:
        search = phrase.split('. ')
        for s in search:
            if text in s:
                in_file = 'negative'
                return "We have it already!"
    
    # if feedback was not found then append the feedback on text
    if in_file=='positive':
        #append text in positive file
        with open(DATA_FILEPATH + "/positive.txt", "a") as infile:
            infile.write("\n" + text)
    else:
        #append text in negative file
        with open(DATA_FILEPATH + "/negative.txt", "a") as infile:
            infile.write("\n" + text)
                
    # update counter add new data
    with open(DATA_FILEPATH + "/count_new_data_added.txt", "r") as infile:
        total = infile.readlines()

    # update counter and write it to file
    with open(DATA_FILEPATH + "/count_new_data_added.txt", "w") as infile:
        total_new = int(total[0]) + 1
        infile.write(str(total_new))

    # then return text that feedback is well received
    return "Your feedback is well received!"

@app.route("/")
def index():
    return "<h1>Sentiment Analysis API using Flask</h1>"

@app.route("/classify", methods=["GET", "POST"])
def classify():
    if request.method == "GET":
        text = request.args.get("text", None)
    elif request.method == "POST":
        json_req = request.get_json()
        text = json_req["text"]
    else:
        return reply_error(code=400, message="Supported method is 'GET' and 'POST'")

    if text:
        # IMPORTANT: Use [text] because sklearn vectorizer expects an iterable as the input
        # IMPORTANT: classifier.predict returns an array, so get the first element
        label = app.classifier.predict(app.feature_extractor.transform([text]))[0]

        return reply_success(data={
            "text": text,
            "sentiment": LABELS[label]
        })

    return reply_error(code=400, message="Text is not specified")

@app.route("/feedback", methods=["GET","POST"])
def feedback():
    if request.method == "GET":
        text = request.args.get("text", None)
    elif request.method == "POST":
        json_req = request.get_json()
        text = json_req["text"]
    else:
        return reply_error(code=400, message="Supported method is 'GET' and 'POST'")
    
    if text:
        is_duplicate = check_duplicate(text)

        label = app.classifier.predict(app.feature_extractor.transform([text]))[0]

        return reply_success(data={
            "text": text,
            "sentiment": LABELS[label],
            "msg": is_duplicate
        })

    return reply_error(code=400, message="Text is not specified")

if __name__ == "__main__":
    dotenv_path = join(dirname(__file__), '.env')
    load_dotenv(dotenv_path)

    port = int(environ.get("PORT"))
    debug = environ.get("DEBUG")

    if debug == "True":
        app.run(threaded=True, port=port, debug=True)
    else:
        app.run(threaded=True, port=port, debug=False)
