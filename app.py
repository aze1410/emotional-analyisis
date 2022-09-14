from flask import Flask, request, render_template
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk


nltk.download('vader_lexicon')
app = Flask(__name__)


@app.route('/', methods=["GET","POST"])

def main():
    if request.method == "POST":
        inp = request.form.get("inp")
        sid = SentimentIntensityAnalyzer()
        score = sid.polarity_scores(inp)
        if score["neg"] != 0:
            return render_template('home.html', message="Negative😐😖")
        else:
            return render_template('home.html', message="Postive😊🥰")
    return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)
    
    
