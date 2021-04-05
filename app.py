from flask import Flask, render_template, request
import nltk
import pickle

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('sentiment_input.html')

@app.route('/predict/', methods=['POST'])
def predict():
	if request.method == 'POST':
		model = pickle.load(open('mnb_model.pkl','rb'))
		user_input = request.form.get('size')
		stopwords = nltk.corpus.stopwords.words('english')
		ps = nltk.stem.PorterStemmer()
		lem = nltk.stem.WordNetLemmatizer()
		cleaned_ui = ' '.join([lem.lemmatize(ps.stem(w.lower())) for w in user_input.split() if not w in set(stopwords)])
		prediction = model.predict([cleaned_ui])[0]
		prediction_proba = max(model.predict_proba([cleaned_ui])[0])*100
		review = 'positive review' if prediction == 1 else 'negative review'
		result = f'I am {prediction_proba:.2f}% confident this is a {review}'
		return render_template('prediction.html', prediction=result)
	else:
		return render_template('sentiment_input.html')

if __name__ == '__main__':
	app.run(debug=True)