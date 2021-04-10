from flask import Flask, render_template, request
from nltk.stem.porter import PorterStemmer
import pickle

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('sentiment_model.html')

@app.route('/predict/', methods=['POST'])
def predict():
	if request.method == 'POST':
		ps = PorterStemmer()
		model = pickle.load(open('bnb.pkl','rb'))
		user_input = request.form.get('size')
		clean_ui = ' '.join([ps.stem(word) for word in user_input.split()])
		prediction = model.predict([clean_ui])[0]
		prediction_proba = max(model.predict_proba([clean_ui])[0])*100
		review = 'positive' if prediction == 4 else 'negative'
		result = f'Prediction: I am {prediction_proba:.2f}% confident this is a {review} statement.'
		return render_template('sentiment_model.html', prediction=result)
	else:
		return render_template('sentiment_model.html')

if __name__ == '__main__':
	app.run(debug=True)