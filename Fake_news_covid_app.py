import streamlit as st 
import joblib,os
import spacy
import pandas as pd
import numpy as np
nlp = spacy.load('en_core_web_sm')
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize


# load Vectorizer For Gender Prediction
news_vectorizer = open('news_classifier_nlp-app\models\\tfidf_vectorizer.pkl',"rb")
news_cv = joblib.load(news_vectorizer)

stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
porter = PorterStemmer()

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

def stopwords_lemma_stemming(data):
  newscolumns = ' '.join([word for word in data.split() if word.lower() not in (stop)])
  newscolumns = ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(newscolumns)])
  newscolumns = stemSentence(newscolumns)
  return newscolumns

def load_prediction_models(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model

# Get the Keys
def get_key(val,my_dict):
	for key,value in my_dict.items():
		if val == value:
			return key



def main():
	"""News Classifier"""
	st.title("News Classifier")
	html_temp = """
	<div style="background-color:blue;padding:10px">
	<h1 style="color:white;text-align:center;">Streamlit ML App </h1>
	</div>

	"""
	st.markdown(html_temp,unsafe_allow_html=True)

	activity = ['NLP','Prediction']
	choice = st.sidebar.selectbox("Select Activity",activity)


	if choice == 'Prediction':
		st.info("Prediction with ML")

		news_text = st.text_area("Enter News Here","Type Here", height= 250)
		all_ml_models = ["Ensemble"]
		model_choice = st.selectbox("Select Model",all_ml_models)

		prediction_labels = {'Fake': 0,'Real': 1}
		if st.button("Predict"):
			news_text = stopwords_lemma_stemming(news_text)
			vect_text = news_cv.transform([news_text]).toarray()
			if model_choice == 'Ensemble':
				predictor = load_prediction_models("news_classifier_nlp-app\models\models.pkl")
				prediction = predictor['stacking'].predict(vect_text)

			final_result = get_key(prediction,prediction_labels)
			st.success("News Categorized as:: {}".format(final_result))
			def mapping(raW_text):
					text = raW_text
					vectext = news_cv.transform([text]).toarray()
					feature_coef = vectext.squeeze()*predictor['LogisticRegression'].coef_.squeeze()
					feature_index = np.where(feature_coef != 0)[0]
					feature_coef_mapping = {}
					for i in feature_index:
						feature_coef_mapping[news_cv.get_feature_names()[i]] = predictor['LogisticRegression'].coef_.squeeze()[i]
					temp = sorted(feature_coef_mapping.items(), key=lambda x: x[1], reverse=True)    
					top_5 = temp[:5]
					bottom_5 = temp[-5:]
					feature_imp = top_5 + bottom_5
					name = []
					number = []
					for i in feature_imp:
						name.append(i[0])
						number.append(i[1])
					number = np.asanyarray(number)
					number = number.astype('float')
					return name, number, feature_coef_mapping
			name, number, feature_coef_mapping  = mapping(news_text)
			for i,v in enumerate(number):
				print('Feature: %0d, Score: %.5f' % (i,v))
			plt.figure(figsize=(10,8))
			plt.bar(name, number)
			plt.grid()
			st.set_option('deprecation.showPyplotGlobalUse', False)
			st.pyplot()
			def freq(str):
				str = str.split()         
				str2 = []
				for w in str:
					if w not in str2:
						str2.append(str.count(w))
				return str2
			frequency = freq(news_text)
			def dataframe(dicts):
				word = dicts.keys()
				values = dicts.values()
				table = zip(word,values,frequency)
				table = list(table)
				new_dataframe = pd.DataFrame(table, columns =['word','number','frequency'])
				new_dataframe = new_dataframe.sort_values(by='number',ascending = False)
				return new_dataframe
			new_df = dataframe(feature_coef_mapping)
			st.dataframe(new_df)

	if choice == 'NLP':
		st.info("Natural Language Processing of Text")
		raw_text = st.text_area("Enter News Here","Type Here", height= 250)
		nlp_task = ["Stopwords"]
		task_choice = st.selectbox("NLP Task",nlp_task)
		if st.button("Process"):
			docx = nlp(raw_text)
			if task_choice == 'Stopwords':
				result = [' '.join([word for word in raw_text.split() if word not in (stop)])]
			st.json(result)
		if st.checkbox("WordCloud"):
			c_text = raw_text
			wordcloud = WordCloud().generate(c_text)
			plt.imshow(wordcloud)
			st.set_option('deprecation.showPyplotGlobalUse', False)
			plt.axis("off")
			st.pyplot()









	# st.sidebar.subheader("About")




if __name__ == '__main__':
	main()

# By Jesse E.Agbe(JCharis)
# Jesus Saves@JCharisTech