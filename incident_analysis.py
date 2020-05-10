from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	df_a = pd.read_csv(r'C:\Users\SanjayMishra\Desktop\DATA Science\Data_science_Projects\multi-class-text-analysis\active_incident.csv',encoding="latin-1")# Dataset is now stored in a Pandas Dataframe
	df_b = pd.read_csv(r'C:\Users\SanjayMishra\Desktop\DATA Science\Data_science_Projects\multi-class-text-analysis\archive_incident.csv',encoding="latin-1")# Dataset is now stored in a Pandas Dataframe
	#df_c = pd.read_csv(r'C:\Users\SanjayMishra\Desktop\DATA Science\Data_science_Projects\multi-class-text-analysis\ar_incident_old.csv',encoding="latin-1")# Dataset is now stored in a Pandas Dataframe
	#df_c = df_c.drop(axis =1 ,columns='sys_archived')
	#df_b = df_b.drop(axis =1 ,columns='sys_archived')
	#Concatinate all data frames
	frames = [df_a,df_b]
	df = pd.concat(frames)
	#No need of other columns apart from these two
	df_x= df[['short_description','cmdb_ci']]
	df_x = df_x.dropna(axis=0)
	#function encode the object as an enumerated type or categorical variable
	df_x['category_id'] = df_x['cmdb_ci'].factorize()[0]
	#create a dataframe with distinct category_id value and their respective cmdb_ci
	category_id_df = df_x[['cmdb_ci', 'category_id']].drop_duplicates().sort_values('category_id')
	#Transforms text to feature vectors that can be used as input to estimator using TfidfVectorizer.
	tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
	features = tfidf.fit_transform(df_x.short_description).toarray()
	labels = df_x.category_id
	X_train_svc,X_test_svc,y_train_svc,y_test_svc = train_test_split(features, labels,test_size=0.30,random_state=42)
	clf = LinearSVC().fit(X_train_svc, y_train_svc)

	if request.method == 'POST':
		message = request.form['message']
		vect = tfidf.transform([message])
		my_prediction = clf.predict(vect)
		category_id_df = category_id_df[category_id_df.values == my_prediction]

	return render_template('result.html',tables=[category_id_df.to_html(classes='data')], titles=category_id_df.columns.values)

if __name__ == '__main__':
	app.run(debug=True)
