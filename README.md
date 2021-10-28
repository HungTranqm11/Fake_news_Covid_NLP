# Fake_news_Covid_NLP
## Technical for model: Pandas, TF-IDF, Logistic Regression, KNeighborsClassifier , SupportVectorClassifier, RandomForestClassifier, Stacking Ensemble
## Technical for EDA: Stopword, Lemmatizer, streamming
## Task Description
- Task: identification fake news covid
- Dataset: 
  + Real label: 4408
  + Fake label: 1133
  + You can download dataset here: https://github.com/bigheiniu/MM-COVID
## Implement
## Base model evaluation
### Logistic Regression, KNeighborsClassifier, SupportVectorClassifier, RandomForestClassifier,
![image](https://user-images.githubusercontent.com/85773711/139311022-1a9344ed-06d4-4908-a30f-64f52981a665.png)
<img src="https://user-images.githubusercontent.com/85773711/139311713-f6b54a66-826e-400a-bb81-ad477869398a.png" width="700" heigh="700" align="center"/>
<!--[image](https://user-images.githubusercontent.com/85773711/139311713-f6b54a66-826e-400a-bb81-ad477869398a.png)-->
- all base model have > 97% on TP and > 45% on TN
### Stacking Ensemble
![image](https://user-images.githubusercontent.com/85773711/139312309-4d01e563-d3ad-40c2-9aa5-8d92cfe4e5e9.png)
- after stacking model logistic regression and KNeighborsClassifier accuracy improved 
## Future improvement
- Try another ensemble like bagging and boosting.
- Collect more data.
- Fine tune model find best param.
- Expand model predict like fake news, prediction true false.



