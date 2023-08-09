# importons les packages nécéssaires
import streamlit as st
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
# importons le dataset :
iris = datasets.load_iris()
# transformons les dataset en dataframe
data=pd.DataFrame({
'sepal length': iris.data[:,0],
'sepal width': iris.data[:,1],
'petal length': iris.data[:,2],
'petal width': iris.data[:,3],
'species': iris.target})
# verifions
data.info()
# divisons les données pour le modele
X=data[['sepal length', 'sepal width', 'petal length', 'petal width']]
y=data['species']
# divisons les données en données de test et d' entrainement
x_train, x_test, y_train, y_test= train_test_split(X, y, test_size=0.3)
# le modele
clf=RandomForestClassifier(n_estimators=10)
clf.fit(x_train, y_train)
y_pred=clf.predict(x_test)
# jeaugeons le modele
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("le resultat est satisfaisant car proche de 1")
st.title("CHECKPOINT 1 STREAMLIT")
st.header("PREDICTION IRIS ")
st.subheader("  Moise senghor")
st.subheader(" Data science bootamp")
st.subheader("    Gomycode")
longueursepale = st.slider('Quelle est la longueur du sepale',data['sepal length'].min() , data['sepal length'].max(), data['sepal length'].mean())
st.write("la longueur du sépale est de ", longueursepale)
largeursepale = st.slider('Quelle est la largeur du sepale',data['sepal width'].min() , data['sepal width'].max(), data['sepal width'].mean())
st.write("la largeur du sépale est de ", largeursepale)
longueurpetale = st.slider('Quelle est la longueur de la  petale',data['petal length'].min() , data['petal length'].max(), data['petal length'].mean())
st.write("la longueur de la pétale  est de ", longueurpetale)
largeurpetale = st.slider('Quelle est la largeur de la  petale',data['petal width'].min() , data['petal width'].max(), data['petal width'].mean())
st.write("la longueur de la pétale  est de ", largeurpetale)
st.write("D' apres le modele la fleur sera de type  ", clf.predict([[longueursepale, largeursepale, longueurpetale, largeurpetale]]), "0 = setosa , 1 = versicolor, 2 = virginica")
st.text("Essayons de prédire le type de fleur en entrant les valeurs ")
st.text("des sepales et des petales .")
st.text("Entrer la longueur du sepale :")
Ls = st.number_input('Entrer la valeur',value=0,key=0)
st.write('Longueur sepale ', Ls)
st.text("Entrer la largeur du sepale :")
ls = st.number_input('Entrer la valeur',value=0,key=1)
st.write('Largeur sepale ', ls)
st.text("Entrer la longueur de la petale :")
Lp = st.number_input('Entrer la valeur', value=0,key=2)
st.write('Longueur petale ', Lp)
st.text("Entrer la longueur de la petale :")
lp = st.number_input('Entrer la valeur',value=0,key=3)
st.write('Largeur petale ', lp)
predict = clf.predict([[Ls, ls, Lp, lp]])
if predict == 0 :
    st.write('Selon le modele la fleur devrait etre setosa.')
if predict == 1 :
    st.write('Selon le modele la fleur devrait etre versicolor.')
else :
    st.write('Selon le modele la fleur devrait etre virginica.')

