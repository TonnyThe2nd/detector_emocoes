import pandas as pd

base_emotion = pd.read_csv('text.csv')

emocoes = {
    0:'Você está triste!',
    1:'Você está feliz!', 
    2:'Você está apaixonado!',  
    3:'Você está com raiva!', 
    4:'Você está com medo!',
    5:'Você está surpreso'
}
base_emotion['label'] = base_emotion['label'].map(emocoes)

#display(base_emotion)
from sklearn.feature_extraction.text import TfidfVectorizer

emotion_codification = TfidfVectorizer()

x_emotion = base_emotion.iloc[:, 1]

x_emotion = emotion_codification.fit_transform(x_emotion)

y_emotion = base_emotion.iloc[:,2]

#print(x_emotion[1].toarray())
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

#x_emotion_treino, x_emotion_teste, y_emotion_treino, y_emotion_teste = train_test_split(x_emotion,y_emotion, test_size=0.75, random_state=0)

import pickle

#with open('emotios.pkl', mode='wb') as f:
#    pickle.dump([x_emotion_treino, x_emotion_teste, y_emotion_treino, y_emotion_teste],f)

#forest_emotion = DecisionTreeClassifier(criterion = 'entropy', random_state=0)
#forest_emotion.fit(x_emotion_treion,y_emotion_treino)
#with open('modelo_emocao.pkl', mode='wb') as f:
#   pickle.dump(forest_emotion, f)

with open('emotios.pkl', mode='rb') as f:
    x_emotion_treino, x_emotion_teste, y_emotion_treino, y_emotion_teste = pickle.load(f)


with open('modelo_emocao.pkl', mode='rb') as f:
    forest_emotion = pickle.load(f)
previsao = forest_emotion.predict(x_emotion_teste)

frase = input('Digite o seu sentimento (em inglês): ')

frase_transformada = emotion_codification.transform([frase])

previsao = forest_emotion.predict(frase_transformada)

print(previsao[0])
