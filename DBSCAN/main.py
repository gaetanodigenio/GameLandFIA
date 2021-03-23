import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Leggiamo il dataset
df = pd.read_csv('../../Desktop/FIA/datasetFinito.csv')
#Non consideriamo la colonna "titolo" non essendo importante ai fini dell'algoritmo
X = df.drop(['Titolo'], axis=1)

#Eseguiamo l'algoritmo DBSCAN
db = DBSCAN(eps=12.6,min_samples=6).fit(X)

#Assegniamo le labels del cluster per ogni campione. Ai campioni rumorosi viene assegnata l'etichetta -1.
X['Labels'] = db.labels_
#Settiamo il grafico
plt.figure(figsize=(12,8))
#Disegniamo il grafico utilizzando come x = AnnoUscite e come y = Prezzo
sns.scatterplot(X['AnnoUscita'], X['Prezzo'], hue=X['Labels'], palette=sns.color_palette('hls', np.unique(db.labels_).shape[0]))
#Impostiamo il titolo del grafico
plt.title('DBSCAN epsilon = 12.6 e minPts=6')
#Mostriamo il grafico
plt.show()
