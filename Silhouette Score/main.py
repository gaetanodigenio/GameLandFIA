import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples,silhouette_score

#Leggiamo il dataset
df = pd.read_csv('../../Desktop/FIA/datasetFinito.csv')
#Non consideriamo la colonna "titolo" non essendo importante ai fini dell'algoritmo
X = df.drop(['Titolo'], axis=1)

#Creiamo una lista che contenga tutte le somme di distanza quadrata
# di ogni campione dal centro del cluster più vicino
clusterErrors = []

for i in range(2, 30):
    #Eseguiamo l'algoritmo KMeans
    km=KMeans(n_clusters=i, max_iter=1000).fit(X)
    #Aggiungiamo alla lista la somma di distanza quadrata di ogni campione
    # dal centro del cluster più vicino
    clusterErrors.append(km.inertia_)
    #Calcoliamo i centri del cluster e
    # prediciamo l'indice del cluster per ogni campione
    y_predict = km.fit_predict(X)
    #Assegniamo le coordinate dei centri dei cluster (i centroidi)
    centroids = km.cluster_centers_
    #Prevediamo il cluster più vicino a cui appartiene ogni campione presente in X
    label = km.predict(X)
    #Stampiamo il Silhouette Score
    print(f'Silhouette Score(n={i}): {silhouette_score(X,label)}')
