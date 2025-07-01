import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("LocationQ.csv",index_col=0)
data=df.iloc[:,0:]
#print(data)
scaler=StandardScaler()
date_standardizate=pd.DataFrame(
    scaler.fit_transform(data)
)
#1. Matricea ierarhie cu informații privind joncțiunile făcute
Z=linkage(date_standardizate,method="ward")
matrice_ierarhie=pd.DataFrame(
    data=Z,
    columns=["C1","C2","Distanta","Frecventa"]
)
print(matrice_ierarhie)

#2. Graficul dendrogramă pentru partiția optimală.
dendrogram(Z)
#plt.show()

#3. Componența partiției optimale.
k=4
clusteri=fcluster(Z,t=k,criterion="maxclust")\

df_clusteri=df.copy()
df_clusteri["Cluster"]=clusteri
print(df_clusteri[["Cluster"]])


