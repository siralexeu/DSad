import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
import seaborn as sns

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#A. 1.Să se determine localitățile cu cifra de afaceri mai mare decât valoarea medie a cifrei de afaceri pe țară.
df_indicatori=pd.read_csv("Indicatori.csv")
df_populatie = pd.read_csv("PopulatieLocalitati.csv")

media=df_indicatori["CFA"].mean()

localitati_filtrate=df_indicatori[df_indicatori["CFA"]>media]

localitati_sortate=localitati_filtrate.sort_values("CFA",ascending=False)

cerinta1 = localitati_sortate[["SIRUTA", "NR_FIRME", "NSAL", "CFA", "PROFITN", "PIERDEREN"]]

cerinta1.to_csv("Cerinta1.csv", index=False)

#A. 2.Să se determine valorile indicatorilor raportate la populație (la 1000 locuitori), la nivel de județ
df_merged = df_indicatori.merge(df_populatie, on="SIRUTA", how="left")

# Grupăm datele la nivel de județ și calculăm suma valorilor indicatorilor și suma populației
df_grouped = df_merged.groupby("Judet", as_index=False).agg({
    "NR_FIRME": "sum",
    "NSAL": "sum",
    "CFA": "sum",
    "PROFITN": "sum",
    "PIERDEREN": "sum",
    "Populatie": "sum"
})

# Calculăm valorile indicatorilor raportate la populație (la 1000 locuitori)
df_grouped["NR_FIRME"] = (df_grouped["NR_FIRME"] * 1000 / df_grouped["Populatie"]).round(3)
df_grouped["NSAL"] = (df_grouped["NSAL"] * 1000 / df_grouped["Populatie"]).round(3)
df_grouped["CFA"] = (df_grouped["CFA"] * 1000 / df_grouped["Populatie"]).round(3)
df_grouped["PROFITN"] = (df_grouped["PROFITN"] * 1000 / df_grouped["Populatie"]).round(3)
df_grouped["PIERDEREN"] = (df_grouped["PIERDEREN"] * 1000 / df_grouped["Populatie"]).round(3)

# Selectăm doar coloanele relevante
cerinta2 = df_grouped[["Judet", "NR_FIRME", "NSAL", "CFA", "PROFITN", "PIERDEREN"]]
cerinta2.to_csv("Cerinta2.csv", index=False)

#B--------------------------------------------------------------------------------------------------------------------------------------------------------------------
df_locatiiq=pd.read_csv("LocationQ.csv")
#ani=df_locatiiq.columns[2:]
ani=['2008', '2009' , '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017',
       '2018', '2019', '2020', '2021']

#print(ani)
df_diversitate=df_locatiiq[ani]
#print(df_diversitate)
#1. Matricea ierarhie cu informații privind joncțiunile făcute
Z=linkage(df_diversitate,method="ward")
matrix=pd.DataFrame(
    data=Z,
    columns=["Cluster1", "Cluster2", "Distanta", "Frecventa"]
)
#print(matrix)
#2. Graficul dendrogramă pentru partiția optimală
dendrogram(Z)
plt.show()

# #3. Componența partiției optimale
K = 3
clusteri = fcluster(Z, t=K, criterion='maxclust')

df_clustered = df_locatiiq.copy()
df_clustered['Cluster'] = clusteri
df_clustered[['Cluster']].to_csv("popt.csv", index=True)

# Calculăm Silhouette Score avg
silhouette = silhouette_score(df_diversitate,clusteri)
print(f"Silhouette Score: {silhouette}")

#alta problema

#2. Componența partiției formate din 3 clusteri. Pentru fiecare instanță se va specifica clusterul din care face parte și scorul Silhouette al instanței.
# Partiția va fi salvată în fișierul p3.csv, pe 4 coloane: codul de țară, denumirea țării, clusterul din care face parte și scorul Sihouette.
# Calculăm scorurile Silhouette pentru fiecare instanță
silhouette_vals = silhouette_samples(df_diversitate, clusteri)
df_clustered['Silhouette'] = silhouette_vals

output = df_clustered[['Judet', 'Cluster', 'Silhouette']]
output.to_csv("p3.csv", index=False)

#3. Graficul histogramă pentrtru partiția din 3 clusteri și variabila GNI. in loc de 2008 pun GNI
sns.histplot(data=df_clustered, x='2008', hue='Cluster', bins=20, kde=True)
plt.show()




