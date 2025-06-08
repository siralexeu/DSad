import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples
from sklearn.decomposition import PCA
from seaborn import scatterplot
from scikitplot.metrics import plot_silhouette

np.set_printoptions(3, threshold=sys.maxsize, suppress=True)

#functii
def nan_replace_t(t: pd.DataFrame):
    for coloana in t.columns:
        if t[coloana].isna().any():
            if pd.api.types.is_numeric_dtype(t[coloana]):
                t.fillna({coloana: t[coloana].mean()}, inplace=True)
            else:
                t.fillna({coloana: t[coloana].mode()[0]}, inplace=True)

def elbow(h: np.ndarray, k=None):
    n = h.shape[0] + 1
    if k is None:
        d = h[1:, 2] - h[:n - 2, 2]
        nr_jonctiuni = np.argmax(d) + 1
        k = n - nr_jonctiuni
    else:
        nr_jonctiuni = n - k
    threshold = (h[nr_jonctiuni, 2] + h[nr_jonctiuni - 1, 2]) / 2
    return k, threshold

def calcul_partitie(x, k):
    model_hclust = AgglomerativeClustering(k)
    c = model_hclust.fit_predict(x)
    partitie = np.array(["C" + str(v + 1) for v in pd.Categorical(c).codes])
    return partitie

def plot_ierarhie(h: np.ndarray, titlu, color_threshold=0, etichete=None):
    fig = plt.figure("Dendrograma - " + titlu, figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(titlu, fontsize=16)
    dendrogram(h, ax=ax, color_threshold=color_threshold, labels=etichete)

def plot_scoruri_silhouette(x, p, titlu):
    fig = plt.figure("Silh." + titlu, figsize=(8, 8))
    fig.suptitle(titlu)
    ax = fig.add_subplot(1, 1, 1)
    plot_silhouette(x, p, ax=ax)

def plot_partitie(z: np.ndarray, p, titlu, etichete=None):
    fig = plt.figure("Scatter_" + titlu, figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Plot partitie. " + titlu, fontdict={"fontsize": 16})
    ax.set_xlabel("Z0")
    ax.set_ylabel("Z1")
    scatterplot(x=z[:, 0], y=z[:, 1], hue=p, hue_order=np.unique(p), ax=ax)
    if etichete is not None:
        n = len(etichete)
        for i in range(n):
            ax.text(z[i, 0], z[i, 1], etichete[i])

def plot_histograme(t: pd.DataFrame, variabila, p, titlu):
    fig = plt.figure("H_" + titlu + "_" + variabila, figsize=(13, 6))
    fig.suptitle("Histograme pentru variabila " + variabila + ". " + titlu)
    etichete_clusteri = np.unique(p)
    q = len(etichete_clusteri)
    axe = fig.subplots(1, q, sharey=True)
    for i in range(q):
        ax = axe[i]
        assert isinstance(ax, plt.Axes)
        y = t[variabila].values[p == etichete_clusteri[i]]
        ax.hist(y, 10, range=(t[variabila].min(), t[variabila].max()), rwidth=0.9)
        ax.set_xlabel(etichete_clusteri[i])

def show():
    plt.show()

#main
set_date = pd.read_csv("data_in/exporturi.csv", index_col=0)
#print(set_date)
nan_replace_t(set_date)

# Extragem valorile pentru datele numerice
indicatori = list(set_date)
x = set_date[indicatori].values

# Aplicam PCA pentru a reduce dimensiunea la 2 componente
pca = PCA(2)
z = pca.fit_transform(x)

# Construire ierarhie
metoda = "ward"
h = linkage(x, metoda)
t_h = pd.DataFrame(h, columns=["Cluster 1", "Cluster 2", "Distanta", "Frecventa"])
t_h.index.name = "Jonctiune"
t_h.to_csv("data_out/ierarhie.csv")

# Plotam dendrograma pentru ierarhie
plot_ierarhie(h, "Plot Ierarhie", etichete=set_date.index)
show()

# Determinam partitia optima folosind algoritmul Elbow
k_opt, color_threshold_opt = elbow(h)
print(f"Numarul optim de clustere este: {k_opt}") #2
p_opt = calcul_partitie(x, k_opt)

# Cream un tabel cu partitiile si scorurile Silhouette
tabel_partitii = pd.DataFrame(index=set_date.index)
tabel_partitii["P_Opt"] = p_opt
tabel_partitii["P_Opt_Silh"] = silhouette_samples(x, p_opt)
silh_opt = np.mean(tabel_partitii["P_Opt_Silh"])
print(f"Scorul Silhouette pentru partitia optima este: {silh_opt:.3f}")

# Plotam rezultatele pentru partitia optima
plot_ierarhie(h, "Partia optimala", color_threshold_opt, set_date.index)
plot_scoruri_silhouette(x, p_opt, "Partitia optimala")
plot_partitie(z, p_opt, "Partia optimala")
for indicator in indicatori:
    plot_histograme(set_date, indicator, p_opt, "Partitia optimala")
show()

#acum pt 3 clustere
k_3 = 3
k_3_, color_threshold_3 = elbow(h, k_3)
p_3 = calcul_partitie(x, k_3)

# Adaugam rezultatele pentru partitia cu 3 clustere
tabel_partitii["P_3"] = p_3
tabel_partitii["P_3_Silh"] = silhouette_samples(x, p_3)
silh_3 = np.mean(tabel_partitii["P_3_Silh"])
print(f"Scorul Silhouette pentru partitia cu 3 clustere este: {silh_3:.3f}")

# Plotam rezultatele pentru partitia cu 3 clustere
plot_ierarhie(h, "Partia din 3 clusteri", color_threshold_3, set_date.index)
plot_scoruri_silhouette(x, p_3, "Partia din 3 clusteri")
plot_partitie(z, p_3, "Partia din 3 clusteri")
for indicator in indicatori:
    plot_histograme(set_date, indicator, p_3, "Partia din 3 clusteri")
show()

#salvam
tabel_partitii.to_csv("data_out/partitii.csv")
