import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.constants import alpha
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("Netflix.csv",index_col=0)
data = df.loc[:, "Librarie":"IndiceEducatie"]
#print(data)








#B.--------------------------------------------------------------------------------------------------------------------------------------------------------------------
scaler=StandardScaler()
data_standardizat=pd.DataFrame(
    scaler.fit_transform(data)
)
pca=PCA()
pca.fit(data_standardizat)
#1. Varianțele componentelor principale. Varianțele vor fi afișate la consolă
varinata_pca=pca.explained_variance_
varinata_pca_ratio=pca.explained_variance_ratio_
print(varinata_pca,varinata_pca_ratio)

#2. Scorurile asociate instanțelor. Scorurile vor fi salvate în fișierul scoruri.csv.
scoruri_pca=pca.transform(data_standardizat)
matrice_scoruri=pd.DataFrame(
    scoruri_pca,
    index=data.index,
    columns=[f"PC{i+1}"for i in range(scoruri_pca.shape[1])]
)
print(matrice_scoruri)

#3. Graficul scorurilor în primele două axe principale.
sns.scatterplot(
    matrice_scoruri,
    x="PC1",
    y="PC2",
    alpha=0.7
)
plt.show()
