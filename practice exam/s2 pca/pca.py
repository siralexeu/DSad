import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

df_netflix=pd.read_csv("Netflix.csv",index_col=0)
df_coduri=pd.read_csv("CoduriTari.csv")
variabile = ["Librarie", "CostLunarBasic", "CostLunarStandard", "CostLunarPremium",
             "Internet", "HDI", "Venit", "IndiceFericire", "IndiceEducatie"]

#A--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#1. Să se standardizeze setul de date
scaler = StandardScaler()
df_standardized = df_netflix.copy()
df_standardized[variabile] = scaler.fit_transform(df_netflix[variabile])

cerinta1 = df_standardized.sort_values(by="Internet", ascending=False)
cerinta1.round(3).to_csv("Cerinta1.csv", index=False)

#2. Să se determine coeficienții de variație pentru fiecare indicator la nivel de continent
df_merged=df_netflix.merge(df_coduri,on="Tara",how="left")

cv_results = (
    df_merged.groupby("Continent")[variabile]
    .agg(lambda x: x.std() / x.mean())
    .reset_index()
)

cerinta2 = cv_results.sort_values(by="Librarie", ascending=False)
cerinta2.round(3).to_csv("Cerinta2.csv", index=False)

#B--------------------------------------------------------------------------------------------------------------------------------------------------------------------
data = df_netflix[variabile]
scaler=StandardScaler()
data_standardized=pd.DataFrame(
    scaler.fit_transform(data),
    index=data.index,
    columns=data.columns
)
#aplicam PCA
pca=PCA()#pca=PCA(n_components=2)
pca.fit(data_standardized)

#1. Varianțele componentelor principale
variante_pca=pca.explained_variance_
variante_pca_ratio=pca.explained_variance_ratio_
print(variante_pca.round(2))
print(variante_pca_ratio.round(2))

#2. Scorurile asociate instanțelor
scoruri_pca=pca.transform(data_standardized)
#print(scoruri_pca)
scoruri_df=pd.DataFrame(
    scoruri_pca,
    index=data.index,
    columns=[f"PC{i+1}" for i in range(scoruri_pca.shape[1])]
)
scoruri_df.round(3).to_csv("scoruri.csv")

# 3. Graficul scorurilor în primele două axe principale
sns.scatterplot(
    data=scoruri_df,
    x="PC1",
    y="PC2",
    alpha=0.7,
)
plt.show()

#ALTE CERINTE POSIBILE
# Calcul cosinusuri
componente_principale=pca.components_
componente_patrat=componente_principale**2
suma_var=np.sum(componente_patrat,axis=0)
cosinusuri=componente_patrat/suma_var
print("Cosinusuri:\n", cosinusuri)

# Calcul contributii
contributii = (pca.explained_variance_ratio_ * 100)
print("Contributii:\n", contributii)

# Calculul corelatii factoriale (corelatii variabile observate - componente)
corelatie_factoriala = componente_principale.T * np.sqrt(variante_pca)
print("Corelatii factoriale:\n", corelatie_factoriala)

# Calcul comunalitati
comunalitati = np.sum(corelatie_factoriala**2, axis=1)
print("Comunalitati:\n", comunalitati)

# Trasare corelograma comunalitati
plt.figure(figsize=(10,8))
sns.heatmap(pd.DataFrame(comunalitati, columns=['Comunalități'], index=data.columns),
            annot=True, cmap='viridis', fmt=".2f")
plt.title("Corelograma comunalitati")
#plt.show()