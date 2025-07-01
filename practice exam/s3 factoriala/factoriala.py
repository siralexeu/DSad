import pandas as pd
import seaborn as sns
from factor_analyzer import calculate_kmo, FactorAnalyzer, calculate_bartlett_sphericity
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

df_angajati=pd.read_csv("CAEN2_2021_NSAL.csv",index_col=0)
df_popLoc=pd.read_csv("PopulatieLocalitati.csv",index_col=0)
#1. Să se determine pentru fiecare localitate procentele de angajați pe fiecare ramură
coduri = list(df_angajati)[0:]

#A--------------------------------------------------------------------------------------------------------------------------------------------------
# 1. Să se determine pentru fiecare localitate procentele de angajați pe fiecare ramură
total = df_angajati.sum()
cerinta1 = (df_angajati / total) * 100
cerinta1.round(2).to_csv("cerinta1.csv")

#A. 2. Să se determine numărul de angajați la 100000 locuitori la nivel de județ și pentru fiecare ramură
def fc(t):
    suma = t[coduri].sum()
    suma1 = t["Populatie"].sum()
    x = (suma * 10000) / suma1
    return pd.Series(x, coduri)

df_merged=df_angajati.merge(df_popLoc,left_index=True,right_index=True)
df_merged.index.name="Siruta"
df_grouped=df_merged.groupby("Judet").apply(func=fc)
df_grouped.round(2).to_csv("cerinta2.csv")


#B--------------------------------------------------------------------------------------------------------------------------------------------------------
# Să se efectueze analiza factorială,fără rotație de factori,
data=df_angajati.iloc[:,0:]
print(data)
scaler=StandardScaler()
df_scaled=pd.DataFrame(
    scaler.fit_transform(data),
    index=data.index,
    columns=data.columns
)
#print(df_scaled)
#1. Aplicarea testului KMO. Se vor calcula și se vor afișa la consolă indecșii KMO.
kmo_all,kmo_model=calculate_kmo(df_scaled)
print("Valoarea kmo:",kmo_all)
print(kmo_model)

#1. Aplicarea testului Bartlett
chi_square, p_value = calculate_bartlett_sphericity(df_scaled)
print(f"Valoarea chi-square: {chi_square}")
print(f"P-Value: {p_value}")

#2. Scorurile factoriale. Vor fi salvate în fișierul f.csv.
# fa=FactorAnalyzer(rotation=None, n_factors=len(df_scaled.columns))
# fa.fit(df_scaled)

fa=FactorAnalyzer(rotation=None, n_factors=5)#sau alegem manual n_factors=5
scores=fa.fit_transform(df_scaled)

scores_matrix=pd.DataFrame(
    data=scores,
    index=df_scaled.index,
    columns=[f"F{i+1}" for i in range(scores.shape[1])]
)
scores_matrix.round(2).to_csv("f.csv")

# 3. Graficul scorurilor factoriale pentru primii doi factori.
sns.scatterplot(
    data=scores_matrix,
    x="F1",
    y="F2",
    alpha=0.7
)
#plt.show()

#ALTE CERINTE
#Corelatii matrice
cor_matrix = pd.DataFrame(
    data=fa.loadings_,
    index=df_scaled.columns,
    columns=[f"Factor{i+1}" for i in range(5)]
)
#print(cor_matrix)

table_variance = pd.DataFrame({
    "Var": fa.get_factor_variance()[0],
    "Proc": fa.get_factor_variance()[1],
    "CumProc": fa.get_factor_variance()[2]
})

#print(table_variance)

#B. Să se efectueze analiza factorială pentru indicii de diversitate, cu rotație Varimax
# data=df_angajati.iloc[:,0:]
# print(data)
# scaler=StandardScaler()
# df_scaled=pd.DataFrame(
#     scaler.fit_transform(data),
#     index=data.index,
#     columns=data.columns
# )
#1. Varianța factorilor comuni. Se va salva tabelul varianței în fișierul Varianta.csv
fa_rotatie = FactorAnalyzer(rotation="varimax", n_factors=5)
fa_rotatie.fit(df_scaled)
varianta = fa_rotatie.get_factor_variance()

varianta_matrix = pd.DataFrame({
        'SS Loadings': varianta[0],
        'Proportion Var': varianta[1],
        'Cumulative Var': varianta[2]
})

varianta_matrix.round(3).to_csv("Varianta.csv",index=False)

#2. Corelațiile factoriale (corelațiile variabile - factori comuni).
coreletii_factoriale_cu = fa_rotatie.loadings_
corelatii_matrix = pd.DataFrame(
    coreletii_factoriale_cu,
    columns=[f'Factor{i+1}' for i in range(fa_rotatie.n_factors)],
    index=data.columns
)
#print("Corelatii factoriale cu rotatie: ",coreletii_factoriale_cu)
corelatii_matrix.round(3).to_csv("Corelatii.csv",index=False)

#3. Trasarea cercului corelațiilor pentru primii doi factori comuni
x = coreletii_factoriale_cu[:, 0]
y = coreletii_factoriale_cu[:, 1]
fig,ax = plt.subplots(figsize=(6,6))

circle = plt.Circle((0,0),1, fill=False,)
ax.add_patch(circle)
plt.scatter(x,y)

plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)

plt.xlim(-1,1)
plt.ylim(-1,1)

plt.grid()
#plt.show()


# Calcul comunalitati si varianta specifica
comunalitati = fa.get_communalities()
varianta_specifica = 1-comunalitati
print("Comunalitati cu rotatie: ", comunalitati)
print("Varianta specifica: ", varianta_specifica)

# Trasare corelograma comunalitati si varianta specifica
plt.figure(figsize=(10,5))
sns.barplot(x=data.columns, y=comunalitati)
plt.xticks(rotation=90)
plt.show()

# Calcul scoruri
scoruri_fara = fa.transform(data)
scoruri_fara_df = pd.DataFrame(scoruri_fara, columns=[f"Scor{i+1}" for i in range(0,scoruri_fara.shape[1])])
print("Scoruri fara rotatie", scoruri_fara_df)

# Trasare plot scoruri
# fara rotatie
plt.figure(figsize=(10,8))
plt.scatter(scoruri_fara[:, 0], scoruri_fara[:, 1], c="Blue", alpha=0.8)
plt.show()
