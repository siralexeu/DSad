
import pandas as pd
import seaborn as sns
from factor_analyzer import calculate_kmo, calculate_bartlett_sphericity, FactorAnalyzer
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("CAEN2_2021_NSAL.csv",index_col=0)
data=df.iloc[:,0:]
#print(data)
scaler=StandardScaler()
data_standardizata=pd.DataFrame(
    scaler.fit_transform(data),
    index=data.index,
    columns=data.columns
)
#B. Să se efectueze analiza factorială, fără rotație de factori
#1. Aplicarea testului KMO. Se vor calcula și se vor afișa la consolă indecșii KMO.
kmo_all,kmo_model=calculate_kmo(data_standardizata)
print(kmo_all,kmo_model)

#1. Aplicarea testului Bartlett
chi_square,p_value=calculate_bartlett_sphericity(data_standardizata)
print(chi_square,p_value)

#2. Scorurile factoriale. Vor fi salvate în fișierul f.csv.
fa=FactorAnalyzer()
scoruri=fa.fit_transform(data_standardizata)
matrice_scoruri=pd.DataFrame(
    data=scoruri,
    index=data.index,
    columns=[f"F{i+1}"for i in range(scoruri.shape[1])]
)
print(matrice_scoruri)

# 3. Graficul scorurilor factoriale pentru primii doi factori.
sns.scatterplot(
    data=matrice_scoruri,
    x="F1",
    y="F2",
    alpha=0.7
)
plt.show()