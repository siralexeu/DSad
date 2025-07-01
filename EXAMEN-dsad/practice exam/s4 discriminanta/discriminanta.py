import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

df_ensal=pd.read_csv("E_NSAL_2008-2021.csv",index_col=0)
df_localitati=pd.read_csv("PopulatieLocalitati.csv",index_col=0)
#ani=list(df_ensal)[0:]
ani=['2008','2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017',
       '2018', '2019', '2020', '2021']

#A---------------------------------------------------------------------------------------------------------------------
#1. Să se determine pentru fiecare localitate anul în care care au fost înregistrați cei mai mulți angajați.
df_ensal["Anul"]=df_ensal[ani].idxmax(axis=1)
cerinta1=df_ensal[["Anul"]]
cerinta1.to_csv("cerinta1.csv")

#2. Să se determine rata ocupării populației pe fiecare an și cea medie (media ratei anilor) la nivel de județ.
df_merge=df_ensal.merge(df_localitati[["Judet","Populatie"]],left_index=True,right_index=True)
#print(df_merge)
df_merge=df_merge.groupby("Judet").sum()
for var in ani:
    df_merge[var]=df_merge[var]/df_merge["Populatie"]
df_merge["Rata medie"]=df_merge[ani].mean(axis=1)
df_merge = df_merge[ ani + ["Rata medie"]]
cerinta2 = df_merge.sort_values(by="Rata medie",ascending=False)
cerinta2.round(3).to_csv("cerinta2.csv")

#B---------------------------------------------------------------------------------------------------------------------
df_pacienti=pd.read_csv("Pacienti.csv",index_col=0)
df_pacienti_apply=pd.read_csv("Pacienti_apply.csv",index_col=0)

#criteria=list(df_pacienti)[:7]
#print(criteria)

#1. Să se aplice analiza liniară discriminantă și să se calculeze scorurile discriminante
X=df_pacienti.drop(columns="DECISION")
Y=df_pacienti["DECISION"]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.4,random_state=0)

lda=LinearDiscriminantAnalysis()
lda.fit(X_train,Y_train)

scores=lda.transform(X_train)
score_dataframe=pd.DataFrame(
    scores,
    columns=[f"LD{i+1}"for i in range(scores.shape[1])]
)
score_dataframe.to_csv("z.csv",index=False)
score_dataframe["DECISION"]=Y_train.values

#2. Să se traseze graficul scorurilor discriminante în primele două axe discriminante.
sns.scatterplot(
    data=score_dataframe,
    x="LD1",
    y="LD2",
    alpha=0.7
)
plt.show()

#3. Să se analizeze performanțele modelului calculând matricea de confuzie și indicatorii de acuratețe.
#Matricea de acuratețe va fi salvată în fișierul matc.csv, iar indicatorii de acuratețe vor fi afișați la consolă.
prediction_test=lda.predict(X_test)

matrice_confuzie=confusion_matrix(Y_test,prediction_test)
#print(matrice_confuzie)
df_matrice_confuzie=pd.DataFrame(
    matrice_confuzie,
    index=lda.classes_,
    columns=lda.classes_
)
df_matrice_confuzie.to_csv("matc.csv", index=False)

#ALTE CERINTE POSIBILE
# Calcularea acurateței globale
score=accuracy_score(Y_test,prediction_test)
print(score)

# Calcularea acurateței medii
raport = classification_report(Y_test, prediction_test, output_dict=True)
acuratete_medie = np.mean([raport[clasa]['recall'] for clasa in lda.classes_])
print(acuratete_medie)


#3. ALTERNATIV sa se efectueze predictiile atat in test cat si in aplicare
prediction_applied=lda.predict(df_pacienti_apply)
pd.DataFrame(data=prediction_test).to_csv("predict_test.csv")
pd.DataFrame(data=prediction_applied).to_csv("predict_apply.csv")

#3.Graficul distributiei in axe
sns.kdeplot(
    data=score_dataframe,
    x="LD1",
    y="LD2",
    fill=True
)
plt.show()