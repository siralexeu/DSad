
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import average
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns

df=pd.read_csv("Pacienti.csv",index_col=0)
df_apply=pd.read_csv("Pacienti_apply.csv",index_col=0)
#1. Să se aplice analiza liniară discriminantă și să se calculeze scorurile discriminante.
x=df.drop(columns="DECISION")
y=df["DECISION"]
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.3,random_state=0)

lda=LinearDiscriminantAnalysis()
lda.fit(x_train,y_train)
scoruri=lda.transform(x_train)
scoruri_matrice=pd.DataFrame(
    scoruri,
    columns=[f"LD{i+1}"for i in range(scoruri.shape[1])]
)
#print(scoruri_matrice)
#2. Să se traseze graficul scorurilor discriminante în primele două axe discriminante.
sns.scatterplot(
    scoruri_matrice,
    x="LD1",
    y="LD2",
    alpha=0.7
)
#plt.show()
#3. Să se analizeze performanțele modelului calculând matricea de confuzie și indicatorii de acuratețe
y_predictie=lda.predict(x_test)

matrice_confuzie=confusion_matrix(y_test,y_predictie)
matrice_confuzie_df=pd.DataFrame(
    matrice_confuzie,
    index=lda.classes_,
    columns=lda.classes_
)
print(matrice_confuzie_df)

accuracy=accuracy_score(y_test,y_predictie)
precision=precision_score(y_test,y_predictie, average='macro')
recall=recall_score(y_test,y_predictie, average='macro')
f1=f1_score(y_test,y_predictie, average='macro')

print("Acuratețea modelului: ",accuracy)
print("Precizie medie:", precision)
print("Recall mediu:", precision)
print("F1-Score mediu:", f1)

