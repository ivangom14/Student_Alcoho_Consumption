import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import decomposition
from sklearn.pipeline import Pipeline
import transformers
from sklearn import svm
import sys
from sklearn.externals import joblib
import getopt
from sklearn import metrics
import seaborn as sns
from sklearn.model_selection import train_test_split

data = pd.read_csv("student-por.csv", sep=";")
#print("Tamanio dataset inicial", data.shape)


######################################################################################

fig = plt.figure(figsize=(12,9))

a = fig.add_subplot(1,4,1)
img1 = data.groupby(by=["school"]).size().plot.bar(color= ('g'))
a.set_title("Colegio estudiantes")

h = fig.add_subplot(1,4,2)
img11 = data.groupby(by=["reason"]).size().plot.bar(color= ('y'))
h.set_title("Razon eleccion colegio")

b = fig.add_subplot(1,4,3)
img2 = data.groupby(by=["sex"]).size().plot.bar(color= ('r'))
b.set_title("Sexo estudiantes")

c = fig.add_subplot(1,4,4)
plt.boxplot(data["age"], labels=['students'])
c.set_title("Edad estudiantes")



######################################################################
fig2 = plt.figure(figsize=(12,9))

img4 = fig2.add_subplot(1,4,1)
plt.pie(data["address"].value_counts(), startangle=90, autopct='%1.2f%%', labels=["U","R"])
img4.set_title("Tipo casa")

img5 = fig2.add_subplot(1,4,2)
plt.pie(data["famsize"].value_counts(), startangle=90, autopct='%1.2f%%', labels=["GT3","LE3"])
img5.set_title("Tamanio familia")

img6 = fig2.add_subplot(1,4,3)
plt.pie(data["Pstatus"].value_counts(), startangle=90, autopct='%1.2f%%', labels=["T","A"])
img6.set_title("Cohabitantes")

i = fig2.add_subplot(1,4,4)
img12 = data.groupby(by=["guardian"]).size().plot.bar(color= ('r'))
i.set_title("Tutor estudiante")

#####################################################################
fig3 = plt.figure(figsize=(12,9))

d = fig3.add_subplot(1,4,1)
plt.pie(data["Medu"].value_counts(), startangle=90, autopct='%1.2f%%', labels=["Higher","4th","Secondary","5th to 9th","None"])
d.set_title("Estudios madre")

e = fig3.add_subplot(1,4,2)
plt.pie(data["Fedu"].value_counts(), startangle=90, autopct='%1.2f%%', labels=["Higher","4th","5th to 9th","Secondary","None"])
e.set_title("Estudios padre")

f = fig3.add_subplot(1,4,3)
img9 = data.groupby(by=["Mjob"]).size().plot.bar(color= ('g'))
f.set_title("Trabajo madre")

g = fig3.add_subplot(1,4,4)
img10 = data.groupby(by=["Fjob"]).size().plot.bar(color= ('g'))
g.set_title("Trabajo padre")

#######################################################################
fig4 = plt.figure(figsize=(12,9))

j = fig4.add_subplot(1,4,1)
plt.pie(data["traveltime"].value_counts(), startangle=90, autopct='%1.2f%%', labels=["15to30 min", "<15 min","30to1 hour",">1 hour"])
j.set_title("Tiempo viaje")

k = fig4.add_subplot(1,4,2)
plt.pie(data["studytime"].value_counts(), startangle=90, autopct='%1.2f%%', labels=["2to5 hour","5to10 hour", "<2 hour", ">10 hour" ])
k.set_title("Tiempo estudio semanal")

l = fig4.add_subplot(1,4,3)
plt.pie(data["freetime"].value_counts(), startangle=90, autopct='%1.2f%%', labels=["medium","low","high","very low", "very high"])
l.set_title("Tiempo libre")

m = fig4.add_subplot(1,4,4)
plt.pie(data["goout"].value_counts(), startangle=90, autopct='%1.2f%%', labels=["high","medium","low","very low","very high"])
m.set_title("Tiempo amigos")

###############################################################
fig5 = plt.figure(figsize=(12,9))

n = fig5.add_subplot(1,2,1)
plt.pie(data["failures"].value_counts(), startangle=90, autopct='%1.2f%%', labels=["0","3","1","2"])
n.set_title("Faltas a clase")

o = fig5.add_subplot(1,2,2)
plt.boxplot(data["absences"], labels=['absences'])
o.set_title("Fatas a escuela")


####################################################################
fig6 = plt.figure(figsize=(12,9))

p = fig6.add_subplot(1,3,1)
img13 = data.groupby(by=["schoolsup"]).size().plot.bar(color= ('g'))
p.set_title("Apoyo educativo adicional")

q = fig6.add_subplot(1,3,2)
img14 = data.groupby(by=["famsup"]).size().plot.bar(color= ('r'))
q.set_title("Apoyo familiar adicional")

r = fig6.add_subplot(1,3,3)
img15 = data.groupby(by=["nursery"]).size().plot.bar(color= ('y'))
r.set_title("Escuela maternal asistida")

####################################################################
fig7 = plt.figure(figsize=(12,9))

s = fig7.add_subplot(1,3,1)
img16 = data.groupby(by=["paid"]).size().plot.bar(color= ('g'))
s.set_title("Clases extra")

t = fig7.add_subplot(1,3,2)
img17 = data.groupby(by=["activities"]).size().plot.bar(color= ('b'))
t.set_title("Actividades extraescolaresl")

u = fig7.add_subplot(1,3,3)
img18 = data.groupby(by=["internet"]).size().plot.bar(color= ('r'))
u.set_title("Acceso a internet en casa")

####################################################################
fig8 = plt.figure(figsize=(12,9))

v = fig8.add_subplot(1,2,1)
img19 = data.groupby(by=["romantic"]).size().plot.bar(color= ('g'))
v.set_title("Relacion romantica")

w = fig8.add_subplot(1,2,2)
img20 = data.groupby(by=["famrel"]).size().plot.bar(color= ('b'))
w.set_title("Relacion familiar")

####################################################################
fig9 = plt.figure(figsize=(12,9))

x = fig9.add_subplot(1,3,1)
img21 = data.groupby(by=["Dalc"]).size().plot.bar(color= ('g'))
x.set_title("Consumo alcohol dia trabajo")

y = fig9.add_subplot(1,3,2)
img22 = data.groupby(by=["Walc"]).size().plot.bar(color= ('r'))
y.set_title("Consumo alcohol fin de semana")

z = fig9.add_subplot(1,3,3)
img23 = data.groupby(by=["health"]).size().plot.bar(color= ('b'))
z.set_title("Estado salud")

fig10 = plt.figure(figsize=(16,14))
img24 = fig10.add_subplot(1,1,1)
sns.heatmap(data.corr(), annot = True)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
img24.set_title("Correlacion de las caracteristicas")

fig11 = plt.figure(figsize=(16,14))
img25 = fig10.add_subplot(1,1,1)
sns.heatmap(data.cov(), annot = True)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
img25.set_title("Covarianza de las caracteristicas")

fig12 = plt.figure(figsize=(16,14))
img26 = fig10.add_subplot(1,1,1)
sns.heatmap(data.describe(), annot = True)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
img26.set_title("Descripcion de las caracteristicas")

plt.show()


