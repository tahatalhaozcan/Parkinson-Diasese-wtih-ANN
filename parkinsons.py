# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 14:37:49 2023

@author: tahat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

#veri seti yükleme
parkinson = pd.read_csv("C:/Users/tahat/PARKINSON-ANN/parkinsons.csv")
df = parkinson.copy()

#veri seti betimsel istatikleri ve değişken bilgileri
df.describe().T
df.info()

#eksik değer sorgulaması
df.isnull().sum() 

df=df.drop(columns=['name'],axis=1) #name değişkeninin veri setinden çıkarılması
for column in df.columns:
    q1, q3 = np.percentile(df[column], [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)] #aykırı değerlerin gösterimi
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)] #aykırı değerlerin veri setinden atılması

df.status.value_counts() # Kişilerden kaçı Parkinson hastası, kaçı Parkinson hastası değil sorgulaması
print("Parkinson Hastası Sayısı: " +str((df.status.value_counts())[1]))
print("Parkinson Olmayan Sayısı: " +str((df.status.value_counts())[0]))

korelasyon = df.corr() #korelasyon matrisi
plt.figure(figsize= (26,18))
sns.set(font_scale=1)
sns.heatmap(korelasyon,annot=True);
plt.show()


x=df.drop(columns=['status'],axis=1) #verisetindeki bağımsız değişkenlerimiz
y=df['status'] #veri setindeki bağımlı değişken(hasta durumu)

#eğitim ve test verilerini ayırmak
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)


#verilerin standartlaştırılması
from sklearn.preprocessing import StandardScaler
sc=StandardScaler() #StandardScaler modulu üzerinden nesne oluşturma 
X_train = sc.fit_transform(x_train) 
X_test = sc.fit_transform(x_test) 

#yapay sinir ağı
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
model1 = Sequential()
model1.add(Dense(12, activation='relu', input_dim=22))
model1.add(Dense(12, activation='relu'))
model1.add(Dense(1, activation='sigmoid'))
model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history1 = model1.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[es])
y_pred = model1.predict(X_test)
y_pred = (y_pred > 0.5)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#dropout ile overfitting durumu engelleme 
model2 = Sequential()
model2.add(Dense(12, activation='relu', input_dim=22))
model2.add(Dropout(0.2))
model2.add(Dense(12, activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ['accuracy'])
history2 = model2.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[es])
y_pred = model2.predict(X_test)
y_pred = (y_pred > 0.5) #hasta olan kişilerin döndürülmesi 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) #karmaşıklık matrisi 
print(cm)


#Grafikler
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('model2 Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

plt.plot(history2.history['accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.title('model2 Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()






