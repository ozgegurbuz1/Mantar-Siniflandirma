"""
Mantar Sınıflandırması 🍄

Bu projede kaggle.com'dan halka açık bir veri seti olan “Mantar Sınıflandırması” veri setini kullanacağız ve bir 
mantarın yenilebilir olup olmadığını anlamaya çalışacağız. Bu veri seti, 23 mantar türüne karşılık gelen varsayımsal 
örneklerin açıklamalarını içerir. Her tür, yenilebilir veya zehirli olarak tanımlanır. Mantar söz konusu olduğunda 
yenilebilirliği belirlemek için basit bir kural yoktur.

Bu sınıflandırma problemini logistic regression, ridge classifier, decision tree, Naive Bayes ve neural networks kullanarak 
çözeceğiz. Her modelin sonuçlarını karşılaştırdıktan sonra, en iyi performansı göstereni bulacağız.


Gerekli kitaplıkları içe aktarma.Her zaman olduğu gibi, gerekli kitaplıkları içe aktarmakla başlayacağız.
📌 "import" ve "from" anahtar sözcüklerini kullanıyoruz.

"""

# Pandas ve Matplotlib içe aktar
import pandas as pd
import matplotlib.pyplot as plt
# Label Encoder ve train_test_split içe aktar
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# Logistic Regression, Ridge Classifier, Decision Tree içe aktar
# Gaussian Naive Bayes, MLP Classifier ve Random Forest modelleri
from sklearn.linear_model import LogisticRegression,RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
# Classification Report fonksiyonu içe aktar
from sklearn.metrics import classification_report

"""
Veriseti ve Önişleme
Veri seti, 8124 mantardan veri içerir. Bu mantar örneklerinin her biri 22 özelliğe sahiptir ve yenilebilir veya zehirli 
olarak sınıflandırılır.

Veri Okuma
.csv dosyasını okuyalım.

📌 Pandas kitaplığının read_csv() işlevini kullanalım.
"""

# "mushroom.csv" dosyasını okuma
data=pd.read_csv("mushrooms.csv")

"""
Veri Görşelleştime
Ardından, *data.head()* işlevini kullanarak veri kümesine bir göz atalım.
"""

# Verilerin ilk 5 satırını görüntülemek için head() işlevini kullanalım
print(data.head())

"""
Şimdi, veri setini daha iyi anlamak için bazı görselleştirme tekniklerini kullanabiliriz. Örneğin bir bar grafik
oluşturarak farklı sınıfları karşılaştırabiliriz.Sınıf başına örnek sayısını bulmakla başlayacağız.

📌 value_counts() methodunu kullanalım.
"""

# Veri nesnesinin "sınıf" sütununda value_counts yöntemini kullanalım
classes=data['class'].value_counts()

# Sonucu yazdıralım
print("\n",classes)

"""
Bu bilgilerle her sınıf için bar grafiği oluşturabilir ve görüntüleyebiliriz.

📌 Grafiği oluşturmak için .bar() yöntemini kullanalım.
📌 plt.show() kullanmayı unutmayalım.
"""

# Yenilebilir sınıf için bar ekleyelim.
plt.bar("Edible",classes["e"])

# Zehirli sınıf için bar ekleyelim.
plt.bar("Poisonous",classes["p"])

# Grafiği yazdıralım.
plt.show()

"""
Özellikler ve Etiketler
Verilerimizi daha iyi anladık. Şimdi onu özelliklere ve karşılık gelen etiketlere ayıracağız.
Bu durumda, özellikler olarak “cap-shape”, “cap-color”, “ring-number” ve “ring-type” sütunlarını kullanacağız.

📌 X ve y verisetlerini oluşturmak için .loc() yöntemini kullanalım.

"""

# Özellikler için X değişkenini oluşturalım.
X=data.loc[:,["cap-shape","cap-color","ring-number","ring-type"]]

# Çıktı etiketleri için y değişkenini oluşturalım.
y=data.loc[:,"class"]

"""
Değerleri dönüştürme

Değerler string biçimindedir. Onlarla matematiksel işlemler yapabilmek için onları integer değerlere 
dönüştürmemiz gerekir. Bunun için etiket kodlamasını kullanacağız.
📌 X verilerinin birden çok sütunu olduğundan, tüm sütunları aynı anda güncelleyebilmemiz için bunu bir for döngüsü içinde yapalım.
📌 y verileri için doğrudan kodlayıcıyı kullanalım.
"""

# Bir LabelEncoder nesnesi oluşturalım
encoder=LabelEncoder()

# Özellikleri bir for döngüsü içindeki integerlara kodlayalım
for i in X.columns:
  X[i]=encoder.fit_transform(X[i])

# Çıkış etiketlerini integerlara kodlayalım
y=encoder.fit_transform(y)

#Son verileri görmek için hem X hem de y'yi yazdıralım.

#X'i yazdıralım.
print("\n X:",X)

# y'yi yazdıralım.
print("\n y:",y)

"""
Verileri Bölme
Son olarak, verilerimizi eğitim ve test veri kümelerine ayırabiliriz.

📌  Sklearn'den train_test_split fonksiyonunu kullanalım.
"""

# Veri setini 70-30 oranında eğitim ve test setlerine ayıralım
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

"""
Model Oluşturma ve Eğitim
Verilerimiz kullanıma hazır! Modellerimizi karşılaştırma eğitimine geçelim.
Halihazırda import ettiğimiz modelleri kullanıyoruz.

📌 Model oluşturmak için ilgili sınıf adlarını kullanalım.
"""

from sklearn import neural_network
# LogisticRegression() sınıfını kullanarak bir nesne oluşturalım
logistic_classifier_model=LogisticRegression()

# RidgeClassifier() sınıfını kullanarak bir nesne oluşturalım
ridge_classifier_model=RidgeClassifier()

# DecisionTreeClassifier() sınıfını kullanarak bir nesne oluşturalım
decision_tree_model=DecisionTreeClassifier()

# GaussianNB() sınıfını kullanarak bir nesne oluşturalım
naive_bayes_model=GaussianNB()

# MLPClassifier() sınıfını kullanarak bir nesne oluşturalım
neural_network_model=MLPClassifier()

"""Daha sonra oluşturduğumuz X_train ve y_train veri seti ile tüm modelleri eğitiyoruz.
📌 Her nesnenin .fit() yöntemini kullanarak tüm modelleri eğitelim.
"""

# Logistic Classifier modelini eğitelim.
logistic_classifier_model.fit(x_train,y_train)

# Ridge Classifier modelini eğitelim.
ridge_classifier_model.fit(x_train,y_train)

# Decision Tree modelini eğitelim.
decision_tree_model.fit(x_train,y_train)

# Naive Bayes modelini eğitelim.
naive_bayes_model.fit(x_train,y_train)

# Neural Network modelini eğitelim.
neural_network_model.fit(x_train,y_train)

"""X_test setini kullanarak her model için tahminler yapıyoruz ve sonuçları karşılık gelen değişkenlere kaydediyoruz.
📌 Her modelde .predict() yöntemini kullanalım.
"""

# Logistic Classifier modelindeki test veri setini kullanarak tahminde bulunalım.
logistic_pred=logistic_classifier_model.predict(x_test)

# Ridge Classifier modelindeki test veri setini kullanarak tahminde bulunalım.
ridge_pred=ridge_classifier_model.predict(x_test)

# Decision Tree modelindeki test veri setini kullanarak tahminde bulunalım.
tree_pred=decision_tree_model.predict(x_test)

# Naive Bayes modelindeki test veri setini kullanarak tahminde bulunalım.
naive_bayes_pred=naive_bayes_model.predict(x_test)

# Neural Network modelindeki test veri setini kullanarak tahminde bulunalım.
neural_network_pred=neural_network_model.predict(x_test)

"""
Performansların karşılaştırılması
Precision, recall, f-1 puanı ve accuracy ayrı ayrı hesaplamak yerine, performansları karşılaştırmak için bir rapor oluşturabiliriz.

📌 classification_report() işlevi, kullanmanız gereken işlevdir.
📌 Tüm modellerin sonuçlarını yazdıralım.
"""

# Logistic Classifier modeli için bir Sınıflandırma Raporu oluşturalım.
logistic_report=classification_report(y_test,logistic_pred)

# Ridge Classifier modeli için bir Sınıflandırma Raporu oluşturalım.
ridge_report=classification_report(y_test,ridge_pred)

# Decision Tree modeli için bir Sınıflandırma Raporu oluşturalım.
tree_report=classification_report(y_test,tree_pred)

# Naive Bayes modeli için bir Sınıflandırma Raporu oluşturalım.
naive_bayes_report=classification_report(y_test,naive_bayes_pred)

# Neural Network modeli için bir Sınıflandırma Raporu oluşturalım.
neural_network_report=classification_report(y_test,neural_network_pred)

# Logistic Regression modelinin raporunu yazdıralım.
print("\n***Logistic Regression***")
print(logistic_report)

# Ridge Regression modelinin raporunu yazdıralım.
print("***Ridge Regression***")
print(ridge_report)

# Decision Tree modelinin raporunu yazdıralım.
print("***Decision Tree***")
print(tree_report)

# Naive Bayes modelinin raporunu yazdıralım.
print("***Naive Bayes***")
print(naive_bayes_report)

# Neural Network modelinin raporunu yazdıralım.
print("***Neural Network***")
print(neural_network_report)

"""
Değerlendirme
Decision tree(Karar ağacı) en iyi performansı gösterdi. Belki işleri bir adım öteye götürebilir ve 
daha iyi çalışıp çalışmadığını görmek için Random Forest algoritmasını deneyebiliriz.

📌 Aynı adımları izleyelim ve Random Forest için sınıflandırma raporunu yazdıralım
"""

# Random Forestn Sınıflandırıcı nesnesi oluşturalım, eğitelim ve tahminler yapalım.
random_forest_model=RandomForestClassifier()
random_forest_model.fit(x_train,y_train)
random_forest_pred=random_forest_model.predict(x_test)

# Random Forest modeli için bir sınıflandırma Raporu oluşturalım.
random_forest_report=classification_report(y_test,random_forest_pred)

# Sınıflandırma raporunu yazdıralım.
print("***Random Forest***")
print(random_forest_report)