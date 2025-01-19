# 🌌 Istanbul Deprem Senaryosu Analizi

Bu proje, İstanbul bölgesinde gece saatlerinde 7.5 Mw büyüklüğünde bir deprem senaryosu üzerine analiz yapılmasını amaçlayan bir veri analizi çalışmasını içermektedir.
Proje, bir deprem senaryosuna ilişkin verilere dayanarak can kaybı tahmini yapmak için farklı makine öğrenimi modellerinin kullanılmasını ve performanslarını karşılaştırmayı amaçlamaktadır.
Projede hem regresyon hem de çeşitli görselleştirme teknikleri kullanılarak verilerin analizi yapılmıştır.

---

## ✨ Proje Hedefleri

Bu projenin ana hedefleri:
Bu proje, İstanbul'da meydana gelebilecek 7.5 büyüklüğünde bir deprem senaryosunun analiz sonuçlarını kullanarak çeşitli makine öğrenimi algoritmalarını uygular.
Bu çalışmanın amacı, verilerden anlamlı sonuçlar çıkararak can kaybı tahmini yapmak ve farklı algoritmaların performanslarını değerlendirmektir.


1. **Deprem Etkileri Analizi:** Depremin gece saatlerinde gerçekleşmesi durumunda potansiyel etkilerin modellenmesi.
2. **Veri Analizi:** Toplanan verilere dayalı olarak öncelikli sorun alanlarının belirlenmesi.
3. **Makine Öğrenimi:** Deprem sonrası etkileri öngörmek için uygun algoritmaların uygulanması.
4. **Görsel Sunum:** Analiz sonuçlarının etkili bir şekilde görsel olarak sunulması.

---
## Veri Seti

Veri seti, İstanbul'daki muhtemel bir depremde oluşabilecek bina hasarlarını, yaralanmaları ve diğer ilgili metrikleri içermektedir. 
Veri seti deprem-senaryosu-analiz-sonuclar.csv dosyasından alınmıştır ve aşağıdaki gibi bir ön işleme tabi tutulmuştur:

1-Eksik ve sonsuz değerlerin kontrolü ve temizlenmesi.

2-Kategorik değişkenlerin one-hot encoding yöntemi ile sayısal hale getirilmesi.

3-Eğitim ve test veri setlerine ayrılması.

## 📂 Proje Yapısı

Bu proje, bir **Jupyter Notebook** olarak yapılandırılmıştır.    

### Proje Tanıtımı
Bu proje, bir deprem senaryosuna ilişkin verilere dayanarak can kaybı tahmini yapmak için farklı makine öğrenimi modellerinin kullanılmasını ve performanslarını karşılaştırmayı amaçlamaktadır. Projede hem regresyon hem de çeşitli görselleştirme teknikleri kullanılarak verilerin analizi yapılmıştır.

### Kullanılan Kütüphaneler

Projede aşağıdaki Python kütüphaneleri kullanılmıştır:

- `numpy`: Sayısal hesaplamalar için kullanılır.
- `pandas`: Veri işleme ve analiz için kullanılır.
- `matplotlib` ve `seaborn`: Veri görselleştirme için kullanılır.
- `scikit-learn`: Makine öğrenimi modelleri ve değerlendirme metrikleri için kullanılır.
- `xgboost`: Daha gelişmiş bir regresyon modeli sağlamak için kullanılmıştır.

## Kod Açıklamaları

### 1. Gerekli Kütüphanelerin Çağrılması
```python
import numpy as np
import pandas as pd
from IPython import get_ipython
from IPython.display import display
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
import graphviz
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
```
**Amaç:** Projede kullanılacak tüm kütüphaneler bu adımda projeye dahil edilmiştir.

---

### 2. Veri Setinin Yüklenmesi ve Kodlanması

#### 2.1 Kodlama ve İlk Kontroller
```python
# Türkçe karakter sorununu düzeltmek için chardet kullanılıyor
!pip install chardet
import chardet

with open("deprem-senaryosu-analiz-sonuclar.csv", "rb") as f:
    result = chardet.detect(f.read())
print(result)

# Veri setini yükleme
df = pd.read_csv("deprem-senaryosu-analiz-sonuclar.csv", encoding="MacRoman", sep=';')
df.head()
```
**Amaç:** Veri setindeki Türkçe karakter sorunlarını gidermek için `chardet` kütüphanesi kullanılmıştır. Veri seti `MacRoman` kodlamasıyla yüklenmiş ve ilk satırlar incelenmiştir.

---

#### 2.2 Eksik Verilerin İşlenmesi
```python
# Eksik değer kontrolü
print(df.isnull().any().any())

# Sonsuz değerlerin ve eksik verilerin temizlenmesi
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
```
**Amaç:** Eksik veya sonsuz değerler, analiz sürecini etkilememesi için veri setinden kaldırılmıştır.

---

### 3. Veri Analizi ve Görselleştirme

#### 3.1 Histogramlar
```python
df.hist(bins=15, figsize=(17, 17))
plt.suptitle("Histogram of Features")
plt.show()
```
**Amaç:** Verilerin dağılımını incelemek için histogram grafikleri oluşturulmuştur.
![histogram](https://github.com/user-attachments/assets/bb3a481f-eed4-4b0d-a3df-405f06cdc534)

---

#### 3.2 Korelasyon Matrisi
```python
numerical_features = df.select_dtypes(include=np.number).columns
corr_matrix = df[numerical_features].drop(columns=['Date'], errors='ignore').corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()
```
**Amaç:** Veriler arasındaki ilişkileri analiz etmek için korelasyon matrisi hesaplanıp görselleştirilmiştir.
![coorelationmatrix](https://github.com/user-attachments/assets/7fdfd6e2-6f15-45cf-9bc2-46b15609b29d)


---

### 4. Makine Öğrenimi Modelleri

#### 4.1 Random Forest Regressor
```python
# Model oluşturma
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Tahmin ve performans değerlendirme
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Random Forest Regressor Performansı:")
print(f"MSE: {mse:.4f}")
print(f"R2 Skor: {r2:.4f}")
```
**Amaç:** Random Forest algoritması ile can kaybı tahmini yapılmış ve model performansı değerlendirilmiştir.

---

#### 4.2 Linear Regression
```python
# Linear Regression modeli oluşturma
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, y_train)

# Test seti üzerinde tahmin ve değerlendirme
y_pred_lr = linear_reg_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred_lr)
r2 = r2_score(y_test, y_pred_lr)

print("Linear Regression Performansı:")
print(f"MSE: {mse:.4f}")
print(f"R² Skoru: {r2 :.4f}")
```
**Amaç:** Doğrusal regresyon modeli ile tahmin yapılmış ve performans metriği olarak MSE ve R² skorları hesaplanmıştır.

---

#### 4.3 XGBoost
```python
# XGBoost modeli oluşturma
params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.1,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}
num_boost_round = 200
xgb_model = train(params, dtrain, num_boost_round=num_boost_round)

# Tahmin ve değerlendirme
y_pred = xgb_model.predict(dtest)
mse = mean_squared_error(y_test, y_pred)
print("Test MSE:", mse)
```
**Amaç:** XGBoost algoritmasıyla tahmin yapılmış ve algoritmanın overfitting yapmasını önlemek amacıyla hiperparametro optimizasyonu yapılmıştır.Bu sayede daha hassas bir model sağlanmıştır.

---

### 5. Yeni Veri ile Tahmin
Her model için yeni bir veri seti kullanılarak tahmin işlemi yapılmıştır. Tahmini can kaybı sayısı çıktı olarak elde edilmiştir.

---

### Sonuç
Bu proje, deprem senaryosu verileri üzerinde farklı makine öğrenimi modellerinin performanslarını değerlendirmek ve can kaybı tahmini yapmak amacıyla gerçekleştirilmiştir.
Random Forest ve XGBoost algoritmaları, Lineer regresyona göre daha iyi performans göstermiştir.


---

## 🔧 Kullanılan Teknolojiler ve Araçlar

Bu projede kullanılan başlıca teknolojiler ve kütüphaneler:

- **Python 3.9+**
- **Jupyter Notebook**
- Veri Analizi:
  - **NumPy**
  - **Pandas**
- Veri Görselleştirme:
  - **Matplotlib**
  - **Seaborn**
- Makine Öğrenimi:
  - **scikit-learn**

---

## 📈 Nasıl Kullanılır?

Projeyi kendi ortamınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1. **Proje Dosyasını Klonlayın:**
   ```bash
   git clone https://github.com/zeystein/istanbul-deprem-senaryosu.git
   ```

2. **Gerekli Bağımlılıkları Yükleyin:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Jupyter Notebook'u Başlatın:**
   ```bash
   jupyter notebook
   ```

4. **Notebook Dosyasını Açın:**
   - `istanbul-deprem-senaryosu.ipynb` dosyasını açarak çalıştırabilirsiniz.

---

##  🏁🔎✨ Sonuç Analizi
##  1. Modellerin Performans Değerlendirmesi 📈✅🧮
Uygulanan modeller arasında en iyi performansı gösteren model XGBoost olmuştur. XGBoost, düşük MSE ve yüksek R² skoru ile diğer algoritmalardan üstün performans sergilemiştir.

Linear Regression modeli, basitliği nedeniyle hızlı bir çözüm sunmuş, ancak doğruluk oranı sınırlı kalmıştır.
Random Forest, doğruluk oranı yüksek bir model olarak etkili sonuçlar üretmiştir.
##  2. Model Sonuçlarının Yorumlanması 🧠💬🔍
Yüksek korelasyon gösteren değişkenler, model performansını olumlu etkilemiştir.
R² = 0.89, modelin yüksek bir açıklama gücüne sahip olduğunu ve bağımsız değişkenlerin deprem sonrası tahmini zararı oldukça iyi açıkladığını göstermektedir.
MSE, tahminlerin gerçek zarar miktarından ortalama 1.133 birim kare hata ile saptığını göstermektedir. Büyük sapmalara karşı daha hassas olan bu metrik, modelin doğruluk oranını ölçmek için önemli bir gösterge sunmaktadır.
MAE, tahmin edilen zarar miktarlarının gerçek değerlerden ortalama 0.83 birim sapma gösterdiğini ifade eder. Bu düşük hata oranı, modelin deprem sonrası zarar tahmini konusunda başarılı olduğunu göstermektedir.
##  3. Öne Çıkan Bulgular 🌟📊📚
Verilerin logaritmik dönüşümü, model performansını önemli ölçüde artırmıştır.
Eksik ve uç değerlerin doğru şekilde yönetilmesi, veri kalitesini yükseltmiş ve model doğruluğunu artırmıştır.
Görselleştirme teknikleri, analiz sonuçlarının daha iyi anlaşılmasını sağlamış ve karar verme süreçlerini desteklemiştir.
##  4. Geliştirme Önerileri 🚀💡🔮
Daha geniş kapsamlı bir veri seti ile modellerin tekrar eğitilmesi, daha doğru sonuçlar sağlayabilir.
Model optimizasyonu için GridSearchCV veya Bayesian Optimization yöntemleri kullanılabilir.
Projenin bir web uygulamasına dönüştürülmesi, daha geniş bir kullanıcı kitlesine ulaşmayı sağlayabilir.
Bu proje, İstanbul’da gece gerçekleşecek bir deprem senaryosu için etkili bir analiz sunmuş ve makine öğrenmesi tekniklerinin afet yönetimi ve risk analizi gibi alanlarda nasıl kullanılabileceğini göstermiştir. Gelecekte yapılacak geliştirmelerle bu tür projeler, karar vericiler ve afet yönetim ekipleri için daha geniş kapsamlı ve faydalı hale getirilebilir. 🌍📈🏙️



---

## 🔗 Katkıda Bulunun

Her türlü katkıya açığız! Sorularınızı, önerilerinizi ve geri bildirimlerinizi GitHub üzerinden issue olarak paylaşabilirsiniz.

---

## 👤 Yazarlar

- **Zeynep Sude Güneş**  

---

## 🔗 Youtube linki

   https://youtu.be/kvJhx0krico

