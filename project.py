import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree


# 1. Veriyi Oku (student-mat.csv dosyasının kodla aynı klasörde olması gerekir)
# Veriler noktalı virgül (;) ile ayrıldığı için bunu belirtiyoruz.
veri = pd.read_csv('student-mat.csv', sep=';')

# 2. Hedef Değişkeni Belirle (Öğrenci Dersi Geçti mi Kaldı mı?)
# G3 (Yıl sonu notu) 10 ve üzeriyse 1 (Geçti), altındaysa 0 (Kaldı) yapıyoruz.
veri['Basari_Durumu'] = (veri['G3'] >= 10).astype(int)

# G1, G2 (ara sınavlar) ve G3 sütunlarını modelden çıkarıyoruz ki sonuç önceden belli olmasın
X = veri.drop(['G1', 'G2', 'G3', 'Basari_Durumu'], axis=1)
y = veri['Basari_Durumu']

# 3. Metin Verilerini Sayılara Çevir (Karar ağaçları metin anlamaz)
le = LabelEncoder()
for sutun in X.select_dtypes(include=['object']).columns:
    X[sutun] = le.fit_transform(X[sutun])

# 4. Veriyi Böl (%80 Eğitim, %20 Test)
X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Karar Ağacını Dik ve Eğit
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_egitim, y_egitim)

# 6. Test Et ve Meyveleri Topla
tahminler = model.predict(X_test)
dogruluk = accuracy_score(y_test, tahminler)

print("--- KARAR AĞACI MODEL SONUÇLARI ---")
print(f"Modelin Doğruluk (Accuracy) Oranı: % {dogruluk * 100:.2f}")
print("\nDetaylı Karne (Sensitivity, F-measure vb. içerir):")
print(classification_report(y_test, tahminler))



# --- 1. GÖRSEL: KARMAŞIKLIK MATRİSİ (CONFUSION MATRIX) ---

# Matrisin matematiksel hesaplaması
cm = confusion_matrix(y_test, tahminler)

# Tablonun çizilmesi ve renklendirilmesi
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Kaldı (0)', 'Geçti (1)'], 
            yticklabels=['Kaldı (0)', 'Geçti (1)'])

plt.title('Karar Ağacı - Karmaşıklık Matrisi')
plt.xlabel('Modelin Tahmin Ettiği')
plt.ylabel('Gerçek Durum')

# Çizilen resmi ekranda göster
plt.show()





# --- 2. GÖRSEL: KARAR AĞACI ÇİZİMİ ---
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, class_names=['Kaldı', 'Geçti'], filled=True, rounded=True, fontsize=8)
plt.title('Öğrenci Başarısı Tahmini - Karar Ağacı Yapısı')

# Çizilen ağacı ekranda göster
plt.show()



# --- 3. GÖRSEL: EN ÖNEMLİ FAKTÖRLER GRAFİĞİ ---

# Modelden faktörlerin önem derecelerini alıyoruz
onem_dereceleri = model.feature_importances_

# Özellikleri ve önemlerini birleştirip büyükten küçüğe sıralıyoruz
onem_tablosu = pd.DataFrame({'Faktör': X.columns, 'Önem': onem_dereceleri})
onem_tablosu = onem_tablosu.sort_values(by='Önem', ascending=False).head(10) # Sadece en önemli ilk 10 faktör

# Yatay sütun grafiğini çizdiriyoruz
plt.figure(figsize=(10, 6))
sns.barplot(x='Önem', y='Faktör', data=onem_tablosu, palette='viridis')
plt.title('Öğrenci Başarısını Etkileyen En Önemli 10 Faktör')
plt.xlabel('Etki (Önem) Derecesi')
plt.ylabel('Sosyolojik ve Akademik Faktörler')
plt.tight_layout()

# Çizilen grafiği ekranda göster
plt.show()