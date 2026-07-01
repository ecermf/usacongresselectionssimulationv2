# Bulanık Mantık ve Sembolik Matematik Tabanlı Hibrit Meclis Simülatörü
# Mehmet Ferhat Ecer

Bu proje, ABD Kongresi'ndeki yasa oylama ihtimallerini tahmin etmek için geliştirilmiş akademik seviyede bir hibrit karar destek simülasyonudur. Projede lobi gücü ve partizan kutuplaşma gibi belirsiz değişkenler Mamdani Bulanık Mantık (Fuzzy Logic) çıkarımıyla modellenmiş; savaş, enflasyon ve skandal gibi dışsal makro faktörler ise SymPy kütüphanesi ile sembolik polinom denklemleri üzerinden sisteme entegre edilmiştir.

## Kullanılan Teknolojiler

- Python 3
- scikit-fuzzy (Bulanık Mantık Motoru)
- sympy (Sembolik Cebir ve Polinom Modelleme)
- numpy (Vektörel Hesaplamalar)
- plotly & matplotlib (Veri Görselleştirme ve Monte Carlo Çan Eğrisi)
- streamlit (Kullanıcı Arayüzü)

## Hibrit Modelleme Yaklaşımı

Sistem iki aşamalı bir hesaplama motorundan oluşur:

1. **Bulanık Mantık (Fuzzy Logic) Katmanı:**
   - **Girişler:** Lobi Gücü (0-100), Partizan Kutuplaşma (0-10)
   - **Çıkış:** Ham Destek Oranı (0-100)
   - *Yöntem:* `trimf` ve `trapmf` üyelik fonksiyonları, Mamdani çıkarımı ve Centroid durulaştırma kullanılmıştır.

2. **Sembolik Matematik (SymPy) Katmanı:**
   - Dışsal kriz faktörleri (Rally Effect, Medya Skandalı, Yüksek Enflasyon) basit if-else bloklarıyla değil; $P = F + 15W - 20Sc - 10Inf$ şeklinde cebirsel bir denklemle modellenmiştir. Denklem `sp.lambdify` ile yüksek performanslı nümerik bir fonksiyona çevrilip nihai olasılık hesaplanmaktadır.

## Kurulum

```bash
git clone [https://github.com/ecermf/usacongresselectionsimulation.git](https://github.com/ecermf/usacongresselectionsimulation.git)
cd usacongresselectionsimulation
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py




Sol menüdeki Senaryo Parametreleri üzerinden Lobi Gücü ve Kutuplaşma değerleri slider aracılığıyla değiştirilebilir. Dışsal kriz senaryoları checkbox'lar ile aktifleştirilebilir. Ana ekranda:

Anlık Fuzzy Ham Skoru ve Nihai SymPy Skoru

Mamdani Membership function (üyelik fonksiyonu) grafikleri

10.000 iterasyonluk Stokastik Monte Carlo dağılımı (Çan Eğrisi)

Çoklu Ajan Modeli ile 435 temsilcili meclis oturma planı ve parti sadakati kırılımları

görüntülenir. Model anlık (real-time) olarak çalışır ve tüm grafikler parametrelere göre dinamik olarak güncellenir.


Test Senaryoları (Duyarlılık Analizi)
Lobi Gücü	Kutuplaşma	     Aktif Makro Kriz	         =   Beklenen Karar Yönelimi
Yüksek (85)	Düşük (2)	 x   Yok	Güçlü Destek.        =   Yasa rahatlıkla geçer.
Orta (50)	Yüksek (9)	 x   Medya Skandalı (-20)	     =   Kritik zayıflık. Parti disiplini kopar, yasa reddedilir.
Zayıf (25)	Orta (5)	 x   Savaş Durumu (+15)	       =   Lobi zayıf olsa da Ulusal Birlik (Rally) etkisiyle yasa destek bulur.
Yüksek (75)	Yüksek (8.5) x	Yüksek Enflasyon (-10)   = 	Bıçak Sırtı (Volatil). Çoklu ajan modelinde fireler görülür.
