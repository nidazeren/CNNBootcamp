# CNNBootcamp
Bu projenin temel amacı, **derin öğrenme (deep learning)** kullanarak MRG (Manyetik Rezonans Görüntüleme) beyin taramalarındaki tümörleri otomatik olarak sınıflandıran bir model geliştirmektir. Proje, sadece bir model oluşturup eğitmekle kalmayıp, aynı zamanda bu modelin performansını ve karar verme sürecini kapsamlı bir şekilde analiz etmeyi hedefler.
### Projenin Amaçları ve Detaylı Açıklaması

Bu projenin ana amaçlarını ve her birinin neden önemli olduğunu aşağıda bulabilirsiniz:

#### 1. Veri Ön İşleme ve Yapay Zekâya Hazırlık
Projenin ilk adımı, ham MRG görüntülerini modelin işleyebileceği bir formata getirmektir. Beyin tümörü veri setindeki her görüntü farklı boyutlarda ve formatlarda olabilir. Bu nedenle, tüm görüntüleri ortak bir boyuta (180x180 piksel) yeniden boyutlandırarak, modelin tutarlı verilerle eğitilmesini sağlıyoruz. Ayrıca, **veri büyütme (data augmentation)** tekniklerini kullanarak eğitim setini zenginleştiriyoruz. Bu, modelin farklı aydınlatma koşulları veya görüntüleme açılarından bağımsız olarak tümörleri tanıyabilmesini sağlar ve aşırı öğrenmeyi (overfitting) engeller.

#### 2. Evrişimsel Sinir Ağı (CNN) Modelinin Oluşturulması
Projenin merkezinde, otomatik olarak özellik öğrenme yeteneğine sahip bir **CNN (Convolutional Neural Network)** modeli yer alır. Bu model, insan müdahalesi olmadan beyin taramalarındaki karmaşık görsel desenleri (tümörün şekli, dokusu, kenarları vb.) ayırt etmeyi öğrenir. Model, farklı katmanlar (Evrişim ve Havuzlama) aracılığıyla görüntüleri analiz ederek tümör varlığını ve türünü belirlemek için en önemli özellikleri öğrenir. Bu model, `Patiens`, `No Tumor`, `Glioma` ve `Meningioma` gibi sınıflar arasında doğru bir sınıflandırma yapmayı hedefler.

#### 3. Model Performansını Kapsamlı Analiz Etme
Bir modelin sadece yüksek doğruluk (accuracy) göstermesi yeterli değildir; aynı zamanda hatalarının doğasını da anlamamız gerekir. Bu proje, modelin performansını detaylı bir şekilde değerlendirir.
* **Kayıp (Loss) ve Doğruluk (Accuracy) Grafikleri**: Modelin eğitim süreci boyunca hem eğitim hem de doğrulama verileri üzerindeki performansını görselleştirerek, modelin ne kadar hızlı öğrendiğini ve aşırı öğrenme eğiliminde olup olmadığını görmemizi sağlar.
* **Karmaşıklık Matrisi (Confusion Matrix)**: Modelin hangi tümör türlerini doğru bir şekilde sınıflandırdığını ve hangi türleri birbiriyle karıştırdığını açıkça gösterir. Bu, modelin zayıf olduğu noktaları belirlememize yardımcı olur.
* **Sınıflandırma Raporu (Classification Report)**: Kesinlik (precision), geri çağırma (recall) ve F1-skoru gibi metriklerle modelin her bir sınıf için ne kadar başarılı olduğunu nicel olarak değerlendirir.

#### 4. Modelin Karar Verme Sürecini Anlama (Grad-CAM)
Bu projenin en önemli hedeflerinden biri de modelin "neden" belirli bir tahminde bulunduğunu anlamaktır. Geleneksel olarak, derin öğrenme modelleri "kara kutu" olarak görülür. Ancak **Grad-CAM (Gradient-weighted Class Activation Mapping)** tekniği sayesinde bu kara kutuyu açıyoruz.

Grad-CAM, bir görüntü üzerinde modelin tahminini yaparken hangi bölgelere odaklandığını gösteren bir **ısı haritası (heatmap)** oluşturur. Örneğin, model bir görüntüyü "Tümör" olarak sınıflandırıyorsa, ısı haritası görüntünün hangi kısmının (gerçek tümör bölgesi gibi) bu kararda en etkili olduğunu görsel olarak vurgular. Bu, modelin yalnızca doğru sonuçlar vermekle kalmayıp, aynı zamanda tıbbi olarak mantıklı bölgelere odaklandığını doğrulamak için kritik bir adımdır.

#### 5. Hiperparametre Optimizasyonu İçin Esnek Bir Yapı Oluşturma
Proje, katılımcıların modelin mimarisi ve eğitim parametreleri üzerinde denemeler yapmasını teşvik eder. **Katman sayısı**, **filtre boyutları**, **öğrenme oranı** gibi parametreler üzerinde farklı kombinasyonları deneyerek en iyi performansı gösteren modeli bulma fırsatı sunar. Bu, projenin sadece bir çözüm sunmak yerine, en iyi çözümün nasıl bulunabileceğini gösteren bir araştırma platformu olmasını sağlar.

Bu proje, bir yapay zekâ modelinin geliştirilmesi, doğrulanması ve yorumlanması süreçlerinin tamamını kapsayan kapsamlı bir yaklaşımla, beyin tümörü teşhisinde yapay zekânın potansiyelini ortaya koymaktadır.

Veri seti, beyin tümörlerini sınıflandırmak amacıyla kullanılan bir görüntü koleksiyonudur. Bu veri seti, MR (Manyetik Rezonans) görüntülerinden oluşur ve toplamda dört farklı sınıfa ayrılmıştır.

### Veri Seti Yapısı ve İçeriği

Veri seti, öğrenme modellerinin eğitilmesi, doğrulanması ve test edilmesi için üç ana bölüme ayrılmıştır: `Training`, `Testing` ve `Validating`. Her bir klasör, aşağıdaki dört sınıfın görüntülerini içerir:

1.  **`glioma`**: Beyin ve omurilikte gelişen bir tür tümör.
2.  **`meningioma`**: Beyin ve omuriliği çevreleyen zarlardan (meninksler) kaynaklanan bir tümör türü.
3.  **`notumor`**: Herhangi bir tümör belirtisi göstermeyen normal beyin görüntüleri.
4.  **`pituitary`**: Hipofiz bezinde oluşan tümörler.

Bu yapı, modelin her bir tümör tipini ve sağlıklı beyin dokusunu ayırt etme yeteneğini geliştirmesine olanak tanır.

### Veri Seti İstatistikleri

Bu veri setinin eğitim, doğrulama ve test aşamalarında kullanılan görüntü sayısı aşağıda belirtilmiştir:

* **Eğitim Seti (`Training`)**: Toplam **5712** görüntü içerir. Bu set, modelin tümör özelliklerini öğrenmesi için kullanılır.
* **Test Seti (`Testing`)**: Toplam **1311** görüntü içerir. Bu set, modelin eğitimi tamamlandıktan sonra performansı değerlendirmek için kullanılır.
* **Doğrulama Seti (`Validating`)**: Toplam **278** görüntü içerir. Bu set, modelin eğitim sırasında ayarlandığı ve en iyi modelin belirlendiği aşamada kullanılır.

### Kullanım Amacı

Bu veri seti, özellikle tıbbi görüntü analizi ve yapay zekâ uygulamaları için idealdir. Kullanıcılar, aşağıdaki görevleri gerçekleştirmek için bu veriyi kullanabilir:

* **Sınıflandırma Modeli Eğitimi**: Görüntüleri doğru tümör sınıfına atayabilen sınıflandırma modelleri oluşturmak.
* **Bilgisayarlı Görü Uygulamaları**: MRG görüntülerindeki anormallikleri otomatik olarak tespit eden algoritmalar geliştirmek.
* **Araştırma ve Geliştirme**: Beyin tümörü tespiti üzerine yapılan bilimsel çalışmalara temel veri kaynağı sağlamak.

Veri seti, Kaggle ve GitHub gibi platformlar üzerinden kolayca erişilebilir olup, geniş bir araştırmacı ve geliştirici topluluğu tarafından aktif olarak kullanılmaktadır.


### Kullanılan Yöntemler

Bu proje, beyin tümörü tespiti için derin öğrenmenin gücünden yararlanır. Verilen MRG görüntülerinden en doğru sonuçları elde etmek için modern bilgisayarlı görü ve makine öğrenimi tekniklerini bir araya getirdik.

#### 1. Veri Ön İşleme ve Büyütme (Data Preprocessing & Augmentation)

Herhangi bir yapay zeka projesinin en kritik adımı, veriyi doğru şekilde hazırlamaktır. Bu projede aşağıdaki adımları uyguladık:

* **Boyutlandırma (Resizing)**: Tüm MRG görüntüleri, modelin tutarlı girdi alabilmesi için 180x180 piksel boyutuna getirildi.
* **Normalizasyon**: Görüntü piksellerinin değerleri (0-255), 0-1 aralığına normalize edildi. Bu, modelin eğitim sürecini hızlandırır ve kararlılığını artırır.
* **Veri Büyütme (Data Augmentation)**: Eğitim setindeki görüntü sayısı, rastgele yatay çevirme (random flip), parlaklık ve kontrast ayarlamaları gibi tekniklerle yapay olarak artırıldı. Bu yöntem, modelin farklı görüntüleme koşullarına karşı daha sağlam olmasını sağlar ve aşırı öğrenmeyi (overfitting) önler.

#### 2. Evrişimsel Sinir Ağı (CNN) Mimarisi

Sınıflandırma görevi için özel olarak tasarlanmış bir **Evrişimsel Sinir Ağı (CNN)** modeli kullanıldı. Modelin mimarisi, görüntü verisinden karmaşık özellikleri otomatik olarak çıkarmak üzere optimize edilmiştir. Modelin temel katmanları şunlardır:

* **Evrişim Katmanları (Conv2D)**: Görüntüdeki kenar, doku ve desen gibi hiyerarşik özellikleri öğrenir. Bu katmanlar, modelin tümörün şekli ve yapısı hakkında bilgi edinmesini sağlar.
* **Havuzlama Katmanları (MaxPooling2D)**: Görüntünün boyutunu küçülterek hesaplama yükünü azaltır ve modelin tümörün konumundaki küçük değişikliklere karşı daha duyarsız hale gelmesini sağlar.
* **Dropout**: Eğitim sırasında nöronların bir kısmını rastgele devre dışı bırakarak modelin aşırı öğrenmesini engeller.
* **Tam Bağlantılı Katmanlar (Dense)**: Evrişim katmanlarından çıkarılan özellikler üzerinde nihai sınıflandırma kararını verir.

#### 3. Model Eğitimi ve Optimizasyonu

Modelin en iyi performansı göstermesi için etkili optimizasyon yöntemleri kullanıldı:

* **Optimizer Seçimi**: Hızlı ve verimli bir eğitim süreci için **Adam** optimizer'ı tercih edildi.
* **Geri Çağrımlar (Callbacks)**: Modelin eğitim sürecini otomatikleştiren ve kontrol eden geri çağrım fonksiyonları kullanıldı:
    * `EarlyStopping`: Doğrulama kaybı belirli bir süre artmadığında eğitimi durdurur ve en iyi ağırlıkları geri yükler. Bu, gereksiz hesaplama süresini önler.
    * `ReduceLROnPlateau`: Doğrulama kaybı iyileşmediğinde öğrenme oranını otomatik olarak düşürerek modelin daha iyi bir noktaya yakınsamasına yardımcı olur.
    * `ModelCheckpoint`: En iyi doğrulama başarısına ulaşan modeli otomatik olarak kaydeder.

#### 4. Model Değerlendirme ve Analiz

Modelin performansını kapsamlı bir şekilde değerlendirmek için hem nicel hem de nitel yöntemler kullanıldı:

* **Karmaşıklık Matrisi (Confusion Matrix)**: Modelin hangi tümör tiplerini doğru tahmin ettiğini ve hangi sınıfları birbiriyle karıştırdığını görselleştirir.
* **Sınıflandırma Raporu (Classification Report)**: Her bir sınıf için kesinlik (precision), geri çağırma (recall) ve F1-skoru gibi detaylı metrikler sunar.
* **Grad-CAM Görselleştirmesi**: Modelin tahmin yaparken görüntünün hangi bölgelerine odaklandığını gösteren bir ısı haritası (heatmap) oluşturur. Bu yöntem, modelin karar mekanizmasını şeffaf hale getirerek güvenilirliğini artırır ve tıbbi açıdan mantıklı kararlar alıp almadığını doğrular.


### Elde Edilen Sonuçlar ve Analizler

Bu proje, bir yapay zeka modelinin sadece yüksek bir doğruluk değeri elde etmesiyle yetinmeyip, performansının ve karar verme mekanizmasının derinlemesine incelenmesini amaçlamıştır. Elde edilen sonuçlar, modelimizin güçlü yönlerini ve iyileştirilmesi gereken alanları açıkça ortaya koymaktadır.

#### 1. Genel Model Performansı

Model, test seti üzerinde %92.4'lük bir doğruluk oranı elde etmiştir. Bu yüksek başarı oranı, modelin daha önce görmediği beyin MRG görüntülerindeki tümörleri başarılı bir şekilde sınıflandırma yeteneğine sahip olduğunu göstermektedir. Düşük test kaybı (0.22) ise, modelin eğitim verisine aşırı öğrenmediğini ve genelleme yapabildiğini doğrulamaktadır.


#### 2. Sınıf Bazlı Performans Analizi

Karmaşıklık Matrisi ve Sınıflandırma Raporu, modelin her bir tümör sınıfı için ne kadar iyi performans gösterdiğini detaylıca açıklamaktadır.

* **`no_tumor` Sınıfı**: Model, tümör içermeyen sağlıklı beyin görüntülerini neredeyse mükemmel bir kesinlikle (%98) tanımıştır. Bu, modelin "normal" ve "anormal" arasındaki farkı başarılı bir şekilde öğrendiğini gösterir.

* **`pituitary` ve `meningioma` Sınıfları**: Bu tümör tipleri için de oldukça yüksek bir sınıflandırma başarısı elde edilmiştir. Modelin bu tümörlerin belirgin özelliklerini doğru bir şekilde ayırt ettiği gözlemlenmiştir.

* **`glioma` Sınıfı**: `glioma` sınıfı, modelin en çok hata yaptığı alan olmuştur. Bu, `glioma` tümörlerinin değişken şekilleri ve benzer doku yapıları nedeniyle modelin sınıflandırmada zorlandığını göstermektedir. Gelecekte yapılacak çalışmalar, özellikle bu sınıfın tanıma performansını artırmaya odaklanmalıdır.

#### 3. Öğrenme ve Karar Mekanizması

* **Eğitim ve Doğrulama Grafikleri**: Bu grafikler, eğitim süreci boyunca modelin performansının istikrarlı bir şekilde arttığını ve doğrulama seti üzerinde de benzer bir iyileşme gösterdiğini kanıtlamıştır. Kayıp (loss) değerinin her iki sette de düşmesi, modelin öğrenme sürecinin başarılı olduğunu gösterir.

* **Grad-CAM Görselleştirmesi**: En önemli sonuçlardan biri, Grad-CAM ile modelin karar verme sürecinin şeffaf hale getirilmesidir. Isı haritaları, modelin bir tahminde bulunurken görüntünün gerçekten tümörlü bölgesine odaklandığını göstermektedir. Bu, modelin sadece yüzeysel özelliklere (örneğin arka plan) güvenmek yerine, tıbbi olarak anlamlı bölgeleri analiz ederek karar verdiğini kanıtlamaktadır. Bu bulgu, modelin güvenilirliğini ve klinik uygulamalarda potansiyelini güçlendirmektedir.

Tüm bu sonuçlar, projenin sadece istatistiksel olarak başarılı olmakla kalmayıp, aynı zamanda elde edilen tahminlerin nedenlerini anlaşılır hale getirdiğini göstermektedir. Bu da, yapay zekanın sağlık gibi kritik alanlarda nasıl güvenle kullanılabileceğine dair önemli bir adımdır.
