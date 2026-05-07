# Qwen2B-VL OCR Görsel Analiz 

## 1) Bağlam ve Süreç

Bu uygulama, kullanıcının yüklediği görseli ve yazdığı prompt’u alır; **Qwen2-VL-2B** modeline gönderir ve üretilen metni ekranda gösterir.  
Arayüzden üç ayar yapılabilir:

- **Çıktı uzunluğu (token)**
- **Görsel çözünürlük önayarı**
- **CUDA karma hassasiyet (AMP) aç/kapa**

Uygulama **referans metin almaz**, doğruluk metriği hesaplamaz; çıktı yalnızca niteliksel (insan gözüyle) değerlendirilir.

---

## 2) Amaçlar

- **Çıktı Kalitesi (nitel):** Üretilen metnin görseldeki içeriği ne kadar yerinde, odaklı ve anlaşılır ifade ettiğini gözlemsel olarak değerlendirmek.
- **Kullanıcı Deneyimi:** Yükle → ayar → üret → görüntüle akışının akıcı olup olmadığını gözlemek.
- **Kararlılık:** Farklı görseller ve ayarlarda hatasız çalışıp çalışmadığını gözlemek.

> **Not:** Bu rapor sayısal doğruluk veya otomatik gecikme/bellek ölçümü içermez; uygulama kapsamı buna uygun değildir.

---

## 3) Kullanılan Parametreler ve Referans Noktaları

**Model:** `Qwen/Qwen2-VL-2B-Instruct`

- 4-bit quantization
- `device_map="auto"`
- FP16

**Arayüz Ayarları:**

- **Token (çıktı uzunluğu):** Kullanıcı slider ile seçer.
- **Çözünürlük önayarı:** Görsel, seçilen sınırlar içinde orantılı küçültülür.
- **AMP (CUDA):** Varsa mixed precision, yoksa etkisiz.

---

## 4) Süreç Özeti (Değerlendirme Prosedürü)

**Görsel çeşitliliği:**

- Metin içeren belgeler
- Pano/fotoğraf üzeri yazılar
- Tablo/şema görüntüleri
- Karma sahneler

**Prompt:**  
Kullanıcı, uygulamada istediği metni prompt olarak girebilir. Herhangi bir sınırlama yoktur.

**Koşum:**  
Uygulama her görsel için yalnızca çıktıyı üretir; odaklılık, kapsam, netlik veya tutarlılık gibi ek nitel değerlendirmeler yapılmaz.  
Referans metin karşılaştırması, doğruluk ölçümü veya zaman/bellek takibi bulunmamaktadır.  
Kullanıcı, çıktı üzerindeki yorumunu kendi gözlemlerine dayanarak yapar.

---

## 5) Sınırlamalar

- **Sayısal doğruluk yok:** Referans metin olmadığı için BLEU, ROUGE, CER, WER gibi ölçütler hesaplanmaz.
- **Performans ölçümü yok:** Gecikme, VRAM veya RAM otomatik kaydedilmez.
- **Prompt etkisi:** Çıktı kalitesi, prompt’un netliği ve göreve uygunluğuna duyarlıdır.

---

## 6) Kullanım Önerileri

- **Odaklı çıktı için:** Prompt’u net yazın.  
  _Örnek:_ “Sadece metinleri çıkar ve 3 madde ile özetle.”
- **Kapsamı artırmak için:** Token sayısını makul düzeyde artırın.  
  _(Aşırı artırmak tekrar ve konu dağılmasına yol açabilir.)_
- **Okunabilirlik sorunlarında:** Daha net, parlak veya kırpılmış görsel yüklemek çıktıyı iyileştirir.

---

## 7) Genel Değerlendirme

Uygulama, insan gözüyle nitel çıktı incelemesine dönük bir **OCR-benzeri görsel analiz** deneyimi sunar.  
Çıktılar; **odaklılık, kapsam, netlik ve tutarlılık** başlıklarında pratik şekilde değerlendirilebilir.  
Bu rapor, sayısal doğruluk ya da performans ölçümüne girmeden, gerçek kullanım senaryolarında çıktı kalitesini gözlemleyerek karar vermek isteyen kullanıcılar için hazırlanmıştır.
