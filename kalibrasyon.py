import cv2
import numpy as np

def nothing(x):
    pass

# Fotoğrafı yükle
img = cv2.imread('depremfoto.png')

if img is None:
    print("Hata: Fotoğraf bulunamadı!")
    exit()

img = cv2.resize(img, (1408, 768))

# Renkleri tamamen boşverip fotoğrafı gri tonlamaya çeviriyoruz
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.namedWindow("Enkaz Tespiti (Kenar Bulma)")

# Ayar çubukları
cv2.createTrackbar("Bulaniklik", "Enkaz Tespiti (Kenar Bulma)", 2, 10, nothing) # Pürüzleri yumuşatmak için
cv2.createTrackbar("Alt Esik", "Enkaz Tespiti (Kenar Bulma)", 50, 255, nothing)
cv2.createTrackbar("Ust Esik", "Enkaz Tespiti (Kenar Bulma)", 150, 255, nothing)
cv2.createTrackbar("Kalinlik", "Enkaz Tespiti (Kenar Bulma)", 1, 10, nothing) # Enkazları birleştirmek için

print("İşlem: Çubuklarla oynayarak yolları SİYAH (boş), enkazları BEYAZ yapmaya çalış.")
print("Bitirince 'q' tuşuna bas.")

while True:
    blur_val = cv2.getTrackbarPos("Bulaniklik", "Enkaz Tespiti (Kenar Bulma)")
    blur_val = blur_val * 2 + 1 # Değer her zaman tek sayı olmalı (3, 5, 7 vb.)
    
    lower = cv2.getTrackbarPos("Alt Esik", "Enkaz Tespiti (Kenar Bulma)")
    upper = cv2.getTrackbarPos("Ust Esik", "Enkaz Tespiti (Kenar Bulma)")
    kalinlik = cv2.getTrackbarPos("Kalinlik", "Enkaz Tespiti (Kenar Bulma)")

    # 1. Adım: Ufak tefek pürüzleri (toz, ufak taşlar) silmek için fotoğrafı bulanıklaştır
    blurred = cv2.GaussianBlur(gray, (blur_val, blur_val), 0)

    # 2. Adım: Canny algoritması ile kenarları (kırık dökük yerleri) bul
    edges = cv2.Canny(blurred, lower, upper)

    # 3. Adım: Bulunan kırık çizgileri kalınlaştırarak o bölgeyi tamamen "kapalı alan" yap
    if kalinlik > 0:
        kernel = np.ones((kalinlik, kalinlik), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

    cv2.imshow("Orjinal Fotograf", img)
    cv2.imshow("Enkaz Haritasi (Beyaz=Enkaz, Siyah=Yol)", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()