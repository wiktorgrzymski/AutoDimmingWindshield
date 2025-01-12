import cv2
import numpy as np

# Ustaw wartość progową jasności
intensity_threshold = 245  # Możesz zmienić tę wartość lub pobrać ją od użytkownika

# Inicjalizacja kamer
cap1 = cv2.VideoCapture(0)  # Pierwsza kamera (do wykrywania jasnych punktów)
cap2 = cv2.VideoCapture(1)  # Druga kamera (do wykrywania twarzy)

# Załadowanie klasyfikatora twarzy (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Funkcja do przesuwania konturów o dane przesunięcie
def move_contours(contours, shift_x, shift_y):
    moved_contours = []
    for contour in contours:
        moved_contours.append(contour + np.array([shift_x, shift_y]))
    return moved_contours

while True:
    # Pobranie obrazu z pierwszej kamery
    ret1, frame1 = cap1.read()
    if not ret1:
        break

    # Pobranie obrazu z drugiej kamery
    ret2, frame2 = cap2.read()
    if not ret2:
        break

    # Konwersja obrazu z pierwszej kamery do skali szarości
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # Zastosowanie CLAHE do poprawy kontrastu
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray1 = clahe.apply(gray1)

    # Normalizacja obrazu, aby zwiększyć zakres jasności
    norm_gray1 = cv2.normalize(enhanced_gray1, None, 0, 255, cv2.NORM_MINMAX)

    # Wykrywanie jasnych punktów z określonym progiem jasności
    _, thresh = cv2.threshold(norm_gray1, intensity_threshold, 255, cv2.THRESH_BINARY)

    # Znalezienie konturów jasnych punktów
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Kopia obrazu z pierwszej kamery do zaciemnienia jasnych punktów
    overlay = frame1.copy()

    for contour in contours:
        if cv2.contourArea(contour) > 5:  # Pominięcie zbyt małych punktów
            # Obliczenie średniej jasności w obszarze konturu
            mask = np.zeros_like(norm_gray1)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_val = cv2.mean(norm_gray1, mask=mask)[0]

            # Ustalanie stopnia przezroczystości zależnie od jasności
            if mean_val >= 250:
                alpha = 1.0  # 100% zaciemnienie
            else:
                alpha = min(mean_val / 250, 1.0)

            # Tworzenie zaciemnionej wersji maski konturu
            temp_overlay = overlay.copy()
            cv2.drawContours(temp_overlay, [contour], -1, (0, 0, 0), -1)
            cv2.addWeighted(temp_overlay, alpha, overlay, 1 - alpha, 0, overlay)

            # Wypisanie jasności na oryginalnym obrazie
            brightness_text = f"{int(mean_val)}"
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(overlay, brightness_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Wykrywanie twarzy w obrazie z drugiej kamery
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray2, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Pobranie pozycji twarzy (pierwszej wykrytej twarzy)
        (x, y, w, h) = faces[0]
        face_center = (x + w // 2, y + h // 2)

        # Obliczamy środek obrazu (gdzie znajdują się jasne punkty)
        frame_center = (frame1.shape[1] // 2, frame1.shape[0] // 2)

        # Obliczanie przesunięcia na podstawie różnicy między środkami twarzy a obrazu
        shift_x = face_center[0] - frame_center[0] # TODO: twarz, nie oczy
        shift_y = face_center[1] - frame_center[1]

        # Przesuwamy wszystkie kontury o obliczone przesunięcie
        contours = move_contours(contours, shift_x, shift_y)

        # Dodanie wykrytych twarzy do obrazu
        for (fx, fy, fw, fh) in faces:
            cv2.rectangle(frame2, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)

    # Nakładanie nakładki na oryginalny obraz z zaciemnionymi punktami
    output_frame = cv2.addWeighted(overlay, 1, frame1, 0, 0)

    # Rysowanie przesuniętych konturów na obrazie z zaciemnieniem
    cv2.drawContours(output_frame, contours, -1, (0, 0, 0), -1)

    # Wyświetlenie obrazu z zaciemnionymi punktami i twarzami
    cv2.imshow("Zaciemnione punkty z efektem chmurek", output_frame)
    cv2.imshow("Wykrywanie twarzy", frame2)

    # Zakończenie programu po naciśnięciu klawisza 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Zwolnienie kamer i zamknięcie wszystkich okien
cap1.release()
cap2.release()
cv2.destroyAllWindows()