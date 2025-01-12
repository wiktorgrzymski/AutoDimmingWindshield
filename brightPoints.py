import cv2
import numpy as np

# Ustaw wartość progową jasności
intensity_threshold = 240  # Możesz zmienić tę wartość lub pobrać ją od użytkownika

# Inicjalizacja kamery (0 oznacza pierwszą kamerę w systemie)
cap = cv2.VideoCapture(0)

while True:
    # Pobranie obrazu z kamery
    ret, frame = cap.read()
    if not ret:
        break

    # Konwersja obrazu do skali szarości
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Zastosowanie CLAHE do poprawy kontrastu
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # Normalizacja obrazu, aby zwiększyć zakres jasności
    norm_gray = cv2.normalize(enhanced_gray, None, 0, 255, cv2.NORM_MINMAX)

    # Wykrywanie jasnych punktów z określonym progiem jasności
    _, thresh = cv2.threshold(norm_gray, intensity_threshold, 255, cv2.THRESH_BINARY)

    # Znalezienie konturów jasnych punktów
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Kopia obrazu do zaciemnienia jasnych punktów
    overlay = frame.copy()

    for contour in contours:
        if cv2.contourArea(contour) > 5:  # Pominięcie zbyt małych punktów
            # Obliczenie średniej jasności w obszarze konturu
            mask = np.zeros_like(norm_gray)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_val = cv2.mean(norm_gray, mask=mask)[0]

            # Ustalanie stopnia przezroczystości zależnie od jasności
            if mean_val >= 250:
                alpha = 1.0  # 100% zaciemnienie
            else:
                # Skala zaciemnienia: Im większa jasność, tym większe zaciemnienie
                alpha = min(mean_val / 250, 1.0)

            # Tworzenie zaciemnionej wersji maski konturu
            temp_overlay = overlay.copy()
            cv2.drawContours(temp_overlay, [contour], -1, (0, 0, 0), -1)
            cv2.addWeighted(temp_overlay, alpha, overlay, 1 - alpha, 0, overlay)

            # Wypisanie jasności na oryginalnym obrazie
            brightness_text = f"{int(mean_val)}"
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(overlay, brightness_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Nakładanie nakładki na oryginalny obraz
    output_frame = cv2.addWeighted(overlay, 1, frame, 0, 0)

    # Wyświetlenie obrazu z zaciemnionymi jasnymi punktami i ich jasnością
    cv2.imshow("Jasne punkty z efektem chmurek", output_frame)

    # Zakończenie programu po naciśnięciu klawisza 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Zwolnienie kamery i zamknięcie wszystkich okien
cap.release()
cv2.destroyAllWindows()