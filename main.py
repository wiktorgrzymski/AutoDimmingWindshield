import cv2
import numpy as np
import time

# Funkcja do znalezienia jasnych punktów w obrazie
def find_bright_spots(image, threshold=0.99):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Konwersja do skali szarości
    max_val = np.max(gray)

    # Prog jasności
    brightness_threshold = max_val * threshold

    # Maska dla jasnych punktów
    mask = gray >= brightness_threshold
    coordinates = np.column_stack(np.where(mask))  # Zwraca współrzędne (y, x)

    return coordinates

# Funkcja do przesuwania punktów
def move_points(points, shift_x, shift_y):
    return [(point[0] + shift_y, point[1] + shift_x) for point in points]


def main():

    # Ustawienia kamer
    face_camera = cv2.VideoCapture(0)  # Kamera patrząca na twarz
    light_camera = cv2.VideoCapture(2)  # Kamera patrząca na szybę

    if not face_camera.isOpened() or not light_camera.isOpened():
        print("Nie udało się otworzyć jednej z kamer.")
        exit()

    # Załadowanie klasyfikatora twarzy
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    fade_start_time = None  # Czas rozpoczęcia zanikania
    fade_duration = 1.0   # Czas trwania zanikania w sekundach

    while True:
        # Czytaj obrazy z obu kamer
        face_success, face_image = face_camera.read()
        light_success, light_image = light_camera.read()

        if not face_success or not light_success:
            print("Nie udało się odczytać obrazu z jednej z kamer.")
            break

        # Znajdź jasne punkty na obrazie światła
        light_bright_spots = find_bright_spots(light_image)

        # Wykrywanie twarzy w obrazie z kamery patrzącej na twarz
        gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Obliczanie przesunięcia na podstawie ruchu twarzy
        if len(faces) > 0:
            (x, y, w, h) = faces[0]  # Użycie pierwszej wykrytej twarzy
            face_center = (x + w // 2, y + h // 2)

            # Oblicz środek obrazu z kamery
            frame_center = (face_image.shape[1] // 2, face_image.shape[0] // 2)

            # Oblicz przesunięcie
            shift_x = face_center[0] - frame_center[0]
            shift_y = face_center[1] - frame_center[1]

            # Przesuń jasne punkty na obrazie światła
            moved_bright_spots = move_points(light_bright_spots, shift_x, shift_y)

            fade_start_time = time.time()  # Resetujemy czas zanikania
            avg_brightness = np.mean(cv2.cvtColor(light_image, cv2.COLOR_BGR2GRAY)) / 255.0 #srednia jasnosc pikseli na obrazie
            alpha = 0.9 - (0.15 * avg_brightness)  # Zaciemnienie od 90% do 75% w zależności od jasności otoczenia (jasniejsze otoczenie -> mniejsza nieprzezroczystoc, zeby nie bylo duzych kontrastow)

        # stopniowe zanikania zaciemnienia
        if fade_start_time is not None:
            elapsed_time = time.time() - fade_start_time
            alpha = max(0, alpha * (1 - elapsed_time / fade_duration))  # Nieprzezroczystość maleje do 0 w czasie fade_duration
                
            overlay = light_image.copy()

             # Rysowanie przesuniętych jasnych punktów i przyciemnianie ich na kopii obrazu światła
            for spot in moved_bright_spots:
                if 0 <= spot[1] < light_image.shape[1] and 0 <= spot[0] < light_image.shape[0]:
                    cv2.circle(overlay, (spot[1], spot[0]), 10, (0, 0, 0), -1)  # Czarny okrąg
                
            # Nakładanie zaciemnienia(overlay) na oryginalny obraz
            cv2.addWeighted(overlay, alpha, light_image, 1 - alpha, 0, light_image)

            # Dodanie prostokąta wokół twarzy
            cv2.rectangle(face_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Wyświetl obraz z kamery patrzącej na szybę
        cv2.imshow("Light Camera", light_image)
        # Wyświetl obraz z kamery patrzącej na twarz
        cv2.imshow("Face Camera", face_image)

        # Wyjście po wciśnięciu klawisza ESC
        if cv2.waitKey(5) & 0xFF == 27:
            break

    # Zwolnij zasoby
    face_camera.release()
    light_camera.release()
    cv2.destroyAllWindows()

    
if __name__ == "__main__":
    main()