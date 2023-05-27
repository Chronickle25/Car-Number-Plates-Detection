import cv2
import logging
import pytesseract

# Configuramos el logging
logging.basicConfig(filename='placas.log', level=logging.INFO, format='%(asctime)s %(message)s')

# Especifique la ubicación del ejecutable de Tesseract en su sistema
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Ruta del archivo xml que contiene el modelo pre-entrenado haarcascade para la detección de matrículas rusas
harcascade = "model/haarcascade_russian_plate_number.xml"

# Abrimos la cámara por defecto (la cámara con índice 0)
cap = cv2.VideoCapture(0)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("No se puede abrir la cámara. Comprueba tu conexión de cámara.")
    exit()

# Configuramos el ancho (640) y la altura (480) de la imagen capturada
cap.set(3, 640) # width
cap.set(4, 480) # height

# Cargamos el modelo haarcascade para la detección de matrículas
plate_cascade = cv2.CascadeClassifier(harcascade)

# Definimos el área mínima (en píxeles) para considerar que hemos detectado una matrícula
min_area = 500

# Contador para las matrículas detectadas
count = 0

# Variable para almacenar la última matrícula registrada
last_plate = None

# Función para verificar si los últimos dos caracteres son dígitos
def check_last_two_chars(plate):
    return plate[-2:].isdigit()

# Función para verificar si todos los caracteres están en mayúsculas
def check_all_chars_upper(plate):
    return plate.isupper()

# Bucle infinito para procesar la imagen de la cámara en tiempo real
while True:
    # Leemos el frame actual de la cámara
    success, img = cap.read()

    # Comprobamos si la cámara se leyó correctamente
    if not success:
        print("No se puede leer la cámara. Intentando de nuevo...")
        continue

    # Convertimos la imagen a escala de grises para la detección de matrículas
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detectamos las matrículas en la imagen
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    # Para cada matrícula detectada en la imagen
    for (x, y, w, h) in plates:
        # Calculamos el área de la matrícula
        area = w * h

        # Si el área es mayor a la mínima definida
        if area > min_area:
            # Dibujamos un rectángulo alrededor de la matrícula
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Colocamos un texto en la imagen indicando que es una matrícula
            cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            # Extraemos el área de la imagen correspondiente a la matrícula
            img_roi = img[y: y + h, x:x + w]

            # Convertimos la imagen a escala de grises para mejorar el reconocimiento de texto
            img_roi_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)

            # Usamos Tesseract para extraer el texto de la imagen de la matrícula
            plate_text = pytesseract.image_to_string(img_roi_gray, config='--psm 6')

            # Limpiamos el texto para eliminar posibles caracteres no deseados
            plate_text = "".join(e for e in plate_text if e.isalnum())

            # Verificamos si el texto de la matrícula tiene exactamente 6 caracteres, los dos últimos son dígitos,
            # todos los caracteres están en mayúsculas y no es la misma matrícula que la última registrada
            if len(plate_text) == 6 and plate_text != last_plate and check_last_two_chars(plate_text) and check_all_chars_upper(plate_text):
                # Registramos la detección de la placa y el texto extraído en el archivo .log
                logging.info(f'Placa detectada: {plate_text}')
                # Guardamos la matrícula actual como la última matrícula registrada
                last_plate = plate_text

            # Mostramos el área de la imagen correspondiente a la matrícula
            cv2.imshow("ROI", img_roi)

    # Mostramos el resultado en una ventana
    cv2.imshow("Result", img)

    # Si el usuario presiona la tecla 's'
    if cv2.waitKey(1) & 0xFF == ord('s'):
        # Guardamos la imagen de la matrícula
        cv2.imwrite("plates/scaned_img_" + str(count) + ".jpg", img_roi)
        # Dibujamos un rectángulo en la imagen y escribimos "Plate Saved"
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        # Mostramos el resultado en una ventana
        cv2.imshow("Results", img)
        # Hacemos una pausa de medio segundo
        cv2.waitKey(500)
        # Incrementamos el contador de matrículas
        count += 1

    # Si el usuario presiona la tecla 'q', se rompe el ciclo
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberamos los recursos de la cámara y cerramos todas las ventanas
cap.release()
cv2.destroyAllWindows()

