from ultralytics import YOLO
import cv2
import os

def run_object_detection(source='0', conf_threshold=0.25):
    """
    Realiza detección de objetos usando YOLO.

    Args:
        source (str o int): Ruta a una imagen/video, URL de video o 0 para webcam.
        conf_threshold (float): Umbral de confianza para las detecciones.
    """
    # Cargar un modelo YOLOv8 pre-entrenado
    # Puedes usar 'yolov8n.pt' (nano), 'yolov8s.pt' (small), 'yolov8m.pt' (medium), etc.
    # Los modelos más grandes son más precisos pero más lentos.
    model = YOLO('yolov8n.pt')

    # Si la fuente es un número, asume que es una webcam
    if isinstance(source, int):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: No se pudo abrir la cámara {source}")
            return
        print(f"Detectando objetos desde la webcam {source}...")
    elif os.path.exists(source) or source.startswith(('http', 'https')):
        if os.path.isfile(source):
            print(f"Detectando objetos en el archivo: {source}...")
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print(f"Error: No se pudo abrir el archivo de video/imagen: {source}")
                return
        else:
            print(f"Procesando imagen: {source}")
            try:
                results = model(source, conf=conf_threshold)
                # Mostrar resultados en la imagen
                for r in results:
                    im_array = r.plot()  # plot a BGR numpy array of predictions
                    im = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)
                    cv2.imshow("YOLO Object Detection", im)
                    cv2.waitKey(0)
                cv2.destroyAllWindows()
                return
            except Exception as e:
                print(f"Error al procesar la imagen: {e}")
                return
    else:
        print(f"Fuente no reconocida: {source}. Por favor, provea una ruta de archivo, URL o '0' para webcam.")
        return

    # Bucle principal para procesamiento de video/webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Realizar detección de objetos en el frame
        results = model(frame, conf=conf_threshold)

        # Iterar sobre los resultados para dibujar bounding boxes y etiquetas
        # r.plot() genera una imagen con las detecciones ya dibujadas
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)
            cv2.imshow("YOLO Object Detection", im)
        cv2.waitKey(0)
        # Presionar 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Ejemplos de uso:

    # 1. Detección en una webcam (usa 0 para la cámara principal)
    # run_object_detection(source=0)

    # 2. Detección en un archivo de video (reemplaza 'video.mp4' con tu ruta)
    # Asegúrate de tener un archivo de video en la misma carpeta o especifica la ruta completa
    # run_object_detection(source='video.mp4', conf_threshold=0.4)

    # 3. Detección en una imagen (reemplaza 'image.jpg' con tu ruta)
    # Asegúrate de tener un archivo de imagen en la misma carpeta o especifica la ruta completa
    run_object_detection(source='images/casa-02.jpg', conf_threshold=0.3)
    run_object_detection(source='images/sala_estar.jpg', conf_threshold=0.2)
    run_object_detection(source='images/dormitorio2.jpg', conf_threshold=0.2)
    run_object_detection(source=0, conf_threshold=0.2)
    run_object_detection(source='https://youtube.com/shorts/TpVRYsSZuU8?si=WBKnMaBiOCpZnO6h', conf_threshold=0.5)
