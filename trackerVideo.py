import cv2
import math

# Abrí el video
video = cv2.VideoCapture('videos_recortados/globo_mediano_r.mov')  # cambiá por la ruta de tu video

# Parámetros dados
altura_caida = 4.23  # en metros
fps = 60 # frames por segundo
 
# Leé el primer frame
ok, frame = video.read() 
if not ok:
    print("No se pudo leer el video")
    exit()

# Seleccioná el ROI (Región de Interés)
bbox = cv2.selectROI("Tracking", frame, False)
tracker = cv2.TrackerCSRT.create()  # También podés usar KCF, MIL, etc.



# Inicializá el tracker con el primer frame y la ROI seleccionada
tracker.init(frame, bbox)
# Inicializá variables para el seguimiento y cálculo de velocidad
frame_count = 0
first_tracked_frame = None
last_tracked_bbox = None
prev_center = None
desplazamientos_pixeles = []


while True:
    ok, frame = video.read()
    if not ok:
        break
    frame_count += 1
    # Actualizá el tracker
    ok, bbox = tracker.update(frame)
    if ok:
        (x, y, w, h) = [int(v) for v in bbox]
        center = (x + w // 2, y + h // 2)
        if prev_center is not None:
            #Calculá el desplazamiento tomando como referencia el centro del bounding box
            dx = center[0] - prev_center[0] #Diferencia en x
            dy = center[1] - prev_center[1] #Diferencia en y
            #Usa pitágoras para calcular la distancia
            # entre el centro del bounding box anterior y el actual
            desplazamiento_pixeles = math.sqrt(dx**2 + dy**2)
            # Calculá la velocidad instantánea
            # Multiplicá por el fps para obtener la velocidad en px/s
            desplazamientos_pixeles.append(desplazamiento_pixeles)
        prev_center = center

    # Dibujá el bounding box
    if ok:
        if first_tracked_frame is None:
            first_tracked_frame = frame_count
        last_tracked_bbox = bbox
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tracking perdido", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    
 
    cv2.imshow("Tracking", frame)

    # Salí con la tecla ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

print(f"Primer frame con tracking exitoso: {first_tracked_frame}")
if last_tracked_bbox:
    print(f"Último bbox válido: {last_tracked_bbox}")
    print(f"Último frame con tracking exitoso: {frame_count}")
else:
    print("No se logró trackear ningún frame exitosamente.")


# --- Cálculo e impresión de la velocidad promedio de caída ---



if first_tracked_frame is not None:
    # Tiempo total de caída en segundos
    duracion_frames = frame_count - first_tracked_frame + 1
    tiempo_caida = duracion_frames / fps

    # Velocidad promedio: distancia / tiempo
    velocidad_promedio = altura_caida / tiempo_caida

    print(f"Altura de la caída: {altura_caida} m")
    print(f"Frames de caída: {duracion_frames}")
    print(f"Tiempo de caída: {tiempo_caida:.3f} s")
    print(f"Velocidad promedio de caída: {velocidad_promedio:.3f} m/s")
else:
    print("No se pudo calcular la velocidad porque no se detectó ningún tracking exitoso.")

# --- Conversión de velocidad instantánea a m/s ---
if desplazamientos_pixeles:
    desplazamiento_total_px = sum(desplazamientos_pixeles)
    factor_px_a_m = altura_caida / desplazamiento_total_px
    print(f"\n--- Velocidades instantáneas ---")
    for i, d_px in enumerate(desplazamientos_pixeles):
        velocidad_instantanea_pixeles = d_px * fps
        v_m_s = d_px * fps * factor_px_a_m
        print(f"Frame {i + first_tracked_frame + 1}: {v_m_s:.2f} m/s" " o " + f"{velocidad_instantanea_pixeles:.2f} px/s")

video.release()
cv2.destroyAllWindows()