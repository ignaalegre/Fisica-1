import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Abrí el video
# cambiá por la ruta de tu video
video = cv2.VideoCapture('videos_recortados/sin_globo_r.mov')

# Parámetros dados
altura_caida = 4.23  # en metros
fps = 60  # frames por segundo
centros = []


# Leé el primer frame
ok, frame = video.read()
if not ok:
    print("No se pudo leer el video")
    exit()

# FLipeo el video
frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

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
desplazamientos_ternarios = []

while True:
    ok, frame = video.read()
    if not ok:
        break
    # FLipeo el video
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frame_count += 1

    # Actualizá el tracker
    ok, bbox = tracker.update(frame)
    if ok:
        (x, y, w, h) = [int(v) for v in bbox]
        center = (x + w // 2, y + h // 2)
        centros.append({'Frame': frame_count, 'X': center[0], 'Y': center[1]})
        if prev_center is not None:
            # Calculá el desplazamiento tomando como referencia el centro del bounding box
            dx = center[0] - prev_center[0]  # Diferencia en x
            dy = center[1] - prev_center[1]  # Diferencia en y
            # Usa pitágoras para calcular la distancia
            # entre el centro del bounding box anterior y el actual
            desplazamiento_pixeles = math.sqrt(dx**2 + dy**2)
            # Calculá la velocidad instantánea
            # Multiplicá por el fps para obtener la velocidad en px/s
            desplazamientos_pixeles.append(desplazamiento_pixeles)
            # Guardar los datos en la lista ternaria
            desplazamientos_ternarios.append((dx, dy, desplazamiento_pixeles))
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

# Calculo desplazamientos usando la lista ternaria
if desplazamientos_ternarios:
    # Calcular el desplazamiento total en el eje Y
    desplazamiento_total_y_px = sum(abs(dy)
                                    for _, dy, _ in desplazamientos_ternarios)
    factor_px_a_m = altura_caida / desplazamiento_total_y_px
    print(f"Desplazamiento total en píxeles (Y): {desplazamiento_total_y_px}")
    print(f"Factor de conversión (px a m): {factor_px_a_m}")
else:
    factor_px_a_m = None
    print("No hay desplazamientos para calcular factor de conversión.")

# --- Guardar posiciones en metros en CSV ---
if centros and factor_px_a_m is not None:
    df_centros = pd.DataFrame(centros)
    df_centros['X_metros'] = df_centros['X'] * factor_px_a_m
    # Ajustar Y para que comience en altura máxima
    df_centros['Y_metros'] = (altura_caida + 2) - \
        (df_centros['Y'] * factor_px_a_m)
    df_centros[['Frame', 'X_metros', 'Y_metros']].to_csv(
        'trayectoria_objeto_metros.csv', index=False)
    print("Archivo 'trayectoria_objeto_metros.csv' generado exitosamente.")
else:
    print("No se registraron centros o no se pudo calcular factor de conversión.")

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
if desplazamientos_ternarios:
    print(f"\n--- Cálculos para gráficos ---")

    # Inicializar listas para velocidades y aceleraciones en x e y
    velocidades_x_m_s = []
    velocidades_y_m_s = []
    aceleraciones_x_m_s2 = []
    aceleraciones_y_m_s2 = []

    # Calcular velocidades en x e y
    for dx, dy, desplazamiento_pixeles in desplazamientos_ternarios:
        v_x_m_s = dx * fps * factor_px_a_m
        v_y_m_s = dy * fps * factor_px_a_m
        velocidades_x_m_s.append(v_x_m_s)
        velocidades_y_m_s.append(v_y_m_s)

    # Calcular aceleraciones en x e y
    for i in range(1, len(velocidades_x_m_s)):
        a_x_m_s2 = (velocidades_x_m_s[i] -
                    velocidades_x_m_s[i - 1]) / (1 / fps)
        a_y_m_s2 = (velocidades_y_m_s[i] -
                    velocidades_y_m_s[i - 1]) / (1 / fps)
        aceleraciones_x_m_s2.append(a_x_m_s2)
        aceleraciones_y_m_s2.append(a_y_m_s2)

    # Crear tiempos para los gráficos
    tiempos = np.arange(first_tracked_frame + 1,
                        first_tracked_frame + 1 + len(velocidades_x_m_s)) / fps
    # Ajustar longitud para aceleraciones
    tiempos_aceleraciones = tiempos[:len(aceleraciones_x_m_s2)]

    # Graficar posición, velocidad y aceleración en x e y
    plt.figure(figsize=(12, 18))

    # Posición
    plt.subplot(3, 2, 1)
    plt.plot(tiempos[:len(centros)], [
             c['X'] * factor_px_a_m for c in centros[:len(tiempos)]], marker='o')
    plt.title('Posición en X vs Tiempo')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Posición X (m)')
    plt.grid()

    plt.subplot(3, 2, 2)
    plt.plot(tiempos[:len(centros)], [(altura_caida + 2) - (c['Y'] * factor_px_a_m)
             for c in centros[:len(tiempos)]], marker='o', color='r')
    plt.title('Posición en Y vs Tiempo')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Altura (m)')
    plt.grid()

    # Velocidad
    plt.subplot(3, 2, 3)
    plt.plot(tiempos, velocidades_x_m_s, marker='o')
    plt.title('Velocidad en X vs Tiempo')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Velocidad X (m/s)')
    plt.grid()

    plt.subplot(3, 2, 4)
    plt.plot(tiempos, velocidades_y_m_s, marker='o', color='r')
    plt.title('Velocidad en Y vs Tiempo')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Velocidad Y (m/s)')
    plt.grid()

    # Aceleración
    plt.subplot(3, 2, 5)
    plt.plot(tiempos_aceleraciones, aceleraciones_x_m_s2, marker='o')
    plt.title('Aceleración en X vs Tiempo')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Aceleración X (m/s²)')
    plt.grid()

    plt.subplot(3, 2, 6)
    plt.plot(tiempos_aceleraciones, aceleraciones_y_m_s2, marker='o', color='r')
    plt.title('Aceleración en Y vs Tiempo')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Aceleración Y (m/s²)')
    plt.grid()

    plt.tight_layout()
    plt.show()
else:
    print("No hay datos en la lista ternaria para calcular gráficos.")

video.release()
cv2.destroyAllWindows()
