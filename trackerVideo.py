import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def inicializar_video(path):
    video = cv2.VideoCapture(path)

    # Lee 1er frame
    ok, frame = video.read()
    if not ok:
        print("No se pudo leer el video")
        exit()

    # Da vuelta el video
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    return video, frame


def seleccionar_roi(frame):
    # Seleccioná el ROI (Región de Interés)
    bbox = cv2.selectROI("Tracking", frame, False)
    tracker = cv2.TrackerCSRT.create()
    # Inicializá el tracker con el primer frame y la ROI seleccionada
    tracker.init(frame, bbox)
    return tracker, bbox


def procesar_video(video, tracker):
    centros = []
    desplazamientos_ternarios = []
    desplazamientos_pixeles = []

    frame_count = 0
    first_tracked_frame = None
    last_tracked_bbox = None
    prev_center = None

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
            centros.append(
                {'Frame': frame_count, 'X': center[0], 'Y': center[1]})

            if prev_center is not None:
                # Calculá el desplazamiento tomando como referencia el centro del bounding box
                dx = center[0] - prev_center[0]
                dy = center[1] - prev_center[1]
                desplazamiento = math.sqrt(dx**2 + dy**2)

                # Guardar desplazamientos
                desplazamientos_pixeles.append(desplazamiento)
                desplazamientos_ternarios.append((dx, dy, desplazamiento))

            prev_center = center

            # Dibujá el bounding box
            if first_tracked_frame is None:
                first_tracked_frame = frame_count
            last_tracked_bbox = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Tracking perdido", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.imshow("Tracking", frame)

        # Salí con la tecla ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    return centros, desplazamientos_ternarios, first_tracked_frame, frame_count, last_tracked_bbox


def calcular_factor_conversion(altura_caida, desplazamientos_ternarios):
    if not desplazamientos_ternarios:
        print("No hay desplazamientos para calcular factor de conversión.")
        return None

    desplazamiento_total_y_px = sum(abs(dy)
                                    for _, dy, _ in desplazamientos_ternarios)
    factor = altura_caida / desplazamiento_total_y_px
    print(f"Desplazamiento total en píxeles (Y): {desplazamiento_total_y_px}")
    print(f"Factor de conversión (px a m): {factor}")
    return factor


def guardar_csv(centros, factor_px_a_m, altura_caida):
    if not centros or factor_px_a_m is None:
        print("No se registraron centros o no se pudo calcular factor de conversión.")
        return

    df = pd.DataFrame(centros)
    df['X_metros'] = df['X'] * factor_px_a_m
    df['Y_metros'] = (altura_caida + 2) - (df['Y'] * factor_px_a_m)
    df[['Frame', 'X_metros', 'Y_metros']].to_csv(
        'trayectoria_objeto_metros.csv', index=False)
    print("Archivo 'trayectoria_objeto_metros.csv' generado exitosamente.")


def calcular_velocidad_promedio(altura_caida, first_frame, last_frame, fps):
    if first_frame is None:
        print(
            "No se pudo calcular la velocidad porque no se detectó ningún tracking exitoso.")
        return

    duracion_frames = last_frame - first_frame + 1
    tiempo = duracion_frames / fps
    velocidad = altura_caida / tiempo

    print(f"Altura de la caída: {altura_caida} m")
    print(f"Frames de caída: {duracion_frames}")
    print(f"Tiempo de caída: {tiempo:.3f} s")
    print(f"Velocidad promedio de caída: {velocidad:.3f} m/s")


def graficar_resultados(centros, desplazamientos_ternarios, factor_px_a_m, altura_caida, first_frame, fps):
    if not desplazamientos_ternarios:
        print("No hay datos en la lista ternaria para calcular gráficos.")
        return

    print("\n--- Cálculos para gráficos ---")

    velocidades_x = []
    velocidades_y = []
    aceleraciones_x = []
    aceleraciones_y = []

    for dx, dy, _ in desplazamientos_ternarios:
        velocidades_x.append(dx * fps * factor_px_a_m)
        velocidades_y.append(dy * fps * factor_px_a_m)

    for i in range(1, len(velocidades_x)):
        aceleraciones_x.append((velocidades_x[i] - velocidades_x[i-1]) * fps)
        aceleraciones_y.append((velocidades_y[i] - velocidades_y[i-1]) * fps)

    tiempos = np.arange(first_frame + 1, first_frame +
                        1 + len(velocidades_x)) / fps
    tiempos_aceleraciones = tiempos[:len(aceleraciones_x)]

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
    plt.plot(tiempos, velocidades_x, marker='o')
    plt.title('Velocidad en X vs Tiempo')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Velocidad X (m/s)')
    plt.grid()

    plt.subplot(3, 2, 4)
    plt.plot(tiempos, velocidades_y, marker='o', color='r')
    plt.title('Velocidad en Y vs Tiempo')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Velocidad Y (m/s)')
    plt.grid()

    # Aceleración
    plt.subplot(3, 2, 5)
    plt.plot(tiempos_aceleraciones, aceleraciones_x, marker='o')
    plt.title('Aceleración en X vs Tiempo')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Aceleración X (m/s²)')
    plt.grid()

    plt.subplot(3, 2, 6)
    plt.plot(tiempos_aceleraciones, aceleraciones_y, marker='o', color='r')
    plt.title('Aceleración en Y vs Tiempo')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Aceleración Y (m/s²)')
    plt.grid()

    plt.tight_layout()
    plt.show()


def main():
    # --- Parámetros dados ---
    VIDEO_PATH = 'oso_recortados/oso_sin_globo.mov'
    ALTURA_CAIDA = 4.28  # en metros
    FPS = 60

    # --- Ejecución del flujo principal ---
    video, first_frame = inicializar_video(VIDEO_PATH)
    tracker, bbox = seleccionar_roi(first_frame)
    centros, desplazamientos_ternarios, first_tracked_frame, last_frame, last_bbox = procesar_video(
        video, tracker)

    print(f"Primer frame con tracking exitoso: {first_tracked_frame}")
    if last_bbox:
        print(f"Último bbox válido: {last_bbox}")
        print(f"Último frame con tracking exitoso: {last_frame}")
    else:
        print("No se logró trackear ningún frame exitosamente.")

    factor_px_a_m = calcular_factor_conversion(
        ALTURA_CAIDA, desplazamientos_ternarios)
    guardar_csv(centros, factor_px_a_m, ALTURA_CAIDA)
    calcular_velocidad_promedio(
        ALTURA_CAIDA, first_tracked_frame, last_frame, FPS)
    graficar_resultados(centros, desplazamientos_ternarios,
                        factor_px_a_m, ALTURA_CAIDA, first_tracked_frame, FPS)

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
