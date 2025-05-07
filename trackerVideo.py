import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter


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
    first_center = None
    prev_velocity = None  # Para calcular la aceleración

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
            # Establecer el primer centro como origen (0, 0)
            if first_center is None:
                first_center = center

            # Ajustar las coordenadas del centro relativo al primer centro
            adjusted_center = (
                center[0] - first_center[0], center[1] - first_center[1])
            centros.append(
                {'Frame': frame_count,
                    'X': adjusted_center[0], 'Y': adjusted_center[1]}
            )

            if prev_center is not None:
                # Calculá el desplazamiento tomando como referencia el centro del bounding box
                dx = center[0] - prev_center[0]
                dy = center[1] - prev_center[1]
                desplazamiento = math.sqrt(dx**2 + dy**2)

                # Guardar desplazamientos
                desplazamientos_pixeles.append(desplazamiento)
                desplazamientos_ternarios.append((dx, dy, desplazamiento))

                # Calcular velocidad
                dt = 1 / 60  # FPS
                velocity = (dx / dt, dy / dt)

                # Dibujar vector de velocidad
                velocity_scale = 0.1  # Escalar para que sea visible
                velocity_end = (int(center[0] + velocity[0] * velocity_scale),
                                int(center[1] + velocity[1] * velocity_scale))
                cv2.arrowedLine(frame, center, velocity_end,
                                (255, 0, 0), 2, tipLength=0.3)

                # Calcular aceleración si hay una velocidad previa
                if prev_velocity is not None:
                    ax = (velocity[0] - prev_velocity[0]) / dt
                    ay = (velocity[1] - prev_velocity[1]) / dt

                    # Dibujar vector de aceleración
                    acceleration_scale = 0.01  # Escalar para que sea visible
                    acceleration_end = (int(center[0] + ax * acceleration_scale),
                                        int(center[1] + ay * acceleration_scale))
                    cv2.arrowedLine(frame, center, acceleration_end,
                                    (0, 255, 0), 2, tipLength=0.3)

                prev_velocity = velocity

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

    # Aplicar filtro Savitzky-Golay a los desplazamientos en píxeles
    if len(desplazamientos_pixeles) > 5:  # Asegurarse de que haya suficientes datos
        desplazamientos_pixeles = savgol_filter(
            desplazamientos_pixeles, window_length=5, polyorder=2).tolist()

    # Aplicar filtro Savitzky-Golay a las posiciones X e Y
    if len(centros) > 5:
        x_positions = [c['X'] for c in centros]
        y_positions = [c['Y'] for c in centros]
        x_positions_filtered = savgol_filter(
            x_positions, window_length=7, polyorder=2).tolist()
        y_positions_filtered = savgol_filter(
            y_positions, window_length=7, polyorder=2).tolist()

        # Actualizar las posiciones filtradas en la lista de centros
        for i, c in enumerate(centros):
            c['X'] = x_positions_filtered[i]
            c['Y'] = y_positions_filtered[i]

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
    df['X_metros'] = df['X_metros'].rolling(3, min_periods=1).mean()
    df['Y_metros'] = df['Y_metros'].rolling(3, min_periods=1).mean()
    df[['Frame', 'X_metros', 'Y_metros']].to_csv(
        'trayectoria_objeto_metros.csv', index=False)
    print("Archivo 'trayectoria_objeto_metros.csv' generado exitosamente.")


def calcular_velocidades_aceleraciones(csv_path, fps):
    # Leer el archivo CSV
    df = pd.read_csv(csv_path)

    # Calcular el intervalo de tiempo entre frames
    dt = 1 / fps

    # Calcular velocidades (diferencias entre posiciones consecutivas)
    df['Velocidad_X'] = df['X_metros'].diff() / dt
    df['Velocidad_Y'] = df['Y_metros'].diff() / dt
    df['Velocidad_X'] = df['Velocidad_X'].rolling(3, min_periods=1).mean()
    df['Velocidad_Y'] = df['Velocidad_Y'].rolling(3, min_periods=1).mean()

    # Calcular aceleraciones (diferencias entre velocidades consecutivas)
    df['Aceleracion_X'] = df['Velocidad_X'].diff() / dt
    df['Aceleracion_Y'] = df['Velocidad_Y'].diff() / dt
    df['Aceleracion_X'] = df['Aceleracion_X'].rolling(3, min_periods=1).mean()
    df['Aceleracion_Y'] = df['Aceleracion_Y'].rolling(3, min_periods=1).mean()

    # Llenar valores NaN generados por diff() con 0 (opcional)
    df.fillna(0, inplace=True)

    # Guardar el DataFrame actualizado en un nuevo archivo CSV
    output_csv_path = 'trayectoria_objeto_completa.csv'
    df.to_csv(output_csv_path, index=False)
    print(
        f"Archivo con velocidades y aceleraciones guardado en: {output_csv_path}")

    return df


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


def graficar_resultados(csv_path, altura_caida):
    # Leer el archivo CSV con los datos completos
    df = pd.read_csv(csv_path)

    # Crear el eje de tiempo basado en los frames
    tiempos = df['Frame'] / 60

    plt.figure(figsize=(12, 18))

    # Posición en X
    plt.subplot(3, 2, 1)
    plt.plot(tiempos, df['X_metros'], marker='o')
    plt.title('Posición en X vs Tiempo')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Posición X (m)')
    plt.grid()

    # Posición en Y
    plt.subplot(3, 2, 2)
    plt.plot(tiempos, df['Y_metros'], marker='o', color='r')
    plt.title('Posición en Y vs Tiempo')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Altura (m)')
    plt.grid()

    # Velocidad en X
    plt.subplot(3, 2, 3)
    plt.plot(tiempos, df['Velocidad_X'], marker='o')
    plt.title('Velocidad en X vs Tiempo')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Velocidad X (m/s)')
    plt.grid()

    # Velocidad en Y
    plt.subplot(3, 2, 4)
    plt.plot(tiempos, df['Velocidad_Y'], marker='o', color='r')
    plt.title('Velocidad en Y vs Tiempo')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Velocidad Y (m/s)')
    plt.grid()

    # Aceleración en X
    plt.subplot(3, 2, 5)
    plt.plot(tiempos, df['Aceleracion_X'], marker='o')
    plt.title('Aceleración en X vs Tiempo')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Aceleración X (m/s²)')
    plt.grid()

    # Aceleración en Y
    plt.subplot(3, 2, 6)
    plt.plot(tiempos, df['Aceleracion_Y'], marker='o', color='r')
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

    # Calcular velocidades y aceleraciones desde el CSV
    calcular_velocidades_aceleraciones('trayectoria_objeto_metros.csv', FPS)

    calcular_velocidad_promedio(
        ALTURA_CAIDA, first_tracked_frame, last_frame, FPS)

    graficar_resultados('trayectoria_objeto_completa.csv', ALTURA_CAIDA)

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
