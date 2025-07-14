import cv2
import math
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit


def main(csv_path, video_path):
    # --- Parámetros dados ---
    video_path
    ALTURA_CAIDA = 4.28  # en metros
    FPS = 60
    MASA_OBJETO = 0.1
    print(f"Procesando video: {video_path}")
    # --- Ejecución del flujo principal ---
    video, first_frame = inicializar_video(video_path)
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

    # Calcular velocidades y aceleraciones desde el CSV
    df = calcular_velocidades_aceleraciones(
        'tracker/csv_generados/trayectoria_objeto_metros.csv', FPS)

    # Calcular constante viscosa
    k_estimado = estimar_constante_viscosa_terminal(df, MASA_OBJETO)
    print(f"K estimado: {k_estimado}")
    tiempos = df['Frame'] / FPS
    calcular_modelo_viscoso(df, tiempos.values, MASA_OBJETO, k_estimado, ALTURA_CAIDA)
    ajuste_pos_Y, coef_pos_Y = ajuste_posicion_viscoso(tiempos, df['Y_metros'], MASA_OBJETO, ALTURA_CAIDA)
    ajuste_vel_Y, coef_vel_Y = ajuste_velocidad_viscoso(tiempos, df['Velocidad_Y'], MASA_OBJETO)
    ajuste_ace_Y, coef_ace_Y = ajuste_constante(tiempos, df['Aceleracion_Y'])

    print("\n--- Ajustes ---")
    print(f"Posición Y: coeficiente={coef_pos_Y[0]:.3f}")
    print(f"Velocidad Y: coeficiente={coef_vel_Y[0]:.3f}")
    print(f"Aceleración Y constante: {coef_ace_Y[0]:.3f}")

    df['Posicion_Ajuste_Viscoso_Y'] = ajuste_pos_Y
    df['Velocidad_Ajuste_Viscoso_Y'] = ajuste_vel_Y
    df['Aceleracion_Ajuste_Viscoso_Y'] = ajuste_ace_Y

    # Calcular fuerza de rozamiento
    calcular_fuerza_rozamiento(df, MASA_OBJETO)
    calcular_fuerza_rozamiento_teorico(df, MASA_OBJETO)
    
    
    # Calcular impulso experimental y teórico
    calcular_impulso_experimental(df, FPS)
    calcular_impulso_teorico(df, FPS)
    
    #Calcular trabajo experimental y teórico
    calcular_trabajo_experimental(df, FPS)
    calcular_trabajo_teorico(df, FPS)
    
    # Calcular energía potencial, cinética y mecánica
    E_pot_real, E_cin_real, E_mec_real = calculos_Energia(
        df['Y_metros'], MASA_OBJETO, df['Velocidad_Y'])

    df['Energia_Potencial'] = E_pot_real
    df['Energia_Cinetica'] = E_cin_real
    df['Energia_Mecanica'] = E_mec_real

    E_pot_teo, E_cin_teo, E_mec_teo = calculos_Energia(
        df['Posicion_Y_Teorico'], MASA_OBJETO, df['Velocidad_Y_Teorico'])

    df['Energia_Potencial_Teorico'] = E_pot_teo
    df['Energia_Cinetica_Teorico'] = E_cin_teo
    df['Energia_Mecanica_Teorico'] = E_mec_teo

    E_pot_ajuste, E_cin_ajuste, E_mec_ajuste = calculos_Energia(
        df['Posicion_Ajuste_Viscoso_Y'], MASA_OBJETO, df['Velocidad_Ajuste_Viscoso_Y'])

    df['Energia_Potencial_Ajuste'] = E_pot_ajuste
    df['Energia_Cinetica_Ajuste'] = E_cin_ajuste
    df['Energia_Mecanica_Ajuste'] = E_mec_ajuste

    df.to_csv(csv_path, index=False)
    print(f"Archivo CSV completo generado en: {csv_path}")
    video.release()
    cv2.destroyAllWindows()


def inicializar_video(path):
    video = cv2.VideoCapture(path)

    # Lee 1er frame
    ok, frame = video.read()
    if not ok:
        print("No se pudo leer el video")
        exit()


    return video, frame


def seleccionar_roi(frame):
    ESCALA_VENTANA = 0.5
    alto, ancho = frame.shape[:2]
    frame_resized = cv2.resize(frame, (int(ancho * ESCALA_VENTANA), int(alto * ESCALA_VENTANA)))
    cv2.namedWindow("Seleccionar ROI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Seleccionar ROI", 600, 1000)

    # Seleccionar ROI sobre el frame reducido
    bbox_resized = cv2.selectROI("Seleccionar ROI", frame_resized, False)
    cv2.destroyWindow("Seleccionar ROI")

    # Escalás de nuevo las coordenadas a las del frame original
    bbox = (
        int(bbox_resized[0] / ESCALA_VENTANA),
        int(bbox_resized[1] / ESCALA_VENTANA),
        int(bbox_resized[2] / ESCALA_VENTANA),
        int(bbox_resized[3] / ESCALA_VENTANA),
    )

    tracker = cv2.TrackerCSRT.create()
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
    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracking", 600, 1000)

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

    return centros, desplazamientos_ternarios, first_tracked_frame, frame_count, last_tracked_bbox


def calcular_factor_conversion(altura_caida, desplazamientos_ternarios):
    """
    Calcula el factor de conversión de píxeles a metros basado en los desplazamientos
    y la altura de caída del objeto.
    """
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
    df['Y_metros'] = (altura_caida) - (df['Y'] * factor_px_a_m)
    df['X_metros'] = df['X_metros'].rolling(3, min_periods=1).mean()
    df['Y_metros'] = df['Y_metros'].rolling(3, min_periods=1).mean()
    df[['Frame', 'X_metros', 'Y_metros']].to_csv(
        'tracker/csv_generados/trayectoria_objeto_metros.csv', index=False)
    print("Archivo 'trayectoria_objeto_metros.csv' generado exitosamente.")

# === FUNCIONES DE AJUSTE ===


def ajuste_parabolico(t, y):
    """Ajusta una curva parabólica a los datos (t, y) usando la forma y = a*t^2 + b*t + c."""
    def modelo(t, a, b, c): return a * t**2 + b * t + c
    params, _ = curve_fit(modelo, t, y)
    return modelo(t, *params), params


def ajuste_lineal(t, y):
    """Ajusta una línea recta a los datos (t, y) usando la forma y = m*t + b."""
    def modelo(t, m, b): return m * t + b
    params, _ = curve_fit(modelo, t, y)
    return modelo(t, *params), params


def ajuste_constante(t, df):
    """Ajusta una constante a los datos (t, df) usando la forma df = c."""
    #Se recortan los primeros 2 frames y los últimos 5 para evitar ruido al inicio y final
    #Como la funcion sera constante, se rellenan con el valor de cualquier otro frame
    t_recortado = t[2:-5]
    df_recortado = df.iloc[2:-5]
    def modelo(t, c): return np.full_like(t, c)
    params, _ = curve_fit(modelo, t_recortado, df_recortado)
    ajuste = modelo(t_recortado, *params)
    
    # Completar con valor constante al inicio y final para que tenga la longitud original
    ajuste_completo = np.full(len(df), ajuste[0])
    ajuste_completo[2:-5] = ajuste
    
    return ajuste_completo, params


def ajuste_posicion_viscoso(t, y, masa, altura_inicial):
    """Ajusta una curva exponencial a los datos de posición (t, y) usando la forma
    y = y0 + v_terminal * t - (m * v_terminal / k) * (1 - exp(-k * t / m))"""
    def modelo_posicion(t, k):
        g = 9.81
        v_terminal = masa * -g / k
        return altura_inicial + v_terminal * t - (masa * v_terminal / k) * (1 - np.exp(-k * t / masa))
    params, _ = curve_fit(modelo_posicion, t, y)
    return modelo_posicion(t, *params), params


def ajuste_velocidad_viscoso(t, y, masa):
    def modelo_velocidad(t, k):
        g = 9.81
        v_terminal = masa * -g / k
        return v_terminal * (1 - np.exp(-k * t / masa))
    params, _ = curve_fit(modelo_velocidad, t, y)
    return modelo_velocidad(t, *params), params


def ajuste_posicion_viscoso(t, y, masa, altura_inicial):
    def modelo_posicion(t, k):
        g = 9.81
        v_terminal = masa * -g / k
        return altura_inicial + v_terminal * t - (masa * v_terminal / k) * (1 - np.exp(-k * t / masa))
    params, _ = curve_fit(modelo_posicion, t, y)
    return modelo_posicion(t, *params), params


def ajuste_velocidad_viscoso(t, y, masa):
    def modelo_velocidad(t, k):
        g = 9.81
        v_terminal = masa * -g / k
        return v_terminal * (1 - np.exp(-k * t / masa))
    params, _ = curve_fit(modelo_velocidad, t, y)
    return modelo_velocidad(t, *params), params


# CINEMATICA

def calcular_velocidades_aceleraciones(csv_path, fps):
    # Leer el archivo CSV
    df = pd.read_csv(csv_path)

    # Calcular el intervalo de tiempo entre frames
    dt = 1 / fps

    # suavizar posición con Savitzky-Golay antes de derivar
    df['X_metros'] = savgol_filter(
        df['X_metros'], window_length=7, polyorder=2)
    df['Y_metros'] = savgol_filter(
        df['Y_metros'], window_length=7, polyorder=2)

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
    output_csv_path = csv_path
    df.to_csv(output_csv_path, index=False)
    print(
        f"Archivo con velocidades y aceleraciones guardado en: {output_csv_path}")

    return df


def velocidad_promedio_y(df):
    return df['Velocidad_Y'].mean()


def aceleración_promedio_y(df):
    return df['Aceleracion_Y'].mean()


# DINAMICA

def estimar_constante_viscosa_terminal(df, masa_objeto):
    """
    Estima la constante viscosa k usando aceleración y velocidad terminales.

    Parámetros:
        - df: DataFrame que contiene las columnas 'Aceleracion_Y' y 'Velocidad_Y'.
        - masa_objeto: Masa del objeto en kg.

    Retorna:
        - k: Constante viscosa estimada en kg/s.
    """
    g = 9.81
    # Promediar últimos valores para suavizar ruido
    a_terminal = df['Aceleracion_Y'].rolling(3).mean().iloc[-6]
    v_terminal = df['Velocidad_Y'].abs().max()

    # Evitar división por cero o valores inválidos
    if abs(v_terminal) < 1e-6:
        print("Velocidad terminal demasiado pequeña para estimar k.")
        return None

    # Cálculo de k basado en equilibrio de fuerzas en estado terminal
    k = masa_objeto * (g - abs(a_terminal)) / abs(v_terminal)
    print(f"Estimación k usando valores terminales: k = {k:.5f} kg/s")
    return k

def estimar_constante_viscosa_con_ajuste_lineal(df, masa_objeto):
    # Realizar el ajuste lineal a_Y = a0 + b * v_Y
    vel_y = df['Velocidad_Y'].iloc[2:-5].values
    acel_y = df['Aceleracion_Y'].iloc[2:-5].values
    _, (m, b) = ajuste_lineal(vel_y, acel_y)
    print(f"m = {m} y b= {b}")
    k = -masa_objeto * m  # porque m ≈ -k/m
    return k


def estimar_constante_viscosa_con_ajuste_cuadrático(df, masa_objeto):
    # Realizar el ajuste lineal a_Y = a0 + b * v_Y
    vel_y = df['Velocidad_Y'].values
    acel_y = df['Aceleracion_Y'].values
    ajustado, (a, b, c) = ajuste_parabolico(vel_y, acel_y)
    k = -masa_objeto * b  # porque b ≈ -k/m
    # Calcular R²
    ss_res = np.sum((acel_y - ajustado) ** 2)
    ss_tot = np.sum((acel_y - np.mean(acel_y)) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"Modelo ajustado: a_Y = {c:.3f} + ({b:.3f})*v_Y + ({a:.3f})*v_Y²")
    print(
        f"Constante viscosa estimada con ajuste parabólico: k = {k:.4f} kg/s")
    print(f"Coeficiente de correlación R² = {r2:.4f}")

    return k


def calcular_fuerza_rozamiento(df, masa_objeto):
    """Calcula la fuerza de rozamiento experimental utilizando la constante viscosa estimada.
    La fuerza de rozamiento es F = -k * v, donde k es la constante de rozamiento
    y v es la velocidad del objeto."""
    k1 = estimar_constante_viscosa_con_ajuste_lineal(df, masa_objeto)
    k2 = estimar_constante_viscosa_con_ajuste_cuadrático(df, masa_objeto)
    k3 = estimar_constante_viscosa_terminal(df, masa_objeto)
    
    print(f"Constante de rozamiento estimada con el ajuste lineal: k1 = {k1} kg/s")
    print(f"Constante de rozamiento estimada con el ajuste cuadrático: k2 = {k2} kg/s")
    print(f"Constante de rozamiento estimada con el método de velocidad terminal: k3 = {k3} kg/s")

    v = df['Velocidad_Y']
    print("Calculando fuerza de rozamiento experimental utilizando la constante estimada con el metodo de la velocidad terminal")
    df['Fuerza_Rozamiento_Y'] = -k3 * v


def calcular_fuerza_rozamiento_teorico(df, masa_objeto):
    """Calcula la fuerza de rozamiento teórica usando la constante viscosa estimada.
    La fuerza de rozamiento es F = -k * v, donde k es la constante de rozamiento
    y v es la velocidad del objeto.
    Parámetros:
        - df: DataFrame con las velocidades teóricas
        - masa_objeto: Masa del objeto (kg)"""
        
    k1 = estimar_constante_viscosa_con_ajuste_lineal(df, masa_objeto)
    k2 = estimar_constante_viscosa_con_ajuste_cuadrático(df, masa_objeto)
    k3 = estimar_constante_viscosa_terminal(df, masa_objeto)

    v = df['Velocidad_Y_Teorico']
    print("Calculando fuerza de rozamiento teorica utilizando la constante estimada con el metodo de la velocidad terminal")
    df['Fuerza_Rozamiento_Y_Teorico'] = -k3 * v
    


def calcular_modelo_viscoso(df, tiempos, masa, k, altura_inicial):
    """
    Derivación:
    Por 2da ley de Newton:
        ∑F = m·a → -mg + (-kv) = m·dv/dt
        dv/dt + (k/m)v = -g   → EDO lineal

    Solución por factor integrante:
        v(t) = -(m*g/k)(1 - e^(-k·t/m))

    Luego, integrando para y(t):
        y(t) = y0 - (mg/k)t + (m²g/k²)(1 - e^(-k·t/m))

    Parámetros:
        - tiempos: np.array con los valores de tiempo
        - masa: masa del objeto (kg)
        - k: constante de rozamiento viscoso (kg/s)
        - altura_inicial: y0 (m)

    """
    g = 9.81
    v_terminal = masa * -g / k
    vel = v_terminal * (1 - np.exp(-k * tiempos / masa))
    pos = altura_inicial - (masa * g / k) * tiempos + (masa**2 * g / k**2) * (1 - np.exp(-k * tiempos / masa))

    df['Velocidad_Y_Teorico'] = vel
    df['Posicion_Y_Teorico'] = pos


# ENERGÍA


def calcular_impulso_experimental(df, fps):
    """ Calcula el impulso experimental a partir de la fuerza de rozamiento medida.
    El impulso es la integral de la fuerza respecto al tiempo.
    Parámetros:
        - df: DataFrame con la columna 'Fuerza_Rozamiento_Y'
        - fps: Frames por segundo del video"""
    fuerza = df['Fuerza_Rozamiento_Y'].values
    dt = 1.0 / fps
    impulso = np.cumsum(fuerza) * dt
    df['Impulso'] = impulso
    print(f"Impulso experimental total = {impulso[-1]:.4f} N·s")


def calcular_impulso_teorico(df, fps):
    """ Calcula el impulso teórico a partir de la fuerza de rozamiento teórica.
    Parámetros:
        - df: DataFrame con la columna 'Fuerza_Rozamiento_Y_Teorico'
        - fps: Frames por segundo del video"""
    fuerza = df['Fuerza_Rozamiento_Y_Teorico'].values
    dt = 1.0 / fps
    impulso = np.cumsum(fuerza) * dt
    df['Impulso_Teorico'] = impulso
    print(f"Impulso teórico total = {impulso[-1]:.4f} N·s")
    
def calcular_trabajo_experimental(df,fps):
    """Calcula el trabajo realizado por la fuerza de rozamiento experimental.
    El trabajo es la integral de la fuerza respecto al desplazamiento.
    En este caso, se asume que el desplazamiento es la velocidad medida por el intervalo de tiempo. Se observo que los resultados
    daban mejores al usar la velocidad medida en lugar de la posición.
    Parámetros:
        - df: DataFrame con la columna 'Fuerza_Rozamiento_Y'
        - fps: Frames por segundo del video"""
    fuerza = df['Fuerza_Rozamiento_Y'].values
    dt = 1.0 / fps
    desplazamientos = df['Velocidad_Y'].values * dt
    trabajo = np.cumsum(fuerza * desplazamientos)
    df['Trabajo_Experimental'] = trabajo
    


def calcular_trabajo_teorico(df,fps):
    """Calcula el trabajo realizado por la fuerza de rozamiento teórica.
    Parámetros:
        - df: DataFrame con la columna 'Fuerza_Rozamiento_Y_Teorico'
        - fps: Frames por segundo del video"""
    fuerza = df['Fuerza_Rozamiento_Y_Teorico'].values
    dt = 1.0 / fps
    desplazamientos = df['Velocidad_Y_Teorico'].values * dt
    trabajo = np.cumsum(fuerza * desplazamientos)
    df['Trabajo_Teorico'] = trabajo
        
    


def calculos_Energia(altura, masa, velocidad):
    """
    Calcula la energía potencial gravitacional y cinética de un objeto en cada instante.

    Parámetros:
        - altura: Altura del objeto sobre el nivel de referencia (m).
        - masa: Masa del objeto (kg).
        - velocidad: Velocidad del objeto en cada instante (m/s)).

    Devuelve:
        E_potencial (gravitacional con respecto al tiempo)
        E_cinetica con respecto al tiempo
        E_mecanica (que es la suma de las dos anteriores)
    """
    g = 9.81  # Aceleración gravitacional en m/s²
    E_Potencial = masa * g * altura
    E_Cinetica = 0.5 * masa * velocidad**2
    E_Mecanica = E_Potencial + E_Cinetica
    return E_Potencial, E_Cinetica, E_Mecanica


if __name__ == "__main__":
    main()
