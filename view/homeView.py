import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import ttk
import plotly.graph_objects as go
import panel as pn
import threading
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tracker.trackerVideo import main as trackear_video


def main():
    ALTURA_CAIDA = 4.28
    MASA_OBJETO = 0.1

    archivos = {
        "Sin Globo": ("data/trayectoria_sin_globo.csv", 'media/oso_recortados/oso_sin_globo.mov'),
        "Globo chico": ("data/trayectoria_globo_chico.csv", 'media/oso_recortados/oso_globo_chico.mov'),
        "Globo mediano": ("data/trayectoria_globo_mediano.csv",'media/oso_recortados/oso_globo_mediano.mov'),
        "Globo grande": ("data/trayectoria_globo_grande.csv",'media/oso_recortados/oso_globo_grande.mov')
    }
    ruta_csv, ruta_video = archivos["Sin Globo"]

    def on_selection(event):
        def task():
            seleccion = combo.get()
            ruta_csv, ruta_video = archivos[seleccion]
        threading.Thread(target=task).start()
        
    def show_graphs():
        def task():
            df = pd.read_csv(ruta_csv)
            estimar_constante_viscosa_con_ajuste_lineal(df, MASA_OBJETO)
            estimar_constante_viscosa_con_ajuste_cuadrático(df, MASA_OBJETO)
            vel_promedio_y = velocidad_promedio_y(df)
            acel_promedio_y = aceleración_promedio_y(df)
            print(f"Aceleración promedio basado en el csv: {acel_promedio_y}")
            print(f"Velocidad promedio basado en el csv: {vel_promedio_y}")

            k = estimar_constante_viscosa_con_ajuste_lineal(df, MASA_OBJETO)
            tabs = graficar_resultados(ruta_csv, ALTURA_CAIDA, MASA_OBJETO, k)
            tabs.show()
        threading.Thread(target=task).start()
        
    def track_video():
        print("Iniciar el tracker de video...", ruta_video)
        trackear_video(ruta_csv,ruta_video)
        print("Tracker de video finalizado.")
        
        
    

    # Crear GUI simple con tkinter
    root = tk.Tk()
    root.title("Visualizador de Experimentos")
    root.geometry("400x180")

    label = tk.Label(root, text="Selecciona un experimento:")
    label.pack(pady=10)
    


    combo = ttk.Combobox(root, values=list(archivos.keys()), state="readonly")
    combo.pack(pady=(0, 10))
    combo.bind("<<ComboboxSelected>>", on_selection)
    combo.set("Sin Globo")  # Valor por defecto
    
    ver_graficos_btn = tk.Button(root, text="Ver Gráficos", command=show_graphs) 
    ver_graficos_btn.pack(pady=5)
    
    trackear_btn = tk.Button(root, text="Trackear Video",command=track_video)
    trackear_btn.pack(pady=5)

    root.mainloop()


def ajuste_parabolico(t, y):
    def modelo(t, a, b, c): return a * t**2 + b * t + c
    params, _ = curve_fit(modelo, t, y)
    return modelo(t, *params), params


def ajuste_lineal(t, y):
    def modelo(t, m, b): return m * t + b
    params, _ = curve_fit(modelo, t, y)
    return modelo(t, *params), params


def ajuste_constante(t, y):
    def modelo(t, c): return np.full_like(t, c)
    params, _ = curve_fit(modelo, t, y)
    return modelo(t, *params), params


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
        v_terminal = masa * g / k
        return -v_terminal * (1 - np.exp(-k * t / masa))
    params, _ = curve_fit(modelo_velocidad, t, y)
    return modelo_velocidad(t, *params), params


def estimar_constante_viscosa_con_ajuste_lineal(df, masa_objeto):
    # Recortar posibles extremos ruidosos
    df = df.iloc[0:-5] if len(df) > 10 else df.copy()

    # Realizar el ajuste lineal a_Y = a0 + b * v_Y
    vel_y = df['Velocidad_Y'].values
    acel_y = df['Aceleracion_Y'].values

    ajustado, (m, b) = ajuste_lineal(vel_y, acel_y)

    k = -masa_objeto * m  # porque m ≈ -k/m

    # Calcular R²
    ss_res = np.sum((acel_y - ajustado) ** 2)
    ss_tot = np.sum((acel_y - np.mean(acel_y)) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"Modelo ajustado: a_Y = {b:.3f} + ({m:.3f}) * v_Y")
    print(f"Constante viscosa estimada con ajuste lineal: k = {k:.4f} kg/s")
    print(f"Coeficiente de correlación R² = {r2:.4f}")

    return k


def estimar_constante_viscosa_con_ajuste_cuadrático(df, masa_objeto):
    # Recortar posibles extremos ruidosos
    df = df.iloc[0:-5] if len(df) > 10 else df.copy()

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


def velocidad_promedio_y(df):
    return df['Velocidad_Y'].mean()


def aceleración_promedio_y(df):
    return df['Aceleracion_Y'].mean()


def calcular_mruv(tiempos, altura_caida):
    # Aceleración por la gravedad
    g = 9.81  # m/s^2
    v0 = 0  # Velocidad inicial
    y0 = altura_caida  # Altura inicial

    # Calcular la posición, velocidad y aceleración en el tiempo para MRUV
    # Suponiendo caída libre desde una altura
    posiciones_mruv = y0 - (1/2) * g * tiempos**2
    velocidades_mruv = -g * tiempos  # Velocidad en caída libre
    # Aceleración constante en el MRUV
    aceleraciones_mruv = np.full_like(tiempos, -g)

    return posiciones_mruv, velocidades_mruv, aceleraciones_mruv


def calcular_modelo_viscoso(tiempos, masa, k, altura_inicial):
    """
    Derivación:
    Por 2da ley de Newton:
        ∑F = m·a → -mg + (-kv) = m·dv/dt
        dv/dt + (k/m)v = -g   → EDO lineal

    Solución por factor integrante:
        v(t) = (m*-g/k)(1 - e^(-k·t/m))

    Luego, integrando para y(t):
        y(t) = y0 - (mg/k)t + (m²g/k²)(1 - e^(-k·t/m))

    Parámetros:
        - tiempos: np.array con los valores de tiempo
        - masa: masa del objeto (kg)
        - k: constante de rozamiento viscoso (kg/s)
        - altura_inicial: y0 (m)

    Devuelve:
        - velocidades según el modelo con rozamiento viscoso
        - posiciones según el mismo modelo
    """
    g = 9.81
    v_terminal = masa * -g / k
    vel = v_terminal * (1 - np.exp(-k * tiempos / masa))
    pos = altura_inicial + v_terminal * tiempos - \
        (masa * v_terminal / k) * (1 - np.exp(-k * tiempos / masa))
    return vel, pos

def calculos_Energia(tiempos, altura, masa, velocidad):
    """
    Calcula la energía potencial gravitacional y cinética de un objeto en cada instante.

    Parámetros:
        - tiempos: np.array con los valores de tiempo.
        - altura: Altura del objeto sobre el nivel de referencia (m).
        - masa: Masa del objeto (kg).
        - velocidad: Velocidad del objeto en cada instante (m/s)).

    Devuelve:
        E_potencial (gravitacional con respecto al tiempo),
        E_cinetica con respecto al tiempo
        E_mecanica (que es la suma de las dos anteriores)
    """
    g = 9.81  # Aceleración gravitacional en m/s²
    E_Potencial =  masa * g * altura
    E_Cinetica = 0.5 * masa * velocidad**2
    E_Mecanica = E_Potencial + E_Cinetica
    return E_Potencial, E_Cinetica, E_Mecanica


def graficar_resultados(csv_path, altura_caida, masa, k_estimado, recorte_bordes=6):
    # Leer el archivo CSV con los datos completos
    df = pd.read_csv(csv_path)

    # Recortar bordes problemáticos
    if len(df) > 2 * recorte_bordes:
        df = df.iloc[0:-recorte_bordes]

    # Crear el eje de tiempo basado en los frames
    tiempos = df['Frame'] / 60

    # Calcular las posiciones, velocidades y aceleraciones según MRUV
    posiciones_mruv, velocidades_mruv, aceleraciones_mruv = calcular_mruv(
        tiempos, altura_caida)

    velocidades_viscoso, posiciones_viscoso = calcular_modelo_viscoso(
        tiempos, masa, k_estimado, altura_caida
    )

    # Calcular energías usando los datos reales
    E_Potencial, E_Cinetica, E_Mecanica = calculos_Energia(
        tiempos, df['Y_metros'], masa, df['Velocidad_Y']
    )

    # Calcular energías teóricas con rozamiento viscoso
    E_Potencial_visc, E_Cinetica_visc, E_Mecanica_visc = calculos_Energia(
        tiempos, posiciones_viscoso, masa, velocidades_viscoso
    )
    ajuste_pos_Y, coef_pos_Y = ajuste_posicion_viscoso(
        tiempos, df['Y_metros'].values, masa, altura_caida)
    ajuste_vel_Y, coef_vel_Y = ajuste_velocidad_viscoso(
        tiempos, df['Velocidad_Y'].values, masa)
    ajuste_ace_Y, coef_ace_Y = ajuste_constante(tiempos, df['Aceleracion_Y'])

    # Calcular energías con el ajuste al modelo viscoso
    E_Potencial_ajuste, E_Cinetica_ajuste, E_Mecanica_ajuste = calculos_Energia(
        tiempos, ajuste_pos_Y, masa, ajuste_vel_Y
    )

    print("\n--- Ajustes ---")
    print(
        f"Posición Y: k={coef_pos_Y[0]:.3f}")
    print(f"Velocidad Y: k={coef_vel_Y[0]:.3f}")

    print(f"Aceleración Y constante: {coef_ace_Y[0]:.3f}")

    def crear_figura(titulo, trazas, xlabel, ylabel):
        fig = go.Figure()
        for traza in trazas:
            fig.add_trace(traza)
        fig.update_layout(
            title=titulo,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            height=500
        )
        return pn.pane.Plotly(fig, config={'responsive': True})

    tabs = pn.Tabs(
        ("Posición en X", crear_figura("Posición en X vs Tiempo", [
            go.Scatter(x=tiempos, y=df['X_metros'],
                       mode='lines+markers', name='Real')
        ], "Tiempo (s)", "Posición X (m)")),

        ("Posición en Y", crear_figura("Posición en Y vs Tiempo", [
            go.Scatter(x=tiempos, y=df['Y_metros'],
                       mode='lines+markers', name='Real'),
            go.Scatter(x=tiempos, y=posiciones_mruv,
                       mode='lines', name='MRUV'),
            go.Scatter(x=tiempos, y=ajuste_pos_Y, mode='lines',
                       name='Ajuste al modelo viscoso', line=dict(dash='dash')),
            go.Scatter(x=tiempos, y=posiciones_viscoso, mode='lines',
                       name='Modelo viscoso', line=dict(dash='dot', color='magenta'))
        ], "Tiempo (s)", "Altura Y (m)")),

        ("Velocidad en X", crear_figura("Velocidad en X vs Tiempo", [
            go.Scatter(x=tiempos, y=df['Velocidad_X'],
                       mode='lines+markers', name='Real')
        ], "Tiempo (s)", "Velocidad X (m/s)")),

        ("Velocidad en Y", crear_figura("Velocidad en Y vs Tiempo", [
            go.Scatter(x=tiempos, y=df['Velocidad_Y'],
                       mode='lines+markers', name='Real'),
            go.Scatter(x=tiempos, y=velocidades_mruv,
                       mode='lines', name='MRUV'),
            go.Scatter(x=tiempos, y=ajuste_vel_Y, mode='lines',
                       name='Ajuste al modelo viscoso', line=dict(dash='dash')),
            go.Scatter(x=tiempos, y=velocidades_viscoso, mode='lines',
                       name='Modelo viscoso', line=dict(dash='dot', color='magenta'))

        ], "Tiempo (s)", "Velocidad Y (m/s)")),

        ("Aceleración en X", crear_figura("Aceleración en X vs Tiempo", [
            go.Scatter(x=tiempos, y=df['Aceleracion_X'],
                       mode='lines+markers', name='Real')
        ], "Tiempo (s)", "Aceleración X (m/s²)")),

        ("Aceleración en Y", crear_figura("Aceleración en Y vs Tiempo", [
            go.Scatter(x=tiempos, y=df['Aceleracion_Y'],
                       mode='lines+markers', name='Real'),
            go.Scatter(x=tiempos, y=aceleraciones_mruv,
                       mode='lines', name='MRUV'),
            go.Scatter(x=tiempos, y=ajuste_ace_Y, mode='lines',
                       name='Ajuste constante', line=dict(dash='dash'))
        ], "Tiempo (s)", "Aceleración Y (m/s²)")),

        ("Fuerza vs Velocidad", crear_figura("Fuerza de Rozamiento vs Velocidad", [
            go.Scatter(x=df['Velocidad_Y'], y=df['Fuerza_Rozamiento_Y'],
                       mode='lines+markers', name='Datos')
        ], "Velocidad Y (m/s)", "Fuerza (N)")),

        ("Fuerza vs Tiempo", crear_figura("Fuerza de Rozamiento vs Tiempo", [
            go.Scatter(x=tiempos, y=df['Fuerza_Rozamiento_Y'],
                       mode='lines+markers', name='Fuerza')
        ], "Tiempo (s)", "Fuerza (N)")),

        ("Impulso vs Tiempo", crear_figura("Impulso vs Tiempo", [
            go.Scatter(x=tiempos, y=df['Impulso'],
                       mode='lines+markers', name='Impulso Experimental', line=dict(color='lightgreen'))
        ], "Tiempo (s)", "Impulso (N·s)")),

        ("Energías vs Tiempo", crear_figura("Energía Potencial, Cinética y Mecánica vs Tiempo", [
            go.Scatter(x=tiempos, y=E_Potencial, mode='lines', name='E. Potencial (Experimental)', line=dict(color='blue')),
            go.Scatter(x=tiempos, y=E_Cinetica, mode='lines', name='E. Cinética (Experimental)', line=dict(color='red')),
            go.Scatter(x=tiempos, y=E_Mecanica, mode='lines', name='E. Mecánica (Experimental)', line=dict(color='green')),
            go.Scatter(x=tiempos, y=E_Potencial_visc, mode='lines', name='E. Potencial (Viscoso)', line=dict(color='navy', dash='dot')),
            go.Scatter(x=tiempos, y=E_Cinetica_visc, mode='lines', name='E. Cinética (Viscoso)', line=dict(color='darkred', dash='dot')),
            go.Scatter(x=tiempos, y=E_Mecanica_visc, mode='lines', name='E. Mecánica (Viscoso)', line=dict(color='darkgreen', dash='dot')),
            go.Scatter(x=tiempos, y=E_Potencial_ajuste, mode='lines', name='E. Potencial (Ajuste)', line=dict(color='deepskyblue', dash='dash')),
            go.Scatter(x=tiempos, y=E_Cinetica_ajuste, mode='lines', name='E. Cinética (Ajuste)', line=dict(color='orange', dash='dash')),
            go.Scatter(x=tiempos, y=E_Mecanica_ajuste, mode='lines', name='E. Mecánica (Ajuste)', line=dict(color='lime', dash='dash'))
        ], "Tiempo (s)", "Energía (J)")),
    )

    return tabs


if __name__ == "__main__":
    main()
