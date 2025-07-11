import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import ttk
import plotly.graph_objects as go
import panel as pn


def main():
    ALTURA_CAIDA = 4.28
    MASA_OBJETO = 0.1

    archivos = {
        "Sin Globo": "data/trayectoria_sin_globo.csv",
        "Globo chico": "data/trayectoria_globo_chico.csv",
        "Globo mediano": "data/trayectoria_globo_mediano.csv",
        "Globo grande": "data/trayectoria_globo_grande.csv"
    }

    def on_selection(event):
        seleccion = combo.get()
        ruta = archivos[seleccion]
        df = pd.read_csv(ruta)

        estimar_constante_viscosa_con_ajuste_lineal(df, MASA_OBJETO)
        estimar_constante_viscosa_con_ajuste_cuadrático(df, MASA_OBJETO)
        vel_promedio_y = velocidad_promedio_y(df)
        acel_promedio_y = aceleración_promedio_y(df)
        print(f"Aceleración promedio basado en el csv: {acel_promedio_y}")
        print(f"Velocidad promedio basado en el csv: {vel_promedio_y}")

        k = estimar_constante_viscosa_con_ajuste_lineal(df, MASA_OBJETO)
        tabs = graficar_resultados(ruta, ALTURA_CAIDA, MASA_OBJETO, k)
        tabs.show()

    # Crear GUI simple con tkinter
    root = tk.Tk()
    root.title("Visualizador de Experimentos")
    root.geometry("400x120")

    label = tk.Label(root, text="Selecciona un experimento:")
    label.pack(pady=10)

    combo = ttk.Combobox(root, values=list(archivos.keys()), state="readonly")
    combo.pack()
    combo.bind("<<ComboboxSelected>>", on_selection)

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

    ajuste_pos_Y, coef_pos_Y = ajuste_posicion_viscoso(
        tiempos, df['Y_metros'].values, masa, altura_caida)
    ajuste_vel_Y, coef_vel_Y = ajuste_velocidad_viscoso(
        tiempos, df['Velocidad_Y'].values, masa)
    ajuste_ace_Y, coef_ace_Y = ajuste_constante(tiempos, df['Aceleracion_Y'])

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


    )

    return tabs


if __name__ == "__main__":
    main()
