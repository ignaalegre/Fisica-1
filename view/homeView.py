import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import ttk


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

        graficar_resultados(ruta, ALTURA_CAIDA)

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


def graficar_resultados(csv_path, altura_caida, recorte_bordes=6):
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
    ajuste_pos_Y, coef_pos_Y = ajuste_parabolico(tiempos, df['Y_metros'])
    ajuste_vel_Y, coef_vel_Y = ajuste_lineal(tiempos, df['Velocidad_Y'])
    ajuste_ace_Y, coef_ace_Y = ajuste_constante(tiempos, df['Aceleracion_Y'])

    print("\n--- Ajustes ---")
    print(
        f"Posición Y: a={coef_pos_Y[0]:.3f}, b={coef_pos_Y[1]:.3f}, c={coef_pos_Y[2]:.3f}")
    print(f"Velocidad Y: m={coef_vel_Y[0]:.3f}, b={coef_vel_Y[1]:.3f}")
    print(f"Aceleración Y constante: {coef_ace_Y[0]:.3f}")

    plt.figure(figsize=(12, 18))

    # Crear una figura más grande para más subplots
    fig, axs = plt.subplots(4, 2, figsize=(14, 20))
    axs = axs.flatten()

    # Posición en X
    plt.subplot(3, 2, 1)
    plt.plot(tiempos, df['X_metros'], marker='o', label='Posición en X Real')
    plt.title('Posición en X vs Tiempo')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Posición X (m)')
    plt.grid()
    plt.legend()

    # Posición en Y
    plt.subplot(3, 2, 2)
    plt.plot(tiempos, df['Y_metros'], marker='o',
             color='hotpink', label='Posición en Y Real')
    plt.plot(tiempos, posiciones_mruv, color='lightskyblue',
             label='Posición en Y (teórico)')
    plt.plot(tiempos, ajuste_pos_Y, '--',
             color='darkviolet', label='Ajuste parabólico')
    plt.title('Posición en Y vs Tiempo')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Altura (m)')
    plt.grid()
    plt.legend()

    # Velocidad en X
    plt.subplot(3, 2, 3)
    plt.plot(tiempos, df['Velocidad_X'], marker='o',
             label='Velocidad en X Real')
    plt.title('Velocidad en X vs Tiempo')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Velocidad X (m/s)')
    plt.grid()
    plt.legend()

    # Velocidad en Y
    plt.subplot(3, 2, 4)
    plt.plot(tiempos, df['Velocidad_Y'], marker='o',
             color='hotpink', label='Velocidad en Y Real')
    plt.plot(tiempos, velocidades_mruv, color='lightskyblue',
             label='Velocidad en Y (teórico)')
    plt.plot(tiempos, ajuste_vel_Y, '--',
             color='darkviolet', label='Ajuste lineal')
    plt.title('Velocidad en Y vs Tiempo')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Velocidad Y (m/s)')
    plt.grid()
    plt.legend()

    # Aceleración en X
    plt.subplot(3, 2, 5)
    plt.plot(tiempos, df['Aceleracion_X'], marker='o',
             label='Aceleración en X Real')
    plt.title('Aceleración en X vs Tiempo')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Aceleración X (m/s²)')
    plt.grid()
    plt.legend()

    # Aceleración en Y
    plt.subplot(3, 2, 6)
    plt.plot(tiempos, df['Aceleracion_Y'], marker='o',
             color='hotpink', label='Aceleración en Y Real')
    plt.plot(tiempos, aceleraciones_mruv, color='lightskyblue',
             label='Aceleración en Y (teórico)')
    plt.plot(tiempos, ajuste_ace_Y, '--',
             color='darkviolet', label='Ajuste constante')
    plt.title('Aceleración en Y vs Tiempo')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Aceleración Y (m/s²)')
    plt.grid()
    plt.legend()
    print(f"*Aceleracion  promedio en Y en base al csv:",
          df['Aceleracion_Y'].mean())
    print(f"*Velocidad promedio en Y en base al csv:",
          df['Velocidad_Y'].mean())

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
