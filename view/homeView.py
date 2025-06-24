import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.stats import linregress
import tkinter as tk
from tkinter import ttk

def main():
    ALTURA_CAIDA = 4.28  # en metros
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

        k = estimar_constante_viscosa(df, MASA_OBJETO)
        print(f"Constante viscosa estimada: {k}")
        vel_promedio_y = velocidad_promedio_y(df)
        acel_promedio_y = aceleración_promedio_y(df)
        calcular_fuerzas(k, MASA_OBJETO, acel_promedio_y, vel_promedio_y)

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
     
    


def estimar_constante_viscosa(df, masa_objeto):
    # Recortar posibles extremos ruidosos
    df = df.iloc[5:-5] if len(df) > 10 else df.copy()

    # Realizar el ajuste lineal a_Y = a0 + b * v_Y
    vel_y = df['Velocidad_Y'].values
    acel_y = df['Aceleracion_Y'].values

    slope, intercept, r_value, _, _ = linregress(vel_y, acel_y)

    k = -masa_objeto * slope  # porque slope = -k/m

    print(f"Modelo ajustado: a_Y = {intercept:.3f} + ({slope:.3f}) * v_Y")
    print(f"Constante viscosa estimada: k = {k:.4f} kg/s")
    print(f"Coeficiente de correlación R² = {r_value**2:.4f}")

    return k

def velocidad_promedio_y(df):
    return df['Velocidad_Y'].mean()


def aceleración_promedio_y(df):
    return df['Aceleracion_Y'].mean()

def calcular_fuerzas(k, masa, aceleración_promedio, velocidad_promedio):
    print(
        f"Sumatoria de fuerzas usando solo la masa y aceleración = {masa*aceleración_promedio}")
    # la fuerza viscosa es negativa porque la velocidad es negativa
    fuerza_viscosa = -k*velocidad_promedio
    print(f"Fuerza viscosa = {fuerza_viscosa}")
    fuerza_peso = masa*-9.8
    print(f"Fuerza peso = {fuerza_peso}")
    print(
        f"Sumatoria de fuerzas sumando las 2 fuerzas calculadas= {-fuerza_peso+fuerza_viscosa}")
    
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
    print(f"*Aceleracion  promedio en Y en base al csv:" ,df['Aceleracion_Y'].mean())
    print(f"*Velocidad promedio en Y en base al csv:" ,df['Velocidad_Y'].mean())
    plt.tight_layout()
    plt.show()
    
    
    
    
if __name__ == "__main__":
    main()