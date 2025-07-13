
from tracker.trackerVideo import main as trackear_video
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
import io

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def main():
    ALTURA_CAIDA = 4.28
    MASA_OBJETO = 0.1

    archivos = {
        "Sin Globo": ("data/trayectoria_sin_globo.csv", 'media/oso_recortados/oso_sin_globo.mov'),
        "Globo chico": ("data/trayectoria_globo_chico.csv", 'media/oso_recortados/oso_globo_chico.mov'),
        "Globo mediano": ("data/trayectoria_globo_mediano.csv", 'media/oso_recortados/oso_globo_mediano.mov'),
        "Globo grande": ("data/trayectoria_globo_grande.csv", 'media/oso_recortados/oso_globo_grande.mov')
    }
    ruta_csv, ruta_video = archivos["Sin Globo"]
    seleccion = "Sin Globo"

    def on_selection(event):
        def task():
            nonlocal ruta_csv, ruta_video, seleccion
            seleccion = combo.get()
            ruta = archivos[seleccion]
            df = pd.read_csv(ruta_csv)

            ruta_csv, ruta_video = archivos[seleccion]
        threading.Thread(target=task).start()

    def show_graphs():
        # Mapea los prints a la GUI
        buffer = io.StringIO()
        sys_stdout_original = sys.stdout
        sys.stdout = buffer

        def task():
            df = pd.read_csv(ruta_csv)

            vel_promedio_y = velocidad_promedio_y(df)
            acel_promedio_y = aceleración_promedio_y(df)
            print(f"Aceleración promedio basado en el csv: {acel_promedio_y}")
            print(f"Velocidad promedio basado en el csv: {vel_promedio_y}")

            panel, tabs = graficar_resultados(
                ruta_csv, ALTURA_CAIDA, seleccion)
            sys.stdout = sys_stdout_original
            resumen_texto = buffer.getvalue()
            resumen_pane = pn.pane.Markdown(
                f"```\n{resumen_texto}\n```", sizing_mode='stretch_width')
            tabs.append(("Resumen", resumen_pane))
            panel.show()

        threading.Thread(target=task, daemon=True).start()

    def track_video():
        print("Iniciar el tracker de video...", ruta_video)
        trackear_video(ruta_csv, ruta_video)
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

    trackear_btn = tk.Button(root, text="Trackear Video", command=track_video)
    trackear_btn.pack(pady=5)
    ver_graficos_btn = tk.Button(
        root, text="Ver Gráficos", command=show_graphs)
    ver_graficos_btn.pack(pady=5)

    root.mainloop()


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


def graficar_resultados(csv_path, altura_caida, titulo, recorte_bordes=6):
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
            go.Scatter(x=tiempos, y=df['Posicion_Ajuste_Viscoso_Y'], mode='lines',
                       name='Ajuste al modelo viscoso', line=dict(dash='dash')),
            go.Scatter(x=tiempos, y=df['Posicion_Y_Teorico'], mode='lines',
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
            go.Scatter(x=tiempos, y=df['Velocidad_Ajuste_Viscoso_Y'], mode='lines',
                       name='Ajuste al modelo viscoso', line=dict(dash='dash')),
            go.Scatter(x=tiempos, y=df['Velocidad_Y_Teorico'], mode='lines',
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
            go.Scatter(x=tiempos, y=df['Aceleracion_Ajuste_Viscoso_Y'], mode='lines',
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
            go.Scatter(x=tiempos, y=df['Energia_Potencial'], mode='lines',
                       name='E. Potencial (Experimental)', line=dict(color='blue')),
            go.Scatter(x=tiempos, y=df['Energia_Cinetica'], mode='lines',
                       name='E. Cinética (Experimental)', line=dict(color='red')),
            go.Scatter(x=tiempos, y=df['Energia_Mecanica'], mode='lines',
                       name='E. Mecánica (Experimental)', line=dict(color='green')),
            go.Scatter(x=tiempos, y=df['Energia_Potencial_Teorico'], mode='lines',
                       name='E. Potencial (Viscoso)', line=dict(color='navy', dash='dot')),
            go.Scatter(x=tiempos, y=df['Energia_Cinetica_Teorico'], mode='lines',
                       name='E. Cinética (Viscoso)', line=dict(color='darkred', dash='dot')),
            go.Scatter(x=tiempos, y=df['Energia_Mecanica_Teorico'], mode='lines',
                       name='E. Mecánica (Viscoso)', line=dict(color='darkgreen', dash='dot')),
            go.Scatter(x=tiempos, y=df['Energia_Potencial_Ajuste'], mode='lines',
                       name='E. Potencial (Ajuste)', line=dict(color='deepskyblue', dash='dash')),
            go.Scatter(x=tiempos, y=df['Energia_Cinetica_Ajuste'], mode='lines',
                       name='E. Cinética (Ajuste)', line=dict(color='orange', dash='dash')),
            go.Scatter(x=tiempos, y=df['Energia_Mecanica_Ajuste'], mode='lines',
                       name='E. Mecánica (Ajuste)', line=dict(color='lime', dash='dash'))
        ], "Tiempo (s)", "Energía (J)")),
    )
    header = pn.pane.Markdown(f"# {titulo}", sizing_mode='stretch_width')
    return pn.Column(header, tabs, sizing_mode='stretch_both'), tabs


if __name__ == "__main__":
    main()
