import io
import os
import sys
import threading
import panel as pn
import plotly.graph_objects as go
from tkinter import ttk
import tkinter as tk

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tracker.trackerVideo import main as trackear_video


def main():
    ALTURA_CAIDA = 4.28

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
            velocidad_maxima = df['Velocidad_Y'].abs().max()
            acel_promedio_y = aceleración_promedio_y(df)
            aceleracion_terminal_aproximada = df['Aceleracion_Y'].rolling(3).mean().iloc[-6] #Recorto los ultimos 6 frames para evitar ruido
            energia_cinetica_maxima = df['Energia_Cinetica'].max()
            energia_mecanica_inicial = df['Energia_Mecanica'].iloc[0]
            energia_mecanica_final_aproximada = df['Energia_Mecanica'].rolling(3).mean().iloc[-6]
            variacion_energia_mecanica = energia_mecanica_final_aproximada - energia_mecanica_inicial
            fuerza_rozamiento_max = df['Fuerza_Rozamiento_Y'].abs().max()
            fuerza_rozamiento_teorica_max = df['Fuerza_Rozamiento_Y_Teorico'].max()
            impulso_maximo = df['Impulso'].abs().max()
            impulso_teorico_maximo = df['Impulso_Teorico'].abs().max()
            diferencia_impulso = impulso_maximo - impulso_teorico_maximo
            tiempo_total = df['Frame'].iloc[-1] / 60
            trabajo_experimental = df['Trabajo_Experimental'].iloc[-1]
            trabajo_teorico = df['Trabajo_Teorico'].iloc[-1]
            
            print(f"Datos del experimento: {seleccion}")
            print("--" * 40)
            print(f"Altura de caída: {ALTURA_CAIDA} m")
            print("Masa del objeto: 0.1 kg")
            print(f"Tiempo total de caída: {tiempo_total} s")
            print("--" * 40)
            print("VELOCIDAD\n")
            print(f"Velocidad máxima en Y: -{velocidad_maxima} m/s")
            print(f"Velocidad promedio basado en el csv: {vel_promedio_y}m/s")
            print("--" * 40)
            print("ACELERACIÓN\n")
            print(f"Aceleración terminal en Y (aproximada): {aceleracion_terminal_aproximada} m/s²")
            print(f"Aceleración promedio basado en el csv: {acel_promedio_y} m/s²")
            print("--" * 40)
            print("FUERZAS\n")
            print(f"Fuerza de rozamiento máxima (experimental): {fuerza_rozamiento_max} N")
            print(f"Fuerza de rozamiento máxima (modelo viscoso): {fuerza_rozamiento_teorica_max} N")
            print("--" * 40)
            print("ENERGÍAS\n")
            print(f"Energía mecánica inicial: {energia_mecanica_inicial} J")
            print(f"Energía mecánica final aproximada: {energia_mecanica_final_aproximada} J")
            print(f"Trabajo de la fuerza viscosa: {variacion_energia_mecanica} J")
            print(f"Energía cinética máxima: {energia_cinetica_maxima} J")
            print("--" * 40)
            print("IMPULSO\n")
            print(f"Impulso máximo (experimental): {impulso_maximo} N·s")
            print(f"Impulso máximo (modelo viscoso): {impulso_teorico_maximo} N·s")
            print(f"Diferencia entre impulso máximo experimental y teórico: {diferencia_impulso} N·s")
            print("--" * 40)
            print("TRABAJO\n")
            print(f"Trabajo experimental: {trabajo_experimental} J")
            print(f"Trabajo teórico: {trabajo_teorico} J")
            
            


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


def velocidad_promedio_y(df, recorte_bordes=5):
    if len(df) > recorte_bordes:
        df = df.iloc[0:-recorte_bordes]
    return df['Velocidad_Y'].mean()


def aceleración_promedio_y(df, recorte_bordes_finales=5, recorte_bordes_iniciales=2):
    if len(df) > (recorte_bordes_finales + recorte_bordes_iniciales):
        df = df.iloc[recorte_bordes_iniciales:-recorte_bordes_finales]
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



def graficar_resultados(csv_path, altura_caida,titulo, recorte_bordes=5):
    # Leer el archivo CSV con los datos completos
    df = pd.read_csv(csv_path)
    
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
    
    #El recorte no se aplica a la posicion ya que el grafico se ve bien.

    tabs = pn.Tabs(
        ("Posición en X", crear_figura("Posición en X vs Tiempo", [
            go.Scatter(x=tiempos, y=df['X_metros'],
                       mode='lines+markers', name='Real')
        ], "Tiempo (s)", "Posición X (m)")),

        ("Posición en Y", crear_figura("Posición en Y vs Tiempo", [
            go.Scatter(x=tiempos, y=df['Y_metros'],
                       mode='lines+markers', name='Real'),
            go.Scatter(x=tiempos, y=posiciones_mruv,
                       mode='lines', name='Caida libre sin rozamiento'),
            go.Scatter(x=tiempos, y=df['Posicion_Ajuste_Viscoso_Y'], mode='lines',
                       name='Ajuste al modelo viscoso', line=dict(dash='dash')),
            go.Scatter(x=tiempos, y=df['Posicion_Y_Teorico'], mode='lines',
                       name='Modelo viscoso', line=dict(dash='dot', color='magenta'))
        ], "Tiempo (s)", "Altura Y (m)")),

        ("Velocidad en X", crear_figura("Velocidad en X vs Tiempo", [
            go.Scatter(x=tiempos[0:-recorte_bordes], y=df['Velocidad_X'],
                       mode='lines+markers', name='Real')
        ], "Tiempo (s)", "Velocidad X (m/s)")),

        ("Velocidad en Y", crear_figura("Velocidad en Y vs Tiempo", [
            go.Scatter(x=tiempos[0:-recorte_bordes], y=df['Velocidad_Y'],
                       mode='lines+markers', name='Real'),
            go.Scatter(x=tiempos[0:-recorte_bordes], y=velocidades_mruv,
                       mode='lines', name='Caida libre sin rozamiento'),
            go.Scatter(x=tiempos[0:-recorte_bordes], y=df['Velocidad_Ajuste_Viscoso_Y'], mode='lines',
                       name='Ajuste al modelo viscoso', line=dict(dash='dash')),
            go.Scatter(x=tiempos[0:-recorte_bordes], y=df['Velocidad_Y_Teorico'], mode='lines',
                       name='Modelo viscoso', line=dict(dash='dot', color='magenta'))

        ], "Tiempo (s)", "Velocidad Y (m/s)")),

        ("Aceleración en X", crear_figura("Aceleración en X vs Tiempo", [
            go.Scatter(x=tiempos[0:-recorte_bordes], y=df['Aceleracion_X'],
                       mode='lines+markers', name='Real')
        ], "Tiempo (s)", "Aceleración X (m/s²)")),

        #En la aceleracion en Y, se recortan los primeros 2 frames ya que quedaban en 0
        ("Aceleración en Y", crear_figura("Aceleración en Y vs Tiempo", [
            go.Scatter(x=tiempos[2:-recorte_bordes], y=df['Aceleracion_Y'].iloc[2:],
                    mode='lines+markers', name='Real'),
            go.Scatter(x=tiempos[2:-recorte_bordes], y=aceleraciones_mruv[2:],
                    mode='lines', name='Caida libre sin rozamiento'),
            go.Scatter(x=tiempos[2:-recorte_bordes], y=df['Aceleracion_Ajuste_Viscoso_Y'].iloc[2:], mode='lines',
                    name='Ajuste constante', line=dict(dash='dash'))
        ], "Tiempo (s)", "Aceleración Y (m/s²)")),

        ("Fuerza vs Velocidad", crear_figura("Fuerza de Rozamiento vs Velocidad", [
            go.Scatter(x=df['Velocidad_Y'], y=df['Fuerza_Rozamiento_Y'],
                       mode='lines+markers', name='Fuerza Viscosa según datos reales', line=dict(color='pink')),
            go.Scatter(x=df['Velocidad_Y'], y=df['Fuerza_Rozamiento_Y_Teorico'],
                       mode='lines+markers', name='Fuerza Viscosa según modelo viscoso')
        ], "Velocidad Y (m/s)", "Fuerza (N)")),

        ("Fuerza vs Tiempo", crear_figura("Fuerza de Rozamiento vs Tiempo", [
            go.Scatter(x=tiempos[0:-recorte_bordes], y=df['Fuerza_Rozamiento_Y'],
                       mode='lines+markers', name='Fuerza Viscosa según datos experimentales'),
            go.Scatter(x=tiempos[0:-recorte_bordes], y=df['Fuerza_Rozamiento_Y_Teorico'],
                       mode='lines+markers', name='Fuerza Viscosa según modelo viscoso')
        ], "Tiempo (s)", "Fuerza (N)")),

        ("Impulso vs Tiempo", crear_figura("Impulso vs Tiempo", [
            go.Scatter(x=tiempos[0:-recorte_bordes], y=df['Impulso'],
                       mode='lines+markers', name='Impulso Experimental', line=dict(color='lightgreen')),
            go.Scatter(x=tiempos[0:-recorte_bordes], y=df['Impulso_Teorico'],
                       mode='lines+markers', name='Impulso según modelo viscoso', line=dict(color='green'))
        ], "Tiempo (s)", "Impulso (N·s)")),
        ("Trabajo vs Tiempo", crear_figura("Trabajo vs Tiempo", [
            go.Scatter(x=tiempos, y=df['Trabajo_Experimental'],
                    mode='lines+markers', name='Trabajo Experimental'),
            go.Scatter(x=tiempos, y=df['Trabajo_Teorico'],
                    mode='lines+markers', name='Trabajo Teórico', line=dict(dash='dot'))
        ], "Tiempo (s)", "Trabajo (J)")),
        
        

        ("Energías vs Tiempo", crear_figura("Energía Potencial, Cinética y Mecánica vs Tiempo", [
            go.Scatter(x=tiempos[0:-recorte_bordes], y=df['Energia_Potencial'], mode='lines',
                       name='E. Potencial (Experimental)', line=dict(color='blue')),
            go.Scatter(x=tiempos[0:-recorte_bordes], y=df['Energia_Cinetica'], mode='lines',
                       name='E. Cinética (Experimental)', line=dict(color='red')),
            go.Scatter(x=tiempos[0:-recorte_bordes], y=df['Energia_Mecanica'], mode='lines',
                       name='E. Mecánica (Experimental)', line=dict(color='green')),
            go.Scatter(x=tiempos[0:-recorte_bordes], y=df['Energia_Potencial_Teorico'], mode='lines',
                       name='E. Potencial (Viscoso)', line=dict(color='navy', dash='dot')),
            go.Scatter(x=tiempos[0:-recorte_bordes], y=df['Energia_Cinetica_Teorico'], mode='lines',
                       name='E. Cinética (Viscoso)', line=dict(color='darkred', dash='dot')),
            go.Scatter(x=tiempos[0:-recorte_bordes], y=df['Energia_Mecanica_Teorico'], mode='lines',
                       name='E. Mecánica (Viscoso)', line=dict(color='darkgreen', dash='dot')),
            go.Scatter(x=tiempos[0:-recorte_bordes], y=df['Energia_Potencial_Ajuste'], mode='lines',
                       name='E. Potencial (Ajuste)', line=dict(color='deepskyblue', dash='dash')),
            go.Scatter(x=tiempos[0:-recorte_bordes], y=df['Energia_Cinetica_Ajuste'], mode='lines',
                       name='E. Cinética (Ajuste)', line=dict(color='orange', dash='dash')),
            go.Scatter(x=tiempos[0:-recorte_bordes], y=df['Energia_Mecanica_Ajuste'], mode='lines',
                       name='E. Mecánica (Ajuste)', line=dict(color='lime', dash='dash'))
        ], "Tiempo (s)", "Energía (J)")),
    )
    header = pn.pane.Markdown(f"# {titulo}", sizing_mode='stretch_width')
    return pn.Column(header, tabs, sizing_mode='stretch_both'), tabs


if __name__ == "__main__":
    main()
