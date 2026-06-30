#!/usr/bin/env python3
"""
🌦️ PROYECTO: Operaciones Vectorizadas con numpy
=================================================
Nivel 1 — Skill: numpy (arrays, operaciones vectorizadas, broadcasting)

¿QUÉ VAS A APRENDER?
- Crear arrays de numpy desde listas, rangos, y datos aleatorios
- Acceder a elementos con indexación y slicing (como listas pero más potente)
- Operaciones vectorizadas: cálculos en TODO el array sin usar for loops
- Broadcasting: operar arrays de diferentes formas
- Filtrado con máscaras booleanas (condiciones sobre arrays)
- Estadísticas descriptivas con numpy
- Comparar velocidad: loops en Python puro vs numpy vectorizado

DATOS SIMULADOS: Sensores meteorológicos en 3 ciudades chilenas
- Santiago, Valparaíso, Concepción
- 7 días de lecturas: temperatura, humedad, presión, viento, lluvia
- Los datos son simulados pero realistas (basados en climatología real)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time  # Para medir tiempos de ejecución

# =====================================================================
# Crear carpeta para gráficos
# =====================================================================
os.makedirs('docs', exist_ok=True)

print("=" * 72)
print("🌦️  BIENVENIDO A NUMPY: OPERACIONES VECTORIZADAS")
print("=" * 72)
print("\n🧠 numpy es el motor que mueve el Machine Learning en Python.")
print("   pandas, scikit-learn, TensorFlow... TODOS usan numpy por debajo.")
print("   Hoy entenderás POR QUÉ es tan rápido y cómo usarlo.")

# =====================================================================
# PASO 1: Creando arrays de numpy — el pan de cada día en ML
# =====================================================================
print("\n" + "=" * 72)
print("📦 PASO 1: Crear arrays — los bloques de construcción")
print("=" * 72)

# ---------------------------------------------------------------------
# 1.1: Array desde una lista de Python (la forma más común)
# ---------------------------------------------------------------------
# Cuando cargas datos en ML, normalmente vienen de un CSV o una DB.
# Con np.array() conviertes cualquier lista en un array optimizado.
print("\n📌 1.1 Array desde una lista:")
temperaturas_santiago = np.array([22.5, 24.1, 19.8, 21.3, 25.7, 23.0, 20.4])
print(f"   temp_santiago = {temperaturas_santiago}")
print(f"   Tipo: {type(temperaturas_santiago)}")
print(f"   dtype: {temperaturas_santiago.dtype}  ← float64 = decimales de 64 bits")
# El dtype es IMPORTANTE: float64 consume más memoria pero tiene más precisión.
# En ML a veces usamos float32 para ahorrar memoria.

# ---------------------------------------------------------------------
# 1.2: Array 2D (matriz) — como una tabla
# ---------------------------------------------------------------------
# En ML trabajamos con matrices: filas = muestras, columnas = características.
# Aquí: 7 filas (días) × 5 columnas (temperatura, humedad, presión, viento, lluvia)
print("\n📌 1.2 Array 2D — una matriz de datos (7 días × 5 sensores):")
# Semilla fija para que los datos sean REPRODUCIBLES (mismos siempre)
np.random.seed(42)

santiago = np.array([
    [22.5, 65, 1013, 12.1, 0.0],  # Día 1: temp, humedad, presión, viento, lluvia
    [24.1, 58, 1011, 15.3, 0.0],  # Día 2
    [19.8, 72, 1015, 8.7,  2.3],  # Día 3: ¡llovió!
    [21.3, 68, 1012, 11.5, 0.5],  # Día 4
    [25.7, 52, 1009, 18.2, 0.0],  # Día 5
    [23.0, 61, 1014, 13.8, 0.0],  # Día 6
    [20.4, 70, 1016, 9.4,  1.8],  # Día 7
])

print(f"   santiago (shape): {santiago.shape}")
print(f"   → {santiago.shape[0]} filas (días) × {santiago.shape[1]} columnas (sensores)")
print(f"   santiago =\n{santiago}")
# shape es el atributo MÁS USADO en ML. Te dice la dimensión de tus datos.
# Siempre revisa shape antes de hacer cualquier operación.

# ---------------------------------------------------------------------
# 1.3: Atributos importantes de los arrays
# ---------------------------------------------------------------------
print("\n📌 1.3 Atributos importantes de un array:")
print(f"   shape:  {santiago.shape}    ← dimensiones (filas, columnas)")
print(f"   ndim:   {santiago.ndim}     ← número de dimensiones")
print(f"   size:   {santiago.size}     ← total de elementos (7×5 = 35)")
print(f"   dtype:  {santiago.dtype}    ← tipo de datos")
print(f"   nbytes: {santiago.nbytes}   ← memoria usada en bytes")
# nbytes te ayuda a saber cuánta RAM usará tu dataset. Útil con datos grandes.

# ---------------------------------------------------------------------
# 1.4: Otras formas de crear arrays
# ---------------------------------------------------------------------
print("\n📌 1.4 Otras formas útiles de crear arrays:")
print("   (Estas funciones las usarás TODOS LOS DÍAS en ML)")

# np.zeros: para inicializar matrices vacías (todo ceros)
# Sirve para crear "contenedores" donde luego pondrás datos
ceros = np.zeros((3, 4))
print(f"   np.zeros((3,4)):\n{ceros}")

# np.ones: igual pero con unos
# Útil para crear biases o matrices de unos en redes neuronales
unos = np.ones((2, 5))
print(f"\n   np.ones((2,5)):\n{unos}")

# np.arange: como range() de Python pero devuelve array
# ¡La versión numpy de range()!
rango = np.arange(0, 20, 5)  # start, stop (exclusive), step
print(f"\n   np.arange(0, 20, 5) = {rango}")

# np.linspace: N puntos igualmente espaciados entre un rango
# MUY usado para crear ejes en gráficos y para evaluar funciones
lineal = np.linspace(0, 1, 5)  # 5 puntos entre 0 y 1
print(f"   np.linspace(0, 1, 5) = {lineal}")

# np.random: datos aleatorios (simulaciones, inicialización de pesos)
aleatorios = np.random.randn(4)  # 4 números de distribución normal (media=0, std=1)
print(f"   np.random.randn(4) = {aleatorios}")

# =====================================================================
# PASO 2: Indexación y slicing — extrayendo datos específicos
# =====================================================================
print("\n" + "=" * 72)
print("🎯 PASO 2: Indexación y slicing — cómo agarrar lo que necesitas")
print("=" * 72)
print("   En ML siempre estás extrayendo filas, columnas, o secciones de datos.")

# Nombres de columnas para referencia (en numpy usamos índices numéricos)
# Columna 0: Temperatura | 1: Humedad | 2: Presión | 3: Viento | 4: Lluvia
print(f"\n   Matriz Santiago:\n{santiago}")
print("\n   Columnas: [Temp, Humedad, Presión, Viento, Lluvia]")

# 2.1: Una celda específica — santiago[fila, columna]
print(f"\n📌 2.1 Una celda: santiago[2, 4] = {santiago[2, 4]}")
print(f"   → Día 3 (índice 2), Lluvia (col 4) = {santiago[2, 4]} mm")

# 2.2: Una fila completa — santiago[fila, :]
print(f"\n📌 2.2 Una fila (un día completo): santiago[1, :]")
print(f"   → Día 2: {santiago[1, :]}")
# El ":" significa "todas las columnas". Es como decir "dame todo de esta fila".

# 2.3: Una columna completa — santiago[:, columna]
print(f"\n📌 2.3 Una columna (un sensor todos los días): santiago[:, 0]")
print(f"   → Temperaturas todos los días: {santiago[:, 0]}")
# Esto es lo que más harás en ML: extraer una "feature" (característica)
# completa para analizarla o usarla como entrada del modelo.

# 2.4: Slicing — rangos de filas y columnas
print(f"\n📌 2.4 Slicing: santiago[1:4, 0:2]")
print(f"   → Días 2 a 4 (filas 1,2,3), Temperatura y Humedad (cols 0,1):")
print(f"   {santiago[1:4, 0:2]}")
# slicing funciona como en listas: [inicio:fin] con fin EXCLUSIVO (no incluye el último)

# 2.5: Slicing con paso
print(f"\n📌 2.5 Slicing con paso: santiago[::2, :]")
print(f"   → Días alternados (1,3,5,7):\n{santiago[::2, :]}")
# ::2 significa "cada 2 elementos". Muy útil para submuestrear datos grandes.

# =====================================================================
# PASO 3: ¡EL PODER REAL DE NUMPY! Operaciones vectorizadas
# =====================================================================
print("\n" + "=" * 72)
print("⚡ PASO 3: Operaciones vectorizadas — la magia de numpy")
print("=" * 72)
print("   En Python normal, para sumar 2 listas necesitas un for loop.")
print("   En numpy, haces lista1 + lista2 y se suma TODO a la vez.")
print("   Esto se llama VECTORIZACIÓN y es lo que hace a numpy RÁPIDO.")

# Creamos datos de otra ciudad para comparar
valparaiso = np.array([
    [18.2, 78, 1015, 22.5, 0.0],
    [19.5, 75, 1013, 25.1, 0.0],
    [16.8, 85, 1017, 18.4, 5.2],
    [17.9, 80, 1014, 20.7, 1.5],
    [20.1, 72, 1010, 28.3, 0.0],
    [18.8, 77, 1016, 23.6, 0.0],
    [17.2, 82, 1018, 19.9, 3.8],
])

print("\n📌 3.1 Sumar todas las lecturas de 2 ciudades: santiago + valparaiso")
suma_ciudades = santiago + valparaiso
print(f"   Resultado:\n{suma_ciudades}")
# ¡Sumó elemento por elemento! Sin un solo for loop.
# En Python normal necesitarías: [s[i]+v[i] for i in range(len(s))]

print("\n📌 3.2 Promedio de las 2 ciudades: (santiago + valparaiso) / 2")
promedio = (santiago + valparaiso) / 2
print(f"   Promedio temp día 1: {promedio[0,0]:.1f}°C")
print(f"   ({(santiago[0,0] + valparaiso[0,0]) / 2:.1f}°C)" + 
      " — mismo resultado, pero con todo el array a la vez")

# ---------------------------------------------------------------------
# 3.3: Otras operaciones aritméticas vectorizadas
# ---------------------------------------------------------------------
print("\n📌 3.3 Más operaciones vectorizadas (todas funcionan elemento a elemento):")
print(f"   Temperaturas Santiago:   {santiago[:, 0]}")
print(f"   Temperaturas Valparaíso: {valparaiso[:, 0]}")
print(f"   Diferencia (S - V):      {np.round(santiago[:, 0] - valparaiso[:, 0], 1)}")
print(f"   Producto (S × V):        {np.round(santiago[:, 0] * valparaiso[:, 0], 0).astype(int)}")
print(f"   Cuadrado de S:           {np.round(santiago[:, 0] ** 2, 1)}")
print(f"   Raíz cuadrada de S:      {np.round(np.sqrt(santiago[:, 0]), 2)}")

# np.sqrt, np.exp, np.sin, np.log... son "funciones universales" (ufuncs)
# que operan sobre arrays completos sin necesidad de loops.
# En ML usamos np.exp() para softmax, np.log() para cross-entropy, etc.

# ---------------------------------------------------------------------
# 3.4: COMPARACIÓN DE VELOCIDAD — numpy vs Python loop
# ---------------------------------------------------------------------
print("\n📌 3.4 ⏱️  Comparación de velocidad: numpy vs Python loop")
print("   Vamos a calcular la temperatura media de 3 ciudades")
print("   usando un loop de Python vs operación vectorizada de numpy.\n")

# Generamos datos más grandes para que se note la diferencia
# 1 MILLÓN de lecturas de temperatura (como un sensor que mide cada segundo)
np.random.seed(123)
datos_grandes = np.random.randn(1_000_000) * 10 + 25  # 1M temperaturas ~25°C

# --- Método 1: Python loop (LENTO) ---
inicio = time.time()
suma_loop = 0
for i in range(len(datos_grandes)):
    suma_loop += datos_grandes[i]
media_loop = suma_loop / len(datos_grandes)
tiempo_loop = time.time() - inicio
print(f"   🐌 Python loop:    {tiempo_loop:.4f} segundos  → media = {media_loop:.4f}°C")

# --- Método 2: numpy vectorizado (RÁPIDO) ---
inicio = time.time()
media_numpy = np.mean(datos_grandes)
tiempo_numpy = time.time() - inicio
print(f"   ⚡ numpy:          {tiempo_numpy:.4f} segundos  → media = {media_numpy:.4f}°C")

if tiempo_loop > 0:
    print(f"\n   🚀 numpy es {tiempo_loop/tiempo_numpy:.0f}x MÁS RÁPIDO que el loop de Python")
print("   ¡Y así con 1 millón de datos! Imagina con datasets de ML")
print("   que tienen millones de filas y cientos de columnas.")
print("   Por eso TODO el ecosistema ML usa numpy internamente.")

# =====================================================================
# PASO 4: Broadcasting — el superpoder de numpy
# =====================================================================
print("\n" + "=" * 72)
print("📡 PASO 4: Broadcasting — arrays de diferentes formas, misma operación")
print("=" * 72)
print("   Broadcasting permite operar arrays de distintas dimensiones.")
print("   numpy \"estira\" el array más pequeño para que encaje con el grande.")
print("   Es como si el array pequeño se repitiera automáticamente.")

# ---------------------------------------------------------------------
# 4.1: Escenario real — calibrar sensores
# ---------------------------------------------------------------------
# Todos los sensores tienen un sesgo (bias). Hay que restarles ese sesgo.
print("\n📌 4.1 Calibración de sensores (restar un sesgo a toda la matriz):")
# Imagina que los sensores de temperatura de Santiago tienen un sesgo de +0.5°C
sesgo_temperatura = 0.5
print(f"   Sesgo del sensor de temperatura: +{sesgo_temperatura}°C")
print(f"   Temperaturas raw (con sesgo):    {santiago[:, 0]}")
santiago_calibrado = santiago[:, 0] - sesgo_temperatura
print(f"   Temperaturas calibradas (sin sesgo): {santiago_calibrado}")
# ¡Restamos un ESCALAR (0.5) a un ARRAY!
# numpy "estiró" el 0.5 para que sea un array de 7 elementos, todos 0.5.
# Esto es broadcasting en acción.

# ---------------------------------------------------------------------
# 4.2: Broadcasting con arrays — convertir °C a °F
# ---------------------------------------------------------------------
print("\n📌 4.2 Convertir temperaturas de °C a °F (broadcasting con escalares):")
temp_c = santiago[:, 0]  # Temperaturas en Celsius
temp_f = temp_c * 9/5 + 32  # Fórmula: °F = °C × 9/5 + 32
print(f"   °C: {temp_c}")
print(f"   °F: {temp_f}")
# numpy aplicó 9/5 y +32 a CADA ELEMENTO del array.
# Sin broadcasting, tendrías que hacer un for loop.

# ---------------------------------------------------------------------
# 4.3: Broadcasting 2D — estandarizar datos (Z-score)
# ---------------------------------------------------------------------
print("\n📌 4.3 Estandarización (Z-score) — preparación de datos para ML:")
# El Z-score es una transformación MUY común en ML:
#   z = (x - media) / desviación_estándar
# Esto hace que los datos tengan media=0 y desviación=1.
# Es necesario cuando las features tienen diferentes escalas (ej: °C vs mm de lluvia).

medias = np.mean(santiago, axis=0)  # axis=0 = promedio por COLUMNA (por sensor)
desvios = np.std(santiago, axis=0)
print(f"   Medias por sensor (temp, hum, pres, viento, lluvia):")
print(f"   {medias}")
print(f"   Desviaciones: {desvios}")

# Broadcasting 2D: restamos un array 1D (medias) de un array 2D (santiago)
# numpy "estira" el array de 5 elementos para que calce con la matriz de 7×5
santiago_std = (santiago - medias) / desvios
print(f"\n   Santiago estandarizado (Z-score):")
print(f"   {santiago_std}")
print(f"   Media después de estandarizar: {np.mean(santiago_std, axis=0)}")
print(f"   (¿Ves? ahora cada columna tiene media ≈ 0 y desviación ≈ 1)")
# Este proceso se llama "feature scaling" y es OBLIGATORIO en muchos algoritmos ML.

# ---------------------------------------------------------------------
# 4.4: Broadcasting más complejo — agregar una dimensión
# ---------------------------------------------------------------------
print("\n📌 4.4 Broadcasting con 3 ciudades — expandir dimensiones:")
# Queremos calcular la temperatura promedio de los 3 ciudades para cada día
concepcion = np.array([
    [15.0, 80, 1014, 18.5, 1.2],
    [16.2, 78, 1012, 20.1, 0.8],
    [13.5, 88, 1016, 15.3, 8.5],
    [14.8, 83, 1013, 17.7, 3.2],
    [17.0, 75, 1011, 22.4, 0.0],
    [15.9, 79, 1015, 19.8, 0.0],
    [14.1, 85, 1017, 16.5, 5.8],
])

# Creamos un array 3D: 3 ciudades × 7 días × 5 sensores
# np.stack apila arrays a lo largo de un nuevo eje
todas_ciudades = np.stack([santiago, valparaiso, concepcion], axis=0)
print(f"   Shape del array 3D: {todas_ciudades.shape}")
print(f"   → {todas_ciudades.shape[0]} ciudades, {todas_ciudades.shape[1]} días, {todas_ciudades.shape[2]} sensores")
# ¡Tres dimensiones! En ML los datos pueden tener muchas dimensiones.
# Imágenes: (alto, ancho, canales), Lotes: (batch, alto, ancho, canales)

# Promedio de temperatura entre las 3 ciudades para cada día
prom_temp_ciudades = np.mean(todas_ciudades[:, :, 0], axis=0)
# axis=0 significa "promediar a lo largo de las ciudades"
print(f"\n   Temperatura promedio (3 ciudades) por día:")
print(f"   {prom_temp_ciudades}")
print(f"   Día más cálido: día {np.argmax(prom_temp_ciudades)+1}" +
      f" con {np.max(prom_temp_ciudades):.1f}°C")

# =====================================================================
# PASO 5: Filtrado con máscaras booleanas — el WHERE de numpy
# =====================================================================
print("\n" + "=" * 72)
print("🔍 PASO 5: Máscaras booleanas — filtrar datos como en SQL")
print("=" * 72)
print("   Usamos condiciones >, <, ==, != para obtener TRUE/FALSE en cada posición.")
print("   Luego usamos esa máscara para filtrar los datos.")

# ---------------------------------------------------------------------
# 5.1: ¿Qué días llovió en Santiago?
# ---------------------------------------------------------------------
lluvia = santiago[:, 4]  # Columna de lluvia (índice 4)
print(f"\n📌 5.1 ¿Qué días llovió en Santiago?")
print(f"   Lluvia por día (mm): {lluvia}")

# La comparación > devuelve un array de booleanos
# Esto es una "máscara" — cada posición es True o False
dias_con_lluvia = lluvia > 0
print(f"   Máscara (lluvia > 0): {dias_con_lluvia}")
# Cuando usas la máscara como índice, obtienes SOLO los elementos True
print(f"   Días con lluvia: {lluvia[dias_con_lluvia]} mm")
print(f"   Total de días con lluvia: {np.sum(dias_con_lluvia)} de 7")
print(f"   Promedio cuando llueve: {np.mean(lluvia[lluvia > 0]):.2f} mm")

# ---------------------------------------------------------------------
# 5.2: Combinar múltiples condiciones
# ---------------------------------------------------------------------
print("\n📌 5.2 Días con temperatura > 22°C Y humedad < 65%:")
temp = santiago[:, 0]
humedad = santiago[:, 1]
mascara = (temp > 22) & (humedad < 65)  # & = AND, | = OR, ~ = NOT
print(f"   Temperaturas:            {temp}")
print(f"   Humedades:               {humedad}")
print(f"   Máscara (temp>22 & hum<65): {mascara}")
print(f"   Días que cumplen: {np.where(mascara)[0] + 1}")  # +1 porque días empezaban en 1
# np.where() te da los ÍNDICES donde la condición es True. ¡Súper útil!

# ---------------------------------------------------------------------
# 5.3: np.where() para reemplazar valores condicionalmente
# ---------------------------------------------------------------------
print("\n📌 5.3 Clasificar temperatura con np.where():")
# np.where(condición, valor_si_true, valor_si_false)
clasificacion = np.where(temp > 22, "Cálido ☀️", "Fresco 🌥️")
print(f"   Temperaturas: {temp}")
print(f"   Clasificación: {clasificacion}")
# Mucho más elegante que un for loop con if-else!

# =====================================================================
# PASO 6: Estadísticas descriptivas con numpy
# =====================================================================
print("\n" + "=" * 72)
print("📊 PASO 6: Estadísticas descriptivas — entender los datos con numpy")
print("=" * 72)
print("   Antes de meter datos a un modelo ML, hay que entenderlos.")
print("   numpy tiene TODAS las funciones estadísticas que necesitas.")

# ---------------------------------------------------------------------
# 6.1: Estadísticas por ciudad (solo temperatura para comparar)
# ---------------------------------------------------------------------
print("\n📌 6.1 Comparación de temperaturas entre ciudades:")
ciudades = ["Santiago", "Valparaíso", "Concepción"]
temps = [santiago[:, 0], valparaiso[:, 0], concepcion[:, 0]]

for i, ciudad in enumerate(ciudades):
    t = temps[i]
    print(f"\n   {ciudad}:")
    print(f"      Mín: {np.min(t):.1f}°C  |  Máx: {np.max(t):.1f}°C")
    print(f"      Media: {np.mean(t):.1f}°C  |  Mediana: {np.median(t):.1f}°C")
    print(f"      Desv. Estándar: {np.std(t):.2f}°C  |  Rango: {np.ptp(t):.1f}°C")
    # La desviación estándar te dice cuán variables son los datos.
    # Un valor alto = mucha variación, bajo = datos estables.

# ---------------------------------------------------------------------
# 6.2: Percentiles — entendiendo la distribución
# ---------------------------------------------------------------------
print("\n📌 6.2 Percentiles de temperatura en Santiago:")
percentiles = np.percentile(santiago[:, 0], [0, 25, 50, 75, 100])
print(f"   Mín (0%):  {percentiles[0]:.1f}°C")
print(f"   Q1  (25%): {percentiles[1]:.1f}°C  ← 25% de los días están bajo esto")
print(f"   Med (50%): {percentiles[2]:.1f}°C  ← El valor del medio")
print(f"   Q3  (75%): {percentiles[3]:.1f}°C  ← 75% de los días están bajo esto")
print(f"   Máx (100%): {percentiles[4]:.1f}°C")
print(f"   Rango intercuartil (IQR): {percentiles[3] - percentiles[1]:.1f}°C")
# El IQR mide la dispersión de la mitad central de los datos.
# Si el IQR es pequeño, los datos son estables.

# ---------------------------------------------------------------------
# 6.3: Correlación entre variables — ¿están relacionadas?
# ---------------------------------------------------------------------
print("\n📌 6.3 Matriz de correlación — ¿qué variables se relacionan?")
# La correlación va de -1 a 1. Cerca de 1 = relación directa, -1 = inversa, 0 = nada.
# En ML usamos la correlación para ver qué features están relacionadas con el target.

# Extraemos todas las columnas de Santiago
temp = santiago[:, 0]
hum  = santiago[:, 1]
pres = santiago[:, 2]
vien = santiago[:, 3]
lluv = santiago[:, 4]

# np.corrcoef() calcula la matriz de correlación entre arrays
correlacion = np.corrcoef([temp, hum, pres, vien, lluv])
print(f"   Matriz de correlación (5×5):")
print(f"   {np.round(correlacion, 2)}")
print(f"   Interpretación:")
print(f"   - Diagonal = 1.0 (cada variable se correlaciona perfectamente consigo misma)")
print(f"   - Positivo → cuando una sube, la otra también")
print(f"   - Negativo → cuando una sube, la otra baja")
print(f"   - Cerca de 0 → no hay relación lineal")

# =====================================================================
# PASO 7: Cambiar formas — reshape y manipulación de arrays
# =====================================================================
print("\n" + "=" * 72)
print("🔄 PASO 7: Reshape — cambiar la forma de los datos")
print("=" * 72)
print("   En ML, constantemente cambiamos la forma de los datos.")
print("   Ej: convertir una imagen 2D a un vector 1D para la red neuronal.")

datos_1d = np.arange(12)
print(f"\n📌 7.1 Array 1D original: {datos_1d}")

# reshape: cambia la forma sin modificar los datos
# El número total de elementos debe mantenerte (12 = 3×4)
matriz_3x4 = datos_1d.reshape(3, 4)
print(f"\n📌 7.2 reshape(3, 4) — matriz 3×4:")
print(matriz_3x4)

# reshape(6, 2)
matriz_6x2 = datos_1d.reshape(6, 2)
print(f"\n📌 7.3 reshape(6, 2):")
print(matriz_6x2)

# flatten() — convierte cualquier matriz a 1D (muy usado para preparar inputs)
print(f"\n📌 7.4 flatten() — convertir matriz de Santiago a vector:")
vector_santiago = santiago.flatten()
print(f"   Vector 1D de {len(vector_santiago)} elementos (antes {santiago.shape}):")
print(f"   {vector_santiago}")
# flatten() es útil cuando tienes imágenes (2D) y las conviertes en vectores (1D)
# para alimentar una red neuronal densa.

# ---------------------------------------------------------------------
# 7.5: Transposición — intercambiar filas y columnas
# ---------------------------------------------------------------------
print(f"\n📌 7.5 Transposición .T — intercambiar filas y columnas:")
print(f"   Original: {santiago.shape} (días × sensores)")
print(f"   Transpuesto: {santiago.T.shape} (sensores × días)")
print(f"   Primera fila de la transpuesta (temperatura todos los días):")
print(f"   {santiago.T[0]}")
# La transpuesta es clave en álgebra lineal, la base de las redes neuronales.

# =====================================================================
# PASO 8: Aplicación práctica — preparación de datos para ML
# =====================================================================
print("\n" + "=" * 72)
print("🏗️  PASO 8: Aplicación práctica — preparando datos para un modelo ML")
print("=" * 72)
print("   Vamos a simular el pipeline real de preparación de datos:")
print("   1. Juntar datos de múltiples fuentes")
print("   2. Crear features derivadas (feature engineering)")
print("   3. Normalizar los datos")
print("   4. Separar en features (X) y target (y)")
print("   5. Crear train/test split")

# ---------------------------------------------------------------------
# 8.1: Feature engineering — crear nuevas características
# ---------------------------------------------------------------------
print("\n📌 8.1 Feature engineering — creando nuevas variables:")
# En ML, las mejores features a veces son combinaciones de las originales.

# Amplitud térmica (temp máxima - mínima del período)
# np.ptp = "peak to peak" = máximo - mínimo
amp_termica_stgo = np.ptp(santiago[:, 0])
amp_termica_valpo = np.ptp(valparaiso[:, 0])
amp_termica_conc = np.ptp(concepcion[:, 0])
print(f"   Amplitud térmica semanal:")
print(f"   Santiago:   {amp_termica_stgo:.1f}°C")
print(f"   Valparaíso: {amp_termica_valpo:.1f}°C  ← efecto moderador del mar")
print(f"   Concepción: {amp_termica_conc:.1f}°C  ← también costera")
# ¿Por qué Santiago tiene más amplitud térmica? Porque está en un valle rodeado
# de montañas. Valparaíso está en la costa, el mar modera la temperatura.

# Feature: índice de sensación térmica (simplificado)
# En días con viento y baja temperatura, la sensación es más fría
print(f"\n   Índice de sensación térmica (simplificado):")
print(f"   Fórmula: sensación = temp - (viento * 0.5)")
sensacion_stgo = santiago[:, 0] - (santiago[:, 3] * 0.5)
print(f"   Santiago: temp real = {santiago[:, 0]}")
print(f"             sensación = {np.round(sensacion_stgo, 1)}")
# Este tipo de transformaciones se llaman "feature engineering".
# En la vida real, los mejores ingenieros de ML crean features ingeniosas.

# ---------------------------------------------------------------------
# 8.2: Normalización Min-Max — escalar datos entre 0 y 1
# ---------------------------------------------------------------------
print("\n📌 8.2 Normalización Min-Max (escalar a [0, 1]):")
# Muchos algoritmos ML requieren que los datos estén en la misma escala.
# Min-Max: x_norm = (x - min) / (max - min)

temp_stgo = santiago[:, 0]
temp_norm = (temp_stgo - np.min(temp_stgo)) / (np.max(temp_stgo) - np.min(temp_stgo))
print(f"   Temperatura original:  {temp_stgo}")
print(f"   Temperatura normalizada: {np.round(temp_norm, 3)}")
print(f"   Ahora todos los valores están entre 0 y 1.")
print(f"   Método usado en: redes neuronales, KNN, SVM, etc.")

# ---------------------------------------------------------------------
# 8.3: Separar features (X) y target (y)
# ---------------------------------------------------------------------
print("\n📌 8.3 Preparación de X (features) y y (target):")
# En ML, X son las variables de entrada, y es lo que queremos predecir.
# Simulemos que queremos predecir la lluvia con las otras variables.
X = santiago[:, :4]  # Temperatura, humedad, presión, viento
y = santiago[:, 4]   # Lluvia
print(f"   X (features) shape: {X.shape}  → {X.shape[1]} variables para predecir")
print(f"   y (target) shape:   {y.shape}")
print(f"   X primeras filas:\n{X}")
print(f"   y (lluvia): {y}")

# ---------------------------------------------------------------------
# 8.4: Train/test split manual (¡en el futuro usarás scikit-learn!)
# ---------------------------------------------------------------------
print("\n📌 8.4 Split entrenamiento / prueba — manual con numpy:")
# En ML separamos datos en: entrenamiento (entrenar el modelo) y prueba (evaluarlo).
# Típicamente 80% train, 20% test.
n_total = len(X)
n_train = int(n_total * 0.8)
print(f"   Total muestras: {n_total}, Train: {n_train}, Test: {n_total - n_train}")

# Slicing para separar
X_train = X[:n_train]  # Primeras 5 filas (80%)
X_test = X[n_train:]   # Últimas 2 filas (20%)
y_train = y[:n_train]
y_test = y[n_train:]

print(f"   X_train shape: {X_train.shape}")
print(f"   X_test shape:  {X_test.shape}")
print(f"   y_train: {y_train}")
print(f"   y_test:  {y_test}")
# Este split manual es didáctico. En proyectos reales usamos
# train_test_split de scikit-learn que además mezcla aleatoriamente.

# =====================================================================
# VISUALIZACIONES — porque los datos hay que verlos para entenderlos
# =====================================================================
print("\n" + "=" * 72)
print("📈 VISUALIZACIONES — generando gráficos con matplotlib + numpy")
print("=" * 72)

# ---------------------------------------------------------------------
# Gráfico 1: Temperaturas diarias de las 3 ciudades
# ---------------------------------------------------------------------
print("\n   Generando gráfico 1: Temperaturas diarias...")
plt.figure(figsize=(12, 6))
dias = np.arange(1, 8)  # Días 1 a 7

plt.plot(dias, santiago[:, 0], 'o-', label='Santiago', linewidth=2, markersize=8)
plt.plot(dias, valparaiso[:, 0], 's-', label='Valparaíso', linewidth=2, markersize=8)
plt.plot(dias, concepcion[:, 0], '^-', label='Concepción', linewidth=2, markersize=8)

plt.title('Temperatura Diaria por Ciudad', fontsize=16, fontweight='bold')
plt.xlabel('Día de la Semana', fontsize=12)
plt.ylabel('Temperatura (°C)', fontsize=12)
plt.xticks(dias, ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'])
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Destacar la amplitud térmica de Santiago con una anotación
plt.annotate('Santiago: mayor\namplitud térmica\n(valle interior)',
             xy=(2, santiago[2, 0]), xytext=(3.5, 16),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=10, color='red', fontweight='bold')

plt.tight_layout()
plt.savefig('docs/temperaturas_diarias.png', dpi=150)
plt.close()
print("   ✅ docs/temperaturas_diarias.png")

# ---------------------------------------------------------------------
# Gráfico 2: Mapa de calor de correlación
# ---------------------------------------------------------------------
print("\n   Generando gráfico 2: Mapa de calor de correlación...")
fig, ax = plt.subplots(figsize=(8, 7))
nombres_vars = ['Temp', 'Humedad', 'Presión', 'Viento', 'Lluvia']
im = ax.imshow(correlacion, cmap='RdYlBu_r', vmin=-1, vmax=1)

# Añadir los valores en cada celda
for i in range(5):
    for j in range(5):
        ax.text(j, i, f'{correlacion[i, j]:.2f}', ha='center', va='center',
                fontsize=12, fontweight='bold',
                color='white' if abs(correlacion[i, j]) > 0.5 else 'black')

ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xticklabels(nombres_vars, fontsize=11)
ax.set_yticklabels(nombres_vars, fontsize=11)
ax.set_title('Matriz de Correlacion — Santiago', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8, label='Correlación (-1 a 1)')
plt.tight_layout()
plt.savefig('docs/matriz_correlacion.png', dpi=150)
plt.close()
print("   ✅ docs/matriz_correlacion.png")

# ---------------------------------------------------------------------
# Gráfico 3: Comparación de rendimiento numpy vs loop
# ---------------------------------------------------------------------
print("\n   Generando gráfico 3: Comparación de velocidad...")

# Medimos con diferentes tamaños para mostrar la tendencia
tamanos = [1_000, 10_000, 100_000, 1_000_000]
tiempos_loop = []
tiempos_numpy = []

for n in tamanos:
    data = np.random.randn(n) * 10 + 25
    
    # Loop de Python
    inicio = time.time()
    s = 0
    for i in range(len(data)):
        s += data[i]
    media_loop = s / len(data)
    tiempos_loop.append(time.time() - inicio)
    
    # numpy
    inicio = time.time()
    media_np = np.mean(data)
    tiempos_numpy.append(time.time() - inicio)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico de barras
x = np.arange(len(tamanos))
ancho = 0.35
ax1.bar(x - ancho/2, tiempos_loop, ancho, label='Python Loop', color='#e74c3c', alpha=0.8)
ax1.bar(x + ancho/2, tiempos_numpy, ancho, label='numpy', color='#3498db', alpha=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels([f'{n:,}' for n in tamanos])
ax1.set_xlabel('Tamaño del array')
ax1.set_ylabel('Tiempo (segundos)')
ax1.set_title('Tiempo de Ejecución', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gráfico de aceleración (speedup)
speedups = [l/n for l, n in zip(tiempos_loop, tiempos_numpy)]
ax2.bar(x, speedups, color='#2ecc71', alpha=0.8, width=0.6)
ax2.set_xticks(x)
ax2.set_xticklabels([f'{n:,}' for n in tamanos])
ax2.set_xlabel('Tamaño del array')
ax2.set_ylabel('Aceleración (x veces más rápido)')
ax2.set_title('numpy vs Loop — Factor de Aceleracion', fontweight='bold')
for i, v in enumerate(speedups):
    ax2.text(i, v + 0.5, f'{v:.0f}x', ha='center', fontweight='bold', fontsize=11)
ax2.grid(True, alpha=0.3)

plt.suptitle('numpy es MASIVAMENTE mas rapido que los loops de Python',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('docs/comparacion_velocidad.png', dpi=150)
plt.close()
print("   ✅ docs/comparacion_velocidad.png")

# ---------------------------------------------------------------------
# Gráfico 4: Distribución de temperaturas (histograma)
# ---------------------------------------------------------------------
print("\n   Generando gráfico 4: Distribución de temperaturas...")
fig, ax = plt.subplots(figsize=(10, 6))

# Juntamos todas las temperaturas
todas_temps = np.concatenate([santiago[:, 0], valparaiso[:, 0], concepcion[:, 0]])
colores = ['#e74c3c', '#3498db', '#2ecc71']
labels_ciudades = ['Santiago', 'Valparaíso', 'Concepción']

for i, (ciudad, color) in enumerate(zip([santiago[:, 0], valparaiso[:, 0], concepcion[:, 0]], colores)):
    ax.hist(ciudad, bins=5, alpha=0.6, label=labels_ciudades[i], color=color, edgecolor='white')

ax.axvline(np.mean(todas_temps), color='gray', linestyle='--', linewidth=2,
           label=f'Media global: {np.mean(todas_temps):.1f}°C')
ax.set_xlabel('Temperatura (°C)', fontsize=12)
ax.set_ylabel('Frecuencia (días)', fontsize=12)
ax.set_title('Distribucion de Temperaturas por Ciudad', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('docs/distribucion_temperaturas.png', dpi=150)
plt.close()
print("   ✅ docs/distribucion_temperaturas.png")

# ---------------------------------------------------------------------
# Gráfico 5: Todas las variables en Santiago (radar plot-like subplots)
# ---------------------------------------------------------------------
print("\n   Generando gráfico 5: Variables meteorológicas (subplots)...")
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()

variables = [
    (santiago[:, 0], 'Temperatura (C)', '#e74c3c'),
    (santiago[:, 1], 'Humedad (%)', '#3498db'),
    (santiago[:, 2], 'Presion (hPa)', '#2ecc71'),
    (santiago[:, 3], 'Viento (km/h)', '#f39c12'),
    (santiago[:, 4], 'Lluvia (mm)', '#9b59b6'),
]

for i, (data, label, color) in enumerate(variables):
    axes[i].plot(dias, data, 'o-', color=color, linewidth=2, markersize=8)
    axes[i].set_title(label, fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Día', fontsize=10)
    axes[i].set_xticks(dias)
    axes[i].set_xticklabels(['L', 'M', 'M', 'J', 'V', 'S', 'D'])
    axes[i].grid(True, alpha=0.3)
    # Línea de la media
    axes[i].axhline(np.mean(data), color='gray', linestyle='--', alpha=0.7,
                    label=f'media={np.mean(data):.1f}')
    axes[i].legend(fontsize=9)

# El sexto subplot vacío, poner texto
axes[5].text(0.5, 0.5, 'numpy permite\nanalizar TODAS\nlas variables\nde forma\nvectorizada',
             ha='center', va='center', fontsize=14, fontweight='bold', color='gray',
             transform=axes[5].transAxes)
axes[5].set_frame_on(False)
axes[5].set_xticks([])
axes[5].set_yticks([])

plt.suptitle('Variables Meteorologicas — Santiago (7 dias)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('docs/variables_meteorologicas.png', dpi=150)
plt.close()
print("   ✅ docs/variables_meteorologicas.png")

# =====================================================================
# 🧠 RESUMEN FINAL
# =====================================================================
print("\n" + "=" * 72)
print("🧠 RESUMEN: Hoy aprendiste numpy para Machine Learning")
print("=" * 72)
print("""
✅ Crear arrays: desde listas, np.zeros, np.ones, np.arange, np.linspace, np.random
✅ Atributos: shape, ndim, size, dtype, nbytes (siempre revisa shape primero)
✅ Indexación y slicing: [fila, columna], : para "todo", ::2 para pasos
✅ ⚡ OPERACIONES VECTORIZADAS: sumar, restar, multiplicar arrays enteros SIN LOOPS
✅ Broadcasting: operar arrays de diferentes tamaños (escalar + array, array 1D + 2D)
✅ Máscaras booleanas: array > 0, condiciones combinadas con & y |
✅ Estadísticas: mean, std, min, max, median, percentile, corrcoef
✅ Reshape y flatten: cambiar dimensiones de los datos
✅ Pipeline ML: feature engineering, normalización, train/test split
✅ 🚀 numpy es 50-100x más rápido que Python loops para datos grandes

🔥 En el próximo proyecto aplicarás estas habilidades con visualizaciones
   más avanzadas usando matplotlib y seaborn.

💡 RECUERDA: Todo ML en Python se basa en numpy.
   - pandas usa numpy internamente
   - scikit-learn recibe arrays de numpy
   - TensorFlow/PyTorch convierten a tensores (arrays con esteroides)
   - ¡Dominar numpy es dominar la base de todo!
""")
