# 🎯 Operaciones Vectorizadas con numpy — Sensores Meteorológicos

**Nivel:** 1 — Fundamentos Python para ML  
**Skill:** numpy (arrays, operaciones vectorizadas, broadcasting)  
**Datos:** Simulados (basados en climatología real de ciudades chilenas)

---

## 📖 Qué vas a aprender

**numpy** es la librería fundamental de Machine Learning en Python. Piensa en ella como "listas de Python con superpoderes matemáticos". Todo el ecosistema ML (pandas, scikit-learn, TensorFlow, PyTorch) usa numpy por debajo.

En este proyecto aprenderás a:

### 1. Crear y manipular arrays
- Desde listas de Python con `np.array()`
- Arrays especializados: `np.zeros()`, `np.ones()`, `np.arange()`
- Datos aleatorios con `np.random.randn()`
- Atributos clave: `shape`, `dtype`, `ndim`, `size`

### 2. Indexación y slicing
- Acceder a elementos individuales: `array[fila, columna]`
- Extraer filas completas: `array[fila, :]`
- Extraer columnas completas: `array[:, columna]` ← ¡esto es lo que más harás en ML!
- Rangos y pasos: `array[inicio:fin:paso]`

### 3. ⚡ Operaciones vectorizadas (el corazón de numpy)
- Sumar, restar, multiplicar arrays enteros **sin usar un solo for loop**
- Esto hace que numpy sea **10-100x más rápido** que Python puro
- Las "ufuncs" (funciones universales): `np.sqrt()`, `np.exp()`, `np.log()`

### 4. 📡 Broadcasting
- Operar arrays de diferentes dimensiones
- Ejemplo real: restar la media de cada columna a toda una matriz (estandarización)
- Convertir °C a °F con una sola línea

### 5. 🔍 Máscaras booleanas
- Filtrar datos con condiciones: `array[array > 0]`
- Combinar condiciones con `&` (AND) y `|` (OR)
- `np.where()` para clasificar valores

### 6. 📊 Estadísticas y preparación de datos para ML
- `np.mean()`, `np.std()`, `np.percentile()`, `np.corrcoef()`
- Normalización Min-Max
- Feature engineering (crear nuevas variables)
- Train/test split manual

### ¿Por qué sensores meteorológicos?

Los datos de sensores son un caso de uso REAL donde numpy brilla:
- ✅ Múltiples ciudades → arrays 2D y 3D
- ✅ Calibración de sensores → broadcasting
- ✅ Detección de anomalías → máscaras booleanas
- ✅ Feature engineering → combinación vectorizada de variables
- ✅ Muchos datos → la velocidad de numpy importa

---

## 🧪 Código paso a paso

### Paso 1: Crear arrays desde listas
```python
import numpy as np

# Convertir una lista a array numpy
temperaturas = np.array([22.5, 24.1, 19.8, 21.3, 25.7, 23.0, 20.4])
```

**¿Por qué es importante?** Cuando cargues datos en ML, todo lo que no sea numpy se convierte a numpy. Incluso pandas DataFrames tienen un atributo `.values` que devuelve un array numpy.

### Paso 2: Crear una matriz de datos (7 días × 5 sensores)
```python
santiago = np.array([
    [22.5, 65, 1013, 12.1, 0.0],  # Día 1: temp, humedad, presión, viento, lluvia
    [24.1, 58, 1011, 15.3, 0.0],  # Día 2
    ...
])
print(santiago.shape)  # (7, 5) → 7 filas, 5 columnas
```

**Concepto clave:** En ML, la convención es:
- **Filas** = muestras (observaciones, ejemplos)
- **Columnas** = características (features, variables)

### Paso 3: Operaciones vectorizadas — la magia
```python
# Sin for loops. Sin list comprehensions. Una línea.
diferencia = santiago[:, 0] - valparaiso[:, 0]  # Diferencia de temperatura

# vs Python loop:
# for i in range(len(santiago)):
#     diferencia[i] = santiago[i, 0] - valparaiso[i, 0]
```

### Paso 4: Broadcasting — calibrar sensores
```python
# Restar un sesgo de +0.5°C a TODAS las temperaturas
santiago_calibrado = santiago[:, 0] - 0.5
# numpy "estira" el 0.5 para restarlo de cada elemento
```

### Paso 5: Máscaras booleanas — ¿qué días llovió?
```python
lluvia = santiago[:, 4]          # Columna de lluvia
dias_con_lluvia = lluvia > 0     # Máscara: [False, False, True, True, False, False, True]
lluvia_real = lluvia[dias_con_lluvia]  # Solo los días con lluvia
```

### Paso 6: Feature engineering para ML
```python
# Crear sensación térmica: temp - (viento × 0.5)
sensacion = santiago[:, 0] - (santiago[:, 3] * 0.5)

# Normalización Min-Max
temp_norm = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
```

---

## 📊 Resultados

### Comparación de velocidad
| Tamaño | Python Loop | numpy | Aceleración |
|--------|-------------|-------|-------------|
| 1.000 | 0.0004s | 0.0001s | ~4x |
| 10.000 | 0.003s | 0.0001s | ~30x |
| 100.000 | 0.03s | 0.0003s | ~100x |
| 1.000.000 | 0.3s | 0.002s | ~150x |

**Conclusión:** numpy es entre 4x y 150x más rápido según el tamaño de los datos. Y la diferencia crece con datasets más grandes.

### Insights de los datos meteorológicos
- **Santiago** tiene la mayor amplitud térmica (valle interior), mientras Valparaíso y Concepción (costeras) son más estables
- La **temperatura y humedad** tienden a correlacionarse negativamente (días cálidos = menos humedad)
- La **presión baja** generalmente precede a días lluviosos

### Visualizaciones generadas
- `docs/temperaturas_diarias.png` — Evolución de temperatura por ciudad
- `docs/matriz_correlacion.png` — Mapa de calor de correlaciones
- `docs/comparacion_velocidad.png` — numpy vs Python loop (gráfico de barras + speedup)
- `docs/distribucion_temperaturas.png` — Histograma por ciudad
- `docs/variables_meteorologicas.png` — Subplots de todas las variables

---

## 💡 Conceptos clave

- **Array numpy**: estructura de datos multidimensional, homogénea (todos los mismos tipo). Más eficiente que listas de Python.
- **Vectorización**: aplicar una operación a todo un array sin escribir loops explícitos. El secreto de la velocidad de numpy.
- **Broadcasting**: numpy "estira" automáticamente arrays de diferentes formas para que puedan operarse. Ej: array 2D + array 1D funciona sin reshape manual.
- **Máscara booleana**: array de True/False que se usa como índice para filtrar datos. Equivalente al WHERE de SQL.
- **Shape**: las dimensiones de un array. Siempre revisa `.shape` antes de operar. Error de shape = error más común en ML.
- **Estandarización (Z-score)**: transformar datos para que tengan media=0 y desviación=1. Necesario en muchos algoritmos ML.
- **Normalización Min-Max**: escalar datos al rango [0, 1]. Útil para redes neuronales.
- **Feature engineering**: crear nuevas variables a partir de las existentes para mejorar modelos ML.

---

## 🚀 Siguiente paso

El próximo proyecto será **matplotlib y seaborn** — visualización avanzada de datos. Aprenderás a:

- Hacer gráficos de publicación con matplotlib
- Descubrir patrones visualmente con seaborn
- Personalizar colores, estilos, anotaciones
- Crear subplots complejos
- Visualizar correlaciones con heatmaps

Después de dominar las visualizaciones, entrarás al **Nivel 2: Análisis Exploratorio de Datos (EDA)** con datasets reales.

---

## 🛠️ Cómo ejecutar

```bash
# 1. Crear entorno virtual
uv venv
source .venv/bin/activate

# 2. Instalar dependencias
uv pip install -r requirements.txt

# 3. Ejecutar
python project.py

# 4. Ver resultados
ls docs/  # Visualizaciones en PNG
```

---

*"numpy es a ML lo que los números son a las matemáticas: la base de todo."*
