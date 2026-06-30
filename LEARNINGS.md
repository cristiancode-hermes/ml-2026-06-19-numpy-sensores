# Learnings — 2026-06-19

## Qué se hizo
Proyecto de numpy: operaciones vectorizadas con datos de sensores meteorológicos.
Se crearon arrays 1D, 2D y 3D representando 7 días de lecturas de 5 sensores
en 3 ciudades chilenas (Santiago, Valparaíso, Concepción).

## Conceptos nuevos
- [x] Crear arrays con np.array(), np.zeros(), np.ones(), np.arange(), np.linspace()
- [x] Atributos: shape, ndim, size, dtype, nbytes
- [x] Indexación: array[fila, columna], slicing [inicio:fin:paso]
- [x] ⚡ Operaciones vectorizadas: suma, resta, multiplicación sin loops
- [x] 📡 Broadcasting: operar arrays de diferentes dimensiones
- [x] 🔍 Máscaras booleanas y np.where()
- [x] Estadísticas: mean, std, min, max, median, percentile, corrcoef
- [x] Reshape y flatten para cambiar dimensiones
- [x] Comparación de velocidad: numpy vs Python loops (50-150x más rápido)
- [x] Feature engineering básico y normalización Min-Max
- [x] Separación train/test manual con slicing

## Dificultades encontradas
- Ninguna en la ejecución del código (todo funcionó a la primera).
- Nota pedagógica: el broadcasting puede ser confuso al principio, se explicó
  con 3 ejemplos progresivos (escalar → 1D → 2D).

## Notas para el próximo proyecto
- El próximo skill pendiente en Nivel 1 es matplotlib: plots básicos
- Se podría combinar matplotlib con lo aprendido de numpy para visualizar
  los mismos datos de sensores meteorológicos
- Alternativa: proyecto de "Data cleaning" (nulos y duplicados) usando
  numpy + pandas

## Visualizaciones generadas
- docs/temperaturas_diarias.png
- docs/matriz_correlacion.png
- docs/comparacion_velocidad.png
- docs/distribucion_temperaturas.png
- docs/variables_meteorologicas.png
