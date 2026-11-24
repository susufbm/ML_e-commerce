# ML_e-commerce

Práctica 1 de Ingeniería del Conocimiento (UCAM, curso 2026-27): modelo de regresión para predecir el precio (`price`, en euros) de productos de un e-commerce.

## Contenido

| Ruta | Qué es |
|---|---|
| `data/train-1.csv` | 8.000 filas con `price` (entrenamiento) |
| `data/test-1.csv` | 2.000 filas sin `price` (conjunto a predecir) |
| `notebooks/Fase1_EDA_Colab.ipynb` | Fase 1: análisis exploratorio |
| `notebooks/Practica1_Parte2_Preprocesado.ipynb` | Fase 2: esquema de validación y preprocesado |
| `decisiones.md` | Bitácora: lo decidido, lo descartado y lo medido en cada fase |
| `contexto.md` | Instrucciones y reglas metodológicas que se le dan al asistente de IA (Claude) al empezar cada chat |
| `docs/enunciado_practica1.pdf` | Enunciado de la práctica |

## Evaluación

- 80 % rendimiento: RMSE en euros sobre el test oculto. Un RMSE de 57 o inferior equivale a un 5; la nota máxima depende de los resultados de todos los grupos.
- 20 % vídeo explicativo (máximo 10 minutos).

## Estado

| Fase | Estado |
|---|---|
| 1. EDA | Cerrada |
| 2. Preprocesado | Cerrada |
| 3. Línea base y comparación de modelos | Pendiente |
| 4. Ingeniería de variables | Pendiente |
| 5. Ajuste de hiperparámetros y conjunto final | Pendiente |
| 6. Submission y validación | Pendiente |
| 7. Notebook final y guion del vídeo | Pendiente |

Referencia actual en validación cruzada (5 folds x 3 repeticiones, semilla 42): regresión lineal 54,06 ± 5,03 y HistGradientBoosting 54,63 ± 4,59 de RMSE.

## Cómo trabajar

1. Antes de empezar, leer `decisiones.md`: no se repite nada que ya esté decidido o descartado ahí.
2. Si se trabaja con un asistente de IA, pegarle `contexto.md` como instrucciones y adjuntarle `decisiones.md`, para que todos partamos de las mismas reglas.
3. Los notebooks están pensados para Google Colab: abrirlos, subir los dos CSV de `data/` cuando lo pidan y ejecutar de arriba abajo.
4. Todas las pruebas usan las mismas particiones (KFold 5x3, `random_state=42`). Una mejora solo cuenta si supera la desviación entre folds.
5. Al cerrar una fase, actualizar `decisiones.md` en el mismo commit que el notebook.
6. Cada cambio en una rama propia y con pull request, para no pisar el trabajo de otro.
