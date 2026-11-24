# Decisiones del proyecto: predicción de `price` (Práctica 1, Ingeniería del Conocimiento)

> Bitácora viva. Se actualiza al cerrar cada fase. Si un dato se contradice con una fase posterior, manda la entrada más reciente.

## 1. Contexto fijo (del enunciado)
- Problema: regresión. Predecir `price` (euros) de productos de e-commerce.
- Datos: train-1.csv (con `price`) y test-1.csv (sin `price`, es el que hay que predecir).
- Métrica de evaluación: RMSE sobre el test oculto, escala original (euros). Menor es mejor.
- Umbral: un RMSE de 57 o inferior equivale a un 5. La nota máxima depende de los resultados de todos los grupos.
- Reparto de nota: 80 % rendimiento, 20 % vídeo (máx. 10 min, YouTube "oculto").
- Entrega: `submission_NombreAlumno.csv` (columnas `id`, `prediction`; una fila por id de test) + notebook .ipynb con portada (nombre y URL del vídeo).
- Comprobación de puntuación: https://ic2526p1.streamlit.app/ (no usarla para ajustar el modelo, solo como confirmación final).

## 2. Hechos verificados sobre los datos (fase 0, revisión inicial; confirmados y ampliados en la Fase 1)
- train: 8.000 filas x 16 columnas. test: 2.000 filas x 15 columnas (las mismas, sin `price`).
- `id` es único, sin solapamiento entre train y test. No es predictora.
- Sin filas duplicadas en train.
- Variables categóricas (7): category (6), brand (6), brand_tier (3: Budget/Media/Premium), material (5), season (4), channel (2: Online/Retail), country (4: ES/FR/IT/PT). Las categorías de test son las mismas que las de train.
- Variables numéricas (7): weight_kg, rating, reviews, shipping_cost, margin, discount, is_promo (0/1).
- Faltantes: solo `material` (252 filas = 3,15 % en train; 58 = 2,9 % en test). Sin otras columnas con faltantes.
- Rangos: weight_kg 0,05–21,5 (mediana 0,80); rating 2,15–5; reviews 3–1.578; shipping_cost 0–7,97; margin 0,05–0,58; discount 0–0,45.
- Objetivo `price`: media 158,0; mediana 139,8; desviación 78,7; mín 29,96; máx 1.238,25; asimetría 2,88 (cola larga a la derecha).
- Correlación lineal con `price`: weight_kg 0,57; shipping_cost 0,37; margin 0,24; discount -0,16; is_promo -0,14; reviews 0,11; rating 0,10.

## 3. Referencias iniciales (no definitivas)
Medidas con 5-fold CV aleatoria (semilla 42), imputación por moda + one-hot de las categóricas, sin ajuste de hiperparámetros. Sirven solo para orientarse.

| Modelo | RMSE CV | MAE CV |
|---|---|---|
| Predecir la media | 78,66 | n/d |
| Regresión lineal | 54,25 | 34,81 |
| Regresión lineal con objetivo log (sin corrección de sesgo) | 112,54 | 36,46 |
| HistGradientBoosting | 54,84 | 35,49 |
| HistGradientBoosting con objetivo log (sin corrección de sesgo) | 54,40 | 34,25 |

Lecturas provisionales:
- Ya hay modelos por debajo de 57 en CV, pero no hay mucho margen y la CV puede no coincidir con el test oculto.
- Invertir el log sin corregir el sesgo puede destrozar el RMSE (caso de la regresión lineal). Hay que tratarlo bien si se usa.
- El RMSE (≈ 54) es muy superior al MAE (≈ 35): la cola de precios altos domina el error. Es el punto clave a trabajar. (Cifras de la Fase 0, no recalculadas en la Fase 1; se recalcularán en la Fase 3.)

## 4. Decisiones por fase

### Fase 1: EDA (CERRADA)
**Entregables de la fase:** `Practica1_Notebook_acumulado.ipynb` (Parte 1, 129 comprobaciones de cifras: 129 OK), `memoria_fase1_EDA.md` (versión corregida), figuras 01–08 en `fig_eda/`.

**Entorno.** Colab del alumno: pandas 2.2.3, numpy 2.4.6, scikit-learn 1.6.1, scipy 1.16.3, matplotlib 3.10.0. Entorno de Claude: pandas 3.0.6, numpy 2.5.4, scikit-learn 1.9.1, scipy 1.18.1, matplotlib 3.11.2. Los resultados coinciden salvo en el AUC adversario (tercer decimal).

**Hechos medidos**
- Precio: asimetría 2,88; con log 0,37; curtosis 19,29. Percentiles: p50 139,8; p95 298,7; p99 427,9.
- Cola (cada fila es un recuento independiente; % de la varianza total): price>300: 392 filas, 54,3 %; >400: 108 filas (1,35 %), 35,1 %; >500: 43 filas, 24,7 %; >700: 10 filas, 13,1 %.
- Cola >400: 37 % Electrónica, 23 % Hogar, 50 % Premium; mediana weight_kg 2,94 (global 0,80). Con weight_kg>3 (396 filas) precio medio 300,0.
- Ruido proporcional (hipótesis): precio medio por decil de peso 109,9 → 261,4; coeficiente de variación (std/media) entre 0,37 y 0,43 en los 10 deciles. No es prueba (la std del decil mezcla ruido y efecto del peso).
- Correlación Pearson con price: weight_kg 0,57; shipping_cost 0,37; margin 0,24; discount -0,16; is_promo -0,14; reviews 0,11; rating 0,10. Entre predictoras solo destacan weight_kg–shipping_cost (0,28) y discount–is_promo (0,80).
- weight_kg: asimetría 4,25 (0,00 con log). El log NO mejora la correlación lineal con price: 0,569 lineal; 0,523 log-peso; 0,536 log-precio; 0,557 log-log.
- Categóricas con señal (ANOVA y Kruskal p<1e-30): brand_tier (Premium 175,2 vs Budget 142,9), brand (142,3 a 176,4), category (Electrónica 184,9 vs Belleza 138,2). Débiles: season (p=0,0014), channel (p=0,018). Sin señal en precio medio: country (p=0,34), material (p=0,085).
- Faltante de material: precio medio 152,7 (n=252) vs 158,2 (n=7.748); Welch p=0,18, IC95 % de la diferencia [-13,6; +2,6]; permutaciones (10.000, semilla 42) p=0,28. Sin dependencia de category, brand_tier, channel, country, season (χ² p entre 0,14 y 0,95; % de faltantes entre 2,4 % y 3,8 %). Poca potencia (252 filas): no es prueba de ausencia de efecto.
- Pendiente log-log precio~weight_kg: 0,27–0,32 por categoría, 0,29–0,31 por tier: sin indicio de interacciones con el peso.
- Redundancias exactas: brand_tier = f(brand) (A,B Premium; C,D Media; E,F Budget); is_promo = (discount > 0,18).
- Truncados (train/test): rating==5: 288/70; margin==0,05: 20/0; discount==0: 79/7; shipping_cost==0: 25/12.
- Train vs test: numéricas muy parecidas (solo discount KS p=0,029, por más ceros en train: 1,0 % vs 0,35 %). Sin categorías nuevas. Desplazamiento leve en category y channel (Online 70,2 % → 75,8 %; Electrónica 20,5 % → 23,1 %; Belleza 9,8 % → 12,8 %). Test con peso máximo 12,78 (train 21,51).
- Validación adversaria (HistGradientBoosting, 5 folds, 3 semillas, sin id): AUC medio ≈ 0,52 (0,52 ± 0,01 según semilla y versión de sklearn); nulo con etiquetas permutadas ≈ 0,49–0,51 (3 permutaciones). Variable más útil para separar: category (importancia por permutación, orientativa).
- id sin deriva (corr con price -0,008). id no es predictora.

**Decisiones**
- Validación: KFold aleatorio 5x3, random_state=42, congelado en SPLITS (mismas particiones para todas las pruebas). Aparte, una CV de control con random_state=2026, solo para confirmar, no para elegir.
- No se eliminan outliers del train (el test tendrá casos similares).
- No se eliminan filas con faltantes de `material`.
- `id` no es predictora.
- Tratamiento de la cola = punto clave: comparar en CV objetivo directo, log con corrección de sesgo y pérdidas robustas; métricas siempre en euros.
- Se mantienen todas las variables como candidatas; ninguna se elimina por el EDA.

**Hipótesis a medir en CV (NO decididas)**
- Fase 2: tratamiento de `material` (categoría "Desconocido" vs imputación por moda vs excluir la variable); con/sin `brand_tier`; con/sin `is_promo`; con/sin `country`; codificación (one-hot vs categóricas nativas); escalado para modelos lineales.
- Fase 4: log de weight_kg y reviews; shipping_cost/weight_kg; interacciones (categoría x brand_tier).
- Opcional: folds estratificados por tramos de precio (para repartir la cola).

### Fase 2: Preprocesado (CERRADA)
**Entregables:** Fase2_Preprocesado.ipynb (Parte 2), memoria_fase2_preprocesado.md, figuras 09-10 (fig_prep/).
**Entorno de referencia:** Colab del alumno (pandas 2.2.3, numpy 2.4.6, scikit-learn 1.6.1). HGB difiere ~0,1 € en otras versiones de sklearn (en el entorno de Claude, 1.9.1); el lineal coincide. F2.2 ejecutado dos veces en Colab: idéntico.
**Decisiones**
- Validación: KFold aleatorio 5x3, random_state=42, congelado en SPLITS. CV de control con random_state=2026, solo para confirmar, no para elegir.
- Conjunto de trabajo (punto de partida): sin material, brand_tier, is_promo ni country (empates; se elige lo simple). Quedan category, brand, season, channel + 6 numéricas. Reabrir en fase 5 con hiperparámetros ajustados (HGB reducido: +0,073 y +0,058 en las dos semillas; pista, no evidencia).
- Codificación: one-hot (nativas HGB: -0,267 ± 0,434, 9/15; empate).
- Escalado: estandarizar solo en la rama lineal (medido: sin efecto). Árboles sin escalar.
- Objetivo: directo en el lineal. HGB: log + Duan es CANDIDATO: OOF -0,700 ± 0,668 (13/15) supera la regla; residuos de train -0,639 ± 0,734 (12/15) empate; en control, train -0,606 ± 0,947 (11/15) empate; OOF sin medir en control.
**Descartado**
- Folds estratificados: bajan la std del RMSE (-29,7 % lineal, -23,4 % HGB) pero la de la diferencia lineal-HGB solo -2,3 % (umbral 10 %).
- Pérdidas robustas: Huber lineal +0,460 ± 0,386; HGB absolute_error +0,374 ± 1,310; GBR Huber +0,353 ± 0,789 (mejoran ≤300 €, empeoran >300 €).
- Log en el lineal con weight_kg sin transformar: RMSE ≈ 100; 6/15 folds >100 € (productos ≥18 kg, predicciones hasta 7.948 €).
**Medido (CV congelada, media ± std de 15 folds)**
- Lineal ref 54,063 ± 5,034 (MAE 34,803; R² 0,5256). HGB ref 54,625 ± 4,591 (MAE 35,431; R² 0,5150).
- RMSE ≤300 / >300 €: lineal 38,802 / 173,403; HGB 40,999 / 166,923. El tramo >300 € aporta 49,8 % / 45,3 % del error cuadrático.
- brand_tier y escalado en el lineal (mínimos cuadrados): predicciones idénticas (redundancia exacta).
- Conjunto reducido frente a ref: lineal -0,009 ± 0,060; HGB +0,073 ± 0,391 (control: +0,018 ± 0,073 y +0,058 ± 0,376). Empate.
**Pendiente:** log(weight_kg) para que el log funcione en el lineal (fase 4); medir log + Duan OOF con otra semilla (fase 5); reabrir conjunto completo y categóricas nativas con hiperparámetros ajustados (fase 5).

### Fase 3: Línea base y comparación de modelos
-

### Fase 4: Ingeniería de variables
-

### Fase 5: Ajuste de hiperparámetros y conjunto final
-

### Fase 6: Submission y validación
-

### Fase 7: Notebook final y guion del vídeo
-

## 5. Descartado (con motivo)
- Eliminar outliers del train: el test tendrá casos parecidos; solo se reconsideraría si la CV demuestra mejora.
- Eliminar filas con `material` faltante: el test también las tiene.
- Usar `id` como predictora: es un identificador sin señal (corr -0,008).
- "El log mejora la relación peso–precio": descartado, el log no mejora la correlación lineal (0,569 → 0,557 en el mejor caso). Solo simetriza el peso (asimetría 4,25 → 0,00) y es útil en el eje de los gráficos.
- Remuestreo o pesos de clase (SMOTE, etc.): no aplican, es un problema de regresión.

## 6. Pendiente / ideas por probar
- Reforzar la validación adversaria: más permutaciones (≈ 20) y un segundo clasificador (Random Forest o regresión logística).
- Cuantificar el efecto del desplazamiento de `category` y `channel` sobre el RMSE (por ejemplo, con RMSE por categoría en CV).
- Mirar el error por tramo de precio en la Fase 3 (comprobar que la cola domina el RMSE).
- Recalcular en la Fase 3 el RMSE ≈ 54 y el MAE ≈ 35 de la Fase 0.
- Folds estratificados por tramos de precio (hipótesis).
- Comparar objetivo directo, log con corrección de sesgo y pérdidas robustas.
- Ratios e interacciones (shipping_cost/weight_kg, categoría x brand_tier) y transformaciones log de weight_kg y reviews (como hipótesis).
- Nombre del alumno y URL del vídeo para la portada y el submission (preguntar antes de generar el archivo final).
