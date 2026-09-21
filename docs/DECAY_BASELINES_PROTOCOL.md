# Protocolo congelado: ¿hace falta entrenar el encoder con JEPA?

Estado: **definido antes de implementar o ejecutar los baselines**. Este
documento fija la comparación y su interpretación. No contiene resultados.

## Pregunta

El experimento continuo mostró que el predictor JEPA sigue la familia

\[
P_\rho=\rho C+(1-\rho)U,
\qquad
K_\rho=\rho C,
\qquad
\rho\in\{0,.25,.5,.75,1\}.
\]

Eso no demuestra todavía que el **entrenamiento JEPA del encoder** sea lo que
permite recuperar la dinámica. La pregunta de este experimento es más causal:

> Con los mismos pares, datos y evaluación, ¿la representación aprendida por
> JEPA estima mejor \(K_\rho\) que los píxeles, una compresión lineal o una CNN
> aleatoria?

No se presupone que JEPA deba ganar. Si un baseline fijo obtiene el mismo
resultado, la conclusión correcta será que la dinámica ya era recuperable sin
aprendizaje autopredictivo.

## Datos y separación final

- Se conservan sin cambios las cuatro fases, el mapa de observación y los cinco
  valores de \(\rho\) del experimento continuo.
- Se usan diez nuevas seeds emparejadas, `101–110`.
- Cada método recibe exactamente los mismos `1024` pares de train, `256` de
  validation y `256` held-out por seed y por \(\rho\).
- Dentro de cada split, todos los valores de \(\rho\) comparten los mismos
  bancos marginales de ventanas; sólo cambia el pairing temporal.
- Train ajusta representaciones y operadores. Validation sólo puede elegir la
  regularización ridge mediante MSE de predicción de features, sin labels de
  fase. El held-out se materializa después de congelar código, tests y configs.
- Los labels de fase del held-out se usan únicamente para medir el resultado,
  nunca para entrenar o seleccionar un método.

Esta separación es procedimental y reproducible, no criptográfica: las seeds
quedan declaradas para que cualquier resultado pueda regenerarse.

## Métodos comparados

| Método | Representación | Ajuste de \(M\) | Papel |
|---|---|---|---|
| `phase_oracle_ols` | indicadores de fase centrados | mínimos cuadrados | sanity ceiling |
| `raw_window_dmd` | ventana centrada de 128 muestras | ridge-DMD | ¿la dinámica ya es lineal en píxeles? |
| `pca3_dmd` | PCA no supervisada, dimensión 3 | ridge-DMD | baseline lineal con igual bottleneck |
| `random_cnn3_dmd` | misma CNN, inicialización aleatoria congelada | ridge-DMD | baseline causal principal |
| `supervised_phase_cnn3_dmd` | CNN de dimensión 3 entrenada con fase | ridge-DMD | techo práctico supervisado |
| `jepa_learned_predictor` | encoder JEPA | predictor aprendido por gradiente | método bajo estudio |

También se reportará `jepa_posthoc_dmd`: el mismo encoder JEPA con un operador
ridge-DMD ajustado después del entrenamiento. Es un diagnóstico para separar
calidad de representación de optimización del predictor, no otro método
entrenado.

La CNN aleatoria comparte arquitectura e inicialización inicial con el encoder
JEPA de la misma seed. El encoder supervisado se entrena una vez por seed sobre
los bancos marginales de train y luego se mantiene fijo para todos los valores
de \(\rho\). PCA también se ajusta una vez por seed, sin labels, sobre la unión
de ventanas actuales y futuras de train.

## Ajuste cerrado del operador

Para los métodos DMD se centran las features usando sólo estadísticas de train
y se resuelve

\[
\widehat M_\lambda
=\arg\min_M
\frac{1}{n}\lVert Z_+-Z_-M^\top\rVert_F^2
+\lambda\lVert M\rVert_F^2.
\]

La grilla queda fijada en
`[0, 1e-6, 1e-4, 1e-2, 1]`. Se elige por menor MSE de features en validation;
los empates favorecen el menor \(\lambda\). Ni el espectro verdadero, ni
\(\rho\), ni los labels de fase participan de esta selección.

## Medición continua primaria

La comparación principal no será la clasificación entre cinco candidatos. Para
cada método y seed se calcula el error de calibración espectral agregado:

\[
\operatorname{MAE}_{\rm spec}
=\frac{1}{5}\sum_{\rho}
\left|\widehat\rho_{\rm spec}(\rho)-\rho\right|,
\qquad
\widehat\rho_{\rm spec}
=\frac{1}{3}\sum_{j=1}^3|\lambda_j(M|_{\operatorname{span}(A)})|.
\]

El alineamiento \(A\) se obtiene a partir de los centroides de fase sólo durante
la evaluación. Para operadores de más de tres dimensiones se restringe \(M\)
al span activo de \(A\); no se eligen post hoc los eigenvalues más parecidos al
target.

El contraste causal primario es pareado por seed:

\[
\Delta_s
=\operatorname{MAE}_{\rm spec}^{\rm random\ CNN}(s)
-\operatorname{MAE}_{\rm spec}^{\rm JEPA}(s).
\]

Se publican los diez valores de \(\Delta_s\), su media y mediana, y un test de
permutación exacto unilateral por cambio de signo sobre la media (`2^10`
asignaciones). Una diferencia positiva favorece a JEPA. No se reemplazarán
estas cantidades por un nuevo gate de aciertos.

Si el span activo de un método no alcanza rango 3 en algún valor de \(\rho\),
la estimación espectral queda indefinida y recibe un error absoluto predeclarado
de `1.0` en el MAE. No se elimina esa condición ni esa seed del contraste.

## Mediciones secundarias

Para cada método, seed y \(\rho\) también se reportan:

- estimación continua por acción, proyectando \(MA\) sobre \(AC\);
- error verdadero \(\lVert MA-AK_\rho\rVert_F/\lVert A\rVert_F\);
- pendiente, intercepto y \(R^2\) de las curvas de calibración;
- rango e invariancia del span activo;
- MSE held-out de predicción de features;
- error de rollout para horizontes `1, 2, 4, 8`;
- identificación discreta en la grilla, sólo para conectar con el experimento
  anterior.

La loss de entrenamiento no decide si se recuperó Koopman.

## Interpretación fijada antes de ver resultados

- Si `random_cnn3_dmd` iguala o supera a JEPA, no podremos atribuir el resultado
  al aprendizaje autopredictivo; las features aleatorias bastan en este toy.
- Si `raw_window_dmd` funciona, la dinámica es directamente recuperable desde
  las ventanas. Se reportará aunque JEPA comprima mejor.
- Si `pca3_dmd` funciona pero DMD crudo no, la compresión y el
  acondicionamiento explican parte importante del resultado.
- Si JEPA supera a los tres baselines no supervisados y se acerca al encoder
  supervisado, habrá evidencia de que aprende una representación dinámica útil.
- Si `jepa_posthoc_dmd` supera al predictor aprendido, el límite está en la
  optimización de \(M\), no necesariamente en el encoder.
- Si el encoder supervisado tampoco recupera el operador, primero se revisará
  la observabilidad o el protocolo; no se ajustará JEPA para compensarlo.

El Koopman autoencoder queda fuera de este bloque inicial porque introduce un
decoder, una loss y decisiones de capacidad nuevas. Se añadirá en un protocolo
separado sólo después de resolver estos falsificadores más directos.
