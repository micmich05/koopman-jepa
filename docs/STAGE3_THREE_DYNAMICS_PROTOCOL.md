# Etapa 3B: identificación neuronal de tres dinámicas

## Pregunta

Con observaciones marginales idénticas, ¿el predictor lineal entrenado por un
JEPA identifica la transición que generó los pares temporales?

Para una representación de fase aprendida

\[
z \approx A\psi_c(r),
\]

la comparación primaria será

\[
E_d(M,A)=\frac{\lVert MA-AK_d\rVert_F}{\lVert A\rVert_F},
\qquad
d\in\{\text{estática},\text{cíclica},\text{independiente}\}.
\]

Los tres operadores candidatos actúan sobre las indicadoras centradas de fase:

- estática: espectro \(\{1,1,1\}\);
- cíclica: espectro \(\{-1,i,-i\}\);
- independiente: espectro \(\{0,0,0\}\).

La normalización por \(\lVert A\rVert_F\) es común a los tres candidatos. Esto
permite comparar sus errores incluso cuando la acción esperada es cero.

## Diseño congelado antes de ejecutar

- Condiciones: `static`, `cyclic`, `independent`.
- Seeds emparejadas: 1–10.
- Las tres condiciones reutilizan exactamente los mismos bancos marginales de
  ventanas dentro de cada seed; sólo cambia el emparejamiento temporal.
- Arquitectura, inicialización, optimizador, número de épocas y regularización
  son idénticos entre condiciones.
- Se reutiliza sin ajuste la receta desarrollada en el ciclo: EMA `0.90`, tasa
  del predictor `4×`, encoder congelado después de la época 3 y 60 épocas.
- El checkpoint es siempre la última época. No se selecciona por labels,
  espectro ni resultado del control.
- Esta primera comparación usa validation y es un experimento de desarrollo,
  no una nueva evaluación ciega.

## Endpoints primarios

Para cada condición y seed:

1. ajustar \(A\) con train y exigir rango activo 3 para que los tres modos de
   fase sean identificables;
2. evaluar \(E_d(M,A)\) para los tres operadores candidatos;
3. asignar al modelo la dinámica con menor error;
4. restringir \(M\) al span activo, comparar su espectro con los tres espectros
   candidatos y asignar el más cercano.

Una seed cuenta como identificación correcta sólo si el span activo tiene rango
3. No se utilizará la loss total para decidir éxito o fracaso.

Con tres candidatos, una clasificación al azar tiene probabilidad `1/3`. El
criterio predeclarado por condición es al menos `8/10` seeds correctas tanto por
acción de entrelazamiento como por espectro. Bajo un modelo binomial nulo,
\(P(X\geq8\mid n=10,p=1/3)\approx0.0034\). La regla no demuestra independencia
perfecta entre seeds, pero explicita el null y evita un umbral porcentual de
loss sin interpretación Koopman.

El resultado global será fuerte para H3 sólo si las tres condiciones alcanzan
el criterio. También se publicarán todos los conteos y márgenes, no sólo el
booleano agregado.

## Diagnósticos secundarios

- rango efectivo, accuracy del probe de fase y error de alineación;
- error absoluto de entrelazamiento para el operador correcto;
- invariancia del span activo bajo el predictor;
- eigenvalues aprendidos y errores contra cada espectro candidato;
- errores de rollout \(\lVert M^hA-AK_d^h\rVert_F/\lVert A\rVert_F\) para
  horizontes `1, 2, 3, 4, 8`;
- comparación entre el predictor entrenado y el operador lineal post-hoc sobre
  embeddings online;
- componentes de loss registradas solamente como diagnóstico de optimización.

## Interpretación anticipada

- Si las tres condiciones se identifican, hay evidencia de que `M` sigue la
  dinámica temporal y no sólo la apariencia marginal de las ventanas.
- Si estática y cíclica funcionan pero independiente no conserva el span de
  fase, el JEPA recupera modos predecibles pero no necesariamente observables
  impredecibles. Eso sería informativo, no un motivo para cambiar el gate.
- Si los embeddings contienen fase pero `M` no distingue los operadores, falla
  el aprendizaje dinámico aunque el probe sea alto.
- Si el operador post-hoc funciona y el predictor entrenado no, la limitación
  está en la optimización o parametrización de `M`, no en el encoder.

