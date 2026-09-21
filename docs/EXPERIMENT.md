# Diseño y alcance del experimento

## Pregunta

¿Aprende el JEPA una representación `z=fθ(x)` y un predictor lineal `M` tales
que `Mz_t≈z_{t+1}` y, sobre el span dinámico, `MA≈AK`?

## Control experimental

- Estado oculto: cuatro fases; rango dinámico verdadero 3 tras centrar.
- Observación: ventanas de longitud 128 con amplitud, offset, jitter y ruido.
- Control clave: los mismos bancos de ventanas se reutilizan en todas las
  dinámicas; sólo cambia el emparejamiento presente–futuro.
- Train/validation: 1024/256 pares por seed y condición.
- Modelo: CNN `1→16→32`, flatten, latent 3, predictor lineal `3×3` sin bias.
- Entrenamiento: 60 épocas, EMA 0.90, LR del predictor `4×`, freeze del encoder
  tras la época 3, regularización de media/varianza/covarianza.

Estas elecciones delimitan la afirmación. En particular, la dimensión latente
correcta y el freeze son supuestos del resultado.

## Evidencia conservada

| Evidencia | Datos | Repeticiones | Estado |
|---|---|---:|---|
| Oracle matemático | fase verdadera, ciclo | determinista | exacto a precisión numérica |
| Auditoría de observación | tres pairings | 1024 pares por condición | marginales idénticos |
| Tres operadores | estática/cíclica/independiente | 10 seeds cada uno | 10/10 por acción y espectro |
| Familia continua | $\rho\in\{0,.25,.5,.75,1\}$ | 10 seeds por nivel | MAE 0.027; $R^2=0.9998$ |
| Baselines | misma familia continua | 1 seed de validation | exploratorio; sin held-out |

## Reglas de evaluación

La acción se mide con

`||MA-AK||_F / ||A||_F`,

y el espectro se calcula restringiendo `M` al span de los centroides latentes.
Para la clasificación se exigieron al menos 8/10 seeds correctas por condición.
La loss no interviene en la decisión.

En la familia continua, el criterio discreto global no pasa porque `rho=1`
obtiene 7/10. La calibración continua sí es fuerte y se reporta con sus valores,
sin convertirla en un nuevo umbral post-hoc.

## Lectura final

La evidencia apoya posibilidad, no necesidad: JEPA aprende el ciclo y la tasa
de decaimiento bajo esta receta. La comparación exploratoria muestra que DMD
crudo y PCA+DMD también resuelven este dataset y que un DMD post-hoc sobre el
encoder JEPA mejora mucho al predictor entrenado. Por lo tanto, el claim final
no es “JEPA descubre Koopman en general”, sino “este JEPA puede representar la
dinámica no trivial del sistema controlado, aunque el problema también admite
soluciones lineales más simples”.

El estudio se cierra aquí. No se abrió el held-out de baselines y no se infiere
una comparación estadística que no fue ejecutada.
