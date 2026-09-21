# Koopman-JEPA

Experimento controlado para preguntar si un JEPA temporal puede aprender una
dinámica lineal no trivial en su espacio latente, sin reconstruir la señal ni
recibir una loss espectral.

## La pregunta

Buscamos una representación \(z_t=f_\theta(x_t)\) y un predictor lineal \(M\)
que satisfagan

\[
Mz_t\approx z_{t+1}.
\]

Si \(z\approx A\psi\) y \(K\) es el operador conocido sobre las variables de
fase, la prueba más directa es

\[
MA\approx AK
\]

y que el espectro de \(M\) sobre el span activo coincida con el de \(K\).

## Respuesta actual

**Sí, en este sistema sintético de cuatro fases.** El predictor entrenado
identifica correctamente tres dinámicas diferentes usando exactamente los
mismos marginales de observación:

| Dinámica verdadera | Espectro esperado | Acción correcta | Espectro correcto | Error de acción mediano |
|---|---:|---:|---:|---:|
| Estática | `{1, 1, 1}` | 10/10 | 10/10 | 0.142 |
| Cíclica | `{-1, i, -i}` | 10/10 | 10/10 | 0.068 |
| Independiente | `{0, 0, 0}` | 10/10 | 10/10 | 0.015 |

Las 30 corridas conservaron rango activo 3 y obtuvieron probe lineal de fase
del 100%. La clasificación usa el error

\[
E_d(M,A)=\frac{\lVert MA-AK_d\rVert_F}{\lVert A\rVert_F}
\]

contra los tres operadores candidatos. La loss total no participa en el
veredicto.

![Errores de acción y espectro para los tres operadores](docs/figures/three_dynamics_operator_identification.png)

La diagonal oscura muestra que cada predictor está mucho más cerca del operador
que realmente generó sus pares temporales. El margen mediano frente al segundo
candidato es `0.806`, `0.877` y `0.981` para estática, cíclica e independiente.

El error multi-step crece para estática y cíclica, por lo que la recuperación no
es exacta y los horizontes largos siguen siendo una limitación:

![Error de rollout por horizonte](docs/figures/three_dynamics_rollout.png)

## Qué hicimos

1. Definimos un sistema oculto de cuatro fases. En el subespacio centrado, el
   ciclo tiene espectro `{-1,i,-i}`.
2. Verificamos con features oracle que las convenciones algebraicas, el espectro
   y los rollouts son correctos.
3. Construimos tres datasets con las mismas ventanas actuales y futuras, pero
   distinto pairing: estático, cíclico e independiente.
4. Entrenamos la misma CNN, predictor lineal y receta de optimización en 10
   seeds emparejadas por condición.
5. Comparamos cada predictor contra los tres operadores posibles. Los 30 modelos
   eligieron el correcto tanto por acción como por espectro.

El notebook principal, con outputs, figuras y análisis escrito, es
[`stage3_three_dynamics_control.ipynb`](notebooks/stage3_three_dynamics_control.ipynb).

## Qué demuestra y qué no

El resultado apoya que, en este toy controlado, el predictor aprende estructura
temporal y no sólo la apariencia de cada fase: los marginales son iguales y el
operador cambia cuando cambia el pairing.

Todavía no demuestra que un JEPA recupere operadores Koopman generales:

- la dinámica tiene sólo cuatro estados y un subespacio verdadero de dimensión 3;
- el encoder se congela después de tres épocas para estabilizar la optimización;
- el control de tres dinámicas usa validation, no una partición ciega nueva;
- falta estudiar horizontes, observación parcial, ruido fuera de distribución y
  eigenvalues con módulo menor que uno.

## Evidencia principal

El [índice breve de notebooks](notebooks/README.md) organiza las cinco piezas de
evidencia necesarias. El [protocolo de tres dinámicas](docs/STAGE3_THREE_DYNAMICS_PROTOCOL.md)
fue commiteado antes de ejecutar las condiciones nuevas.

Los numerosos smokes previos sirvieron para detectar lag del target EMA,
optimización lenta de `M` e inestabilidad de escala. No son resultados
científicos independientes. Se conservan, separados de la lectura principal,
en la [bitácora técnica](notebooks/EXPERIMENT_LOG.md).

La reproducción del paper de invariantes \(\lambda=1\) es un antecedente
parcial y está documentada por separado en
[`PAPER_REPLICATION_SPEC.md`](docs/PAPER_REPLICATION_SPEC.md). La pureza MLP
publicada no fue reproducida; eso no se usa como evidencia a favor ni en contra
del experimento no trivial.

## Estructura del repositorio

- [`notebooks/README.md`](notebooks/README.md): recorrido científico corto.
- [`notebooks/EXPERIMENT_LOG.md`](notebooks/EXPERIMENT_LOG.md): historial técnico completo.
- [`RESEARCH_BRIEF.md`](RESEARCH_BRIEF.md): formulación matemática, hipótesis y alcance.
- [`configs/`](configs): protocolos y configuraciones versionadas.
- [`src/koopman_jepa/`](src/koopman_jepa): datos, modelos, entrenamiento y métricas.
- [`tests/`](tests): contratos numéricos y reproducibilidad de notebooks.

## Reproducir

```bash
uv sync --extra dev
uv run jupyter nbconvert \
  --to notebook --execute --inplace \
  notebooks/stage3_three_dynamics_control.ipynb \
  --ExecutePreprocessor.timeout=1800
```

Verificación del repositorio:

```bash
uv run pytest -q
uv run ruff check .
```
