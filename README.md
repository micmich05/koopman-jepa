# Koopman-JEPA

Experimento controlado para preguntar si un JEPA temporal puede aprender una
dinámica lineal no trivial en su espacio latente, sin reconstruir la señal y
sin recibir una loss espectral.

> **Respuesta corta:** sí, con un límite cuantitativo claro. El predictor
> distingue dinámica estática, cíclica e independiente en `10/10` seeds por
> condición. Al interpolar entre ciclo e independencia también aprende la tasa
> de decaimiento casi linealmente, pero la identificación exacta de cinco
> niveles no supera el criterio global predeclarado: el extremo cíclico queda en
> `7/10`.

La [versión compilada en PDF](output/pdf/koopman_jepa_overview.pdf) presenta
esta misma historia con ecuaciones, arquitectura y figuras en LaTeX.

## 1. Pregunta científica

Buscamos una representación

$$
z_t=f_\theta(x_t)
$$

y un predictor lineal $M$ tales que

$$
Mz_t\approx z_{t+1}.
$$

Sabemos cuál es la dinámica verdadera sobre las variables ocultas de fase. Si
$\psi(r)$ representa esas fases, $K$ es su operador y el encoder aprende
$z\approx A\psi$, entonces la relación que debe cumplir el predictor es

$$
\boxed{MA\approx AK}.
$$

No buscamos que las matrices $M$ y $K$ sean iguales: el encoder puede elegir
otra base latente. La ecuación de entrelazamiento comprueba que ambas matrices
describen la misma dinámica en coordenadas diferentes.

## 2. Dataset controlado

La variable oculta tiene cuatro fases, $r\in\{0,1,2,3\}$. Cada fase genera una
ventana de longitud 128 con un pulso principal y uno secundario. A cada ventana
se le aplican nuisance factors independientes:

$$
x_t[\tau]
=a_t\,s_{r_t}(\tau-\delta_t)+b_t+\epsilon_{t,\tau},
$$

donde la amplitud $a_t$, el desplazamiento $\delta_t$, el offset $b_t$ y el
ruido $\epsilon$ cambian entre ejemplos. Por eso la red no puede memorizar una
única señal por fase.

Dentro de cada seed se generan una sola vez los bancos de ventanas actuales y
futuras. Las tres condiciones reutilizan exactamente esos bancos; sólo cambia
cómo se forman los pares temporales:

| Condición | Pairing de fases | Acción esperada | Espectro activo |
|---|---|---|---|
| Estática | $r_{t+1}=r_t$ | conservar cada fase | $\{1,1,1\}$ |
| Cíclica | $r_{t+1}=r_t+1\pmod 4$ | rotar las fases | $\{-1,i,-i\}$ |
| Independiente | todos los pares balanceados | media condicional cero | $\{0,0,0\}$ |

Cada condición y seed contiene:

- `1024` pares de train y `256` pares de validation;
- ventanas de forma `1×128`;
- marginales uniformes de fase actual y futura;
- exactamente la misma arquitectura, inicialización y presupuesto de training.

Este control es crucial: si $M$ cambia entre condiciones, el cambio sólo puede
provenir del pairing temporal, no de diferencias visuales en el dataset.

## 3. Arquitectura

```mermaid
flowchart LR
    xt["Ventana actual x_t<br/>1 × 128"] --> online["Encoder online f_θ<br/>Conv 1→16, k=7, s=2<br/>Conv 16→32, k=5, s=2<br/>Flatten 32×32<br/>Linear 1024→3"]
    online --> zt["z_t ∈ R³"]
    zt --> predictor["Predictor lineal<br/>M ∈ R³ˣ³, sin bias"]
    predictor --> prediction["ẑ_(t+1) = M z_t"]

    xnext["Ventana futura x_(t+1)<br/>1 × 128"] --> target["Encoder target f_ξ<br/>misma CNN<br/>sin gradiente"]
    target --> ztarget["z⁺_(t+1) ∈ R³"]
    prediction --> mse["MSE predictiva"]
    ztarget --> mse

    online -. "EMA ξ ← 0.90 ξ + 0.10 θ" .-> target
    zt --> regularization["Anti-colapso<br/>media + 5·varianza + covarianza"]
```

Detalles de optimización:

- CNN online: AdamW con learning rate `1e-3`;
- predictor $M$: learning rate `4e-3`;
- target encoder: media móvil exponencial con momentum `0.90`;
- encoder online congelado después de la época 3;
- predictor entrenado durante 60 épocas;
- dimensión latente `3`, igual a la dimensión del subespacio centrado de cuatro
  fases.

Congelar el encoder no forma parte de la teoría de Koopman: fue una decisión de
optimización para evitar que la escala del latent siguiera cambiando mientras
$M$ intentaba alcanzarlo. Por eso se reporta como una limitación.

## 4. Cómo evaluamos el operador

Para cada modelo estimamos $A$ a partir de los centroides latentes de fase y
comparamos el predictor con los tres candidatos:

$$
E_d(M,A)=
\frac{\lVert MA-AK_d\rVert_F}{\lVert A\rVert_F},
\qquad
d\in\{\text{estática, cíclica, independiente}\}.
$$

La dinámica predicha es la que obtiene menor $E_d$. Repetimos la comparación
con los eigenvalues de $M$ restringido al span activo. Una seed sólo cuenta si
ese span tiene rango 3.

La regla fue fijada antes de ejecutar las condiciones nuevas: al menos `8/10`
seeds correctas por condición. Con tres candidatos elegidos al azar,
$P(X\geq8\mid n=10,p=1/3)\approx0.0034$. La loss total se registra para
diagnóstico, pero no decide si se aprendió Koopman.

## 5. Resultado principal

| Dinámica verdadera | Acción correcta | Espectro correcto | Error de acción mediano | Margen al segundo candidato |
|---|---:|---:|---:|---:|
| Estática | 10/10 | 10/10 | 0.142 | 0.806 |
| Cíclica | 10/10 | 10/10 | 0.068 | 0.877 |
| Independiente | 10/10 | 10/10 | 0.015 | 0.981 |

Las 30 corridas conservaron rango activo 3 y lograron probe lineal de fase del
100%.

### Cómo leer el mapa de calor

![Errores de acción y espectro para los tres operadores](docs/figures/three_dynamics_operator_identification.png)

- Las filas indican la dinámica usada para entrenar.
- Las columnas son los tres operadores candidatos usados para explicar $M$.
- Un valor bajo y oscuro significa mayor compatibilidad.
- La diagonal es el operador correcto. Es el mínimo en las seis comparaciones:
  tanto por acción $MA-AK_d$ como por espectro.
- Los valores fuera de la diagonal quedan aproximadamente entre `0.85` y
  `1.58`, lejos de los errores correctos `0.015–0.203`. No es una victoria por
  diferencias diminutas o por redondeo.

La figura es la evidencia más directa de que $M$ responde a la dinámica. Si el
modelo sólo reconociera la apariencia de las cuatro fases, las tres filas
deberían producir predictores similares porque sus marginales son idénticas.

### Qué dicen los rollouts

![Error de rollout por horizonte](docs/figures/three_dynamics_rollout.png)

El segundo gráfico evalúa

$$
\frac{\lVert M^hA-AK^h\rVert_F}{\lVert A\rVert_F}
$$

para horizontes $h\in\{1,2,3,4,8\}$.

- Independiente cae prácticamente a error cero desde $h=2$: un predictor
  contractivo converge rápido a la acción nula esperada.
- Estática comienza cerca de `0.142` y llega aproximadamente a `0.305` en
  $h=8$.
- Cíclica comienza cerca de `0.068` pero llega aproximadamente a `0.377` en
  $h=8$.

Por lo tanto, el operador correcto está claramente identificado a un paso, pero
los pequeños errores se acumulan. La consistencia a largo horizonte es la
principal limitación cuantitativa visible en este experimento.

## 6. Generalización: ¿aprende la tasa de decaimiento?

El control de tres dinámicas sólo comparaba operadores muy separados. Para
hacer la pregunta más exigente construimos una familia continua:

$$
P_\rho=\rho C+(1-\rho)U,
\qquad
K_\rho=\rho C,
\qquad
\operatorname{spec}(K_\rho)=\rho\{-1,i,-i\}.
$$

$C$ es el ciclo y $U$ genera una fase futura uniforme e independiente. Usamos
$\rho\in\{0,.25,.5,.75,1\}$: ahora el modelo debe recuperar no sólo el ángulo
oscilatorio, sino también cuánto persiste la dinámica. Las ventanas, los
marginales, la arquitectura y la receta permanecen fijos; se entrenan 10 seeds
por nivel.

| $\rho$ verdadero | Acción correcta | Espectro correcto | Módulo espectral mediano |
|---:|---:|---:|---:|
| 0.00 | 10/10 | 10/10 | 0.016 |
| 0.25 | 10/10 | 10/10 | 0.239 |
| 0.50 | 9/10 | 9/10 | 0.476 |
| 0.75 | 8/10 | 8/10 | 0.718 |
| 1.00 | 7/10 | 7/10 | 0.948 |

La regla congelada exigía al menos `8/10` aciertos por acción y espectro en
**cada** fila. Por eso el resultado global formal es negativo: $\rho=1$ no
alcanza el corte. No se cambió el criterio después de ver los datos.

![Identificación entre cinco tasas de decaimiento](docs/figures/decay_operator_identification.png)

El mapa de calor sí exhibe una diagonal nítida. Los errores crecen gradualmente
al alejarse del $\rho$ verdadero; los fallos exactos aparecen cuando una seed
queda más cerca del nivel inmediatamente inferior. Esto es más exigente que el
control anterior, donde $\rho=.75$ ni siquiera era un candidato.

![Calibración del módulo y espectro complejo](docs/figures/decay_spectral_calibration.png)

La medición continua es fuerte: los cinco módulos medianos tienen MAE `0.027`,
la recta aprendido-versus-verdadero tiene pendiente `0.937`, intercepto `0.011`
y $R^2=0.9998$. El sesgo es contractivo, especialmente en algunas seeds de
$\rho=.75$ y $1$. Por eso podemos afirmar que el JEPA aprende un **continuo de
operadores amortiguados en promedio**, pero todavía no que cuantifica cada
nivel con robustez seed-a-seed.

El rollout refuerza esa lectura: para $\rho<1$ el error decrece a horizontes
largos porque tanto el operador verdadero como el aprendido se contraen; para
$\rho=1$ se acumula hasta aproximadamente `0.38` en $h=8$. La figura completa
está en el [notebook ejecutado](notebooks/koopman_decay_generalization.ipynb).

## 7. Conclusión

En este toy controlado, el JEPA aprende una representación de fase de rango 3 y
un predictor lineal cuya acción y espectro cambian con la transición temporal.
Esto constituye evidencia positiva de aprendizaje de una restricción Koopman
no trivial, no del operador Koopman completo. La extensión amortiguada muestra
además que el espectro responde de manera calibrada a un parámetro dinámico
continuo, aunque el criterio discreto más estricto no pasa en todas las seeds.

Todavía no demuestra generalidad:

- sólo estudiamos cuatro estados y un latent de dimensión conocida;
- el control conjunto usa validation, no una nueva partición ciega;
- el encoder se congela temprano;
- los rollouts estático y cíclico acumulan error;
- sólo variamos una familia de un parámetro con el mismo mapa de observación;
- el predictor presenta un sesgo contractivo que confunde niveles vecinos en
  algunas seeds.

El próximo paso confirmatorio es repetir la familia $K_\rho$ sobre emisiones
nuevas y ciegas, sin retocar la receta a partir de estas mismas 50 corridas.
Después corresponde variar el mapa de observación o el número de fases y
estudiar cómo reducir el sesgo contractivo sin introducir los eigenvalues
verdaderos en la loss.

## 8. Dónde mirar

- [Generalización amortiguada](notebooks/koopman_decay_generalization.ipynb): 50
  corridas, tres figuras y análisis del resultado mixto.
- [Protocolo de generalización](docs/DECAY_GENERALIZATION_PROTOCOL.md): familia,
  decisión y alcance fijados antes de entrenar.
- [Control base de tres operadores](notebooks/stage3_three_dynamics_control.ipynb):
  identificación gruesa en 30 corridas.
- [Recorrido científico corto](notebooks/README.md): notebooks esenciales.
- [Protocolo del control base](docs/STAGE3_THREE_DYNAMICS_PROTOCOL.md): decisión
  fijada antes de entrenar las tres condiciones.
- [Research brief](RESEARCH_BRIEF.md): matemática, hipótesis y alcance completo.
- [Bitácora técnica](notebooks/EXPERIMENT_LOG.md): debugging y gates históricos.
- [Configuraciones](configs/README.md): protocolo principal frente a archivos
  de soporte e historial.

## Reproducir

```bash
uv sync --extra dev
uv run jupyter nbconvert \
  --to notebook --execute --inplace \
  notebooks/koopman_decay_generalization.ipynb \
  --ExecutePreprocessor.timeout=-1
```

```bash
uv run pytest -q
uv run ruff check .
```

Compilar la versión LaTeX:

```bash
mkdir -p output/pdf
tectonic docs/koopman_jepa_overview.tex --outdir output/pdf
```
