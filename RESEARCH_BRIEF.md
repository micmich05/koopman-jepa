# Research brief: JEPA y dinámicas de Koopman no triviales

Estado: protocolo previo a implementación  
Fecha: 19 de septiembre de 2026

## 1. Objetivo

Estudiar si una arquitectura predictiva de embeddings conjuntos para series temporales puede aprender, sin reconstrucción de la observación y sin una loss espectral explícita, un subespacio finito aproximadamente invariante bajo Koopman con dinámica no trivial.

La pregunta general es:

> Cuando los datos contienen un subespacio de Koopman finito y conocido, ¿puede un JEPA temporal no colapsado aprender una representación latente cerrada bajo un predictor lineal y recuperar sus modos dinámicos no triviales?

La primera pregunta experimental será más limitada:

> En un sistema oculto de cuatro fases con transición cíclica, ¿puede un JEPA con predictor lineal recuperar el subespacio centrado de fase y el espectro \(\{-1,i,-i\}\), sin imponer esos eigenvalues durante el entrenamiento?

Este proyecto no busca demostrar que JEPA aprende el operador de Koopman completo. Busca evidencia controlada sobre la recuperación de una restricción finita y conocida del operador.

## 2. Motivación y antecedente directo

El antecedente principal es:

Pablo Ruiz-Morales, Dries Vanoost, Davy Pissoort y Mathias Verbeke. *Koopman Invariants as Drivers of Emergent Time-Series Clustering in Joint-Embedding Predictive Architectures*. AAAI 2026.

El paper estudia mezclas de regímenes ergódicos dinámicamente inmiscibles. Las indicadoras de esos regímenes son eigenfunctions de Koopman con eigenvalue \(1\). Bajo hipótesis idealizadas, esas funciones proporcionan una solución de mínimo global para la loss JEPA. Empíricamente, el trabajo encuentra clustering por régimen y una matriz predictora cercana a la identidad cuando el predictor lineal se inicializa en la identidad.

El paper propone como trabajo futuro estudiar otras eigenfunctions, incluidas las asociadas a oscilaciones y decaimiento lento. Nuestro proyecto comienza por el caso oscilatorio más simple.

Limitaciones relevantes del antecedente:

- la loss admite representaciones constantes o colapsadas;
- las indicadoras constituyen una solución óptima, pero la loss no garantiza por sí sola que se recupere todo su span;
- la interpretabilidad del predictor depende fuertemente de su inicialización cerca de la identidad;
- un buen error predictivo no identifica de manera única una representación Koopman.

Por eso nuestro objeto de estudio no será sólo la existencia de una solución Koopman, sino también su selección durante el entrenamiento.

## 3. Formulación matemática

Para una dinámica determinista discreta

\[
x_{t+1}=F(x_t),
\]

el operador de Koopman actúa sobre observables mediante

\[
(\mathcal K g)(x)=g(F(x)).
\]

En un sistema estocástico, usaremos el operador de esperanza condicional

\[
(\mathcal K g)(x)
=
\mathbb E[g(X_{t+1})\mid X_t=x].
\]

En este segundo caso, \(\mathcal K\phi=\lambda\phi\) es una igualdad en esperanza condicional y no necesariamente una igualdad trayectoria por trayectoria.

Si un vector de observables \(\psi\) genera un subespacio finito invariante,

\[
\psi(F(x))=K\psi(x).
\]

Si el encoder aprende

\[
z=f_\theta(x)\approx A\psi(x),
\]

la condición general que queremos comprobar es

\[
\boxed{MA\approx AK.}
\]

Cuando \(A\) es cuadrada e invertible, esto se reduce a

\[
M\approx AKA^{-1}.
\]

La relación de entrelazamiento \(MA\approx AK\) será preferida en las evaluaciones porque también es válida para representaciones sobrecompletas.

## 4. Sistema de cuatro fases

La variable oculta sigue

\[
r_{t+1}=r_t+1\pmod 4.
\]

Las cuatro indicadoras se transforman mediante la matriz de permutación

\[
P=
\begin{pmatrix}
0&0&0&1\\
1&0&0&0\\
0&1&0&0\\
0&0&1&0
\end{pmatrix},
\qquad
\operatorname{spec}(P)=\{1,i,-1,-i\}.
\]

Estas cuatro posiciones se denominarán **fases**, no regímenes inmiscibles: cada fase se transforma en la siguiente.

La función constante corresponde al eigenvalue \(1\) y no informa sobre la fase. Por eso definimos las indicadoras centradas

\[
\psi_c(r)=e_r-\frac14\mathbf 1.
\]

Su span es el subespacio tridimensional \(\mathbf 1^\perp\), donde

\[
\boxed{
\operatorname{spec}(P|_{\mathbf 1^\perp})
=
\{-1,i,-i\}.
}
\]

La dimensión latente principal será, por tanto, \(d=3\). Dimensiones mayores se estudiarán sólo después y su espectro se analizará sobre el subespacio activo, no sobre toda la matriz predictora.

## 5. Hipótesis

### H0: reproducción del caso invariante

Una implementación JEPA compatible con el paper AAAI puede aprender representaciones constantes dentro de cada régimen inmiscible y un predictor que actúa aproximadamente como la identidad sobre el subespacio activo.

### H1: ciclo no trivial

Con observaciones que permiten identificar la fase, un JEPA no colapsado con predictor lineal puede aprender una representación equivalente al subespacio centrado de las fases y recuperar \(\{-1,i,-i\}\).

### H2: el anti-colapso es necesario pero no debe imponer el espectro

EMA y prediction loss pueden producir colapso total o recuperación parcial. Una restricción genérica de centrado y whitening puede favorecer rango completo sin codificar la transición cíclica ni sus eigenvalues.

### H3: la estructura aprendida depende de la dinámica, no de las observaciones marginales

Manteniendo fija la distribución marginal de observaciones, el espectro aprendido debe cambiar al sustituir la transición cíclica por una transición estática o independiente.

## 6. Hoja de ruta experimental

### Etapa 0 — Replicación mecanística del paper AAAI

Antes de probar la extensión, reproduciremos los mecanismos centrales del paper, no necesariamente cada cifra reportada.

Objetivos:

1. generar varios regímenes temporalmente persistentes e inmiscibles;
2. entrenar un encoder temporal, target encoder EMA y predictor;
3. observar separación por régimen en datos de test;
4. comprobar que un predictor lineal inicializado en identidad actúa cerca de la identidad sobre el subespacio latente activo;
5. repetir con inicialización aleatoria para medir cuánto depende el resultado del sesgo de inicialización;
6. monitorizar colapso y rango efectivo, además de clustering.

No será necesario igualar exactamente la pureza reportada ni ejecutar desde el principio los 18 regímenes y 180 000 pares del paper. La primera reproducción usará una versión reducida pero estructuralmente equivalente. Si el mecanismo no aparece, se acercará progresivamente la configuración a la del artículo.

**Gate 0:** no se avanza al ciclo hasta que el pipeline reproduzca cualitativamente el caso identidad y sepamos si el resultado depende de la inicialización.

### Etapa 1 — Unit test Koopman sin encoder

Usar directamente las indicadoras centradas \(\psi_c(r_t)\) y ajustar el operador por mínimos cuadrados.

Debe recuperarse, hasta precisión numérica,

\[
\{-1,i,-i\}.
\]

Esta etapa valida orientación de matrices, cálculo espectral, matching, left eigenvectors y rollouts antes de introducir optimización neuronal.

### Etapa 2 — Tres dinámicas con observaciones idénticas

Usaremos el mismo generador de observaciones y tres leyes de transición:

| Condición | Transición de fase | Espectro esperado en el espacio centrado |
|---|---|---|
| Estática | \(r_{t+1}=r_t\) | \(\{1,1,1\}\) |
| Cíclica | \(r_{t+1}=r_t+1\pmod 4\) | \(\{-1,i,-i\}\) |
| Independiente | \(r_{t+1}\sim\mathrm{Uniforme}\{0,1,2,3\}\) | \(\{0,0,0\}\) |

Las tres condiciones tendrán las mismas frecuencias marginales de fase. Sólo cambiará la dependencia temporal.

**Gate 1:** el mismo pipeline e hiperparámetros deben distinguir correctamente las tres dinámicas en varias seeds.

### Etapa 3 — Ocultar la fase tras ventanas temporales

Cada macroestado \(t\) genera una ventana

\[
X_t[\tau]
=
a_t s_{r_t}(\tau-\delta_t)+b_t+\epsilon_{t,\tau},
\]

donde \(a_t\), \(\delta_t\), \(b_t\) y \(\epsilon_{t,\tau}\) son factores nuisance.

Progresión:

1. templates deterministas y fácilmente separables;
2. amplitud y fase interna variables;
3. ruido y distribuciones parcialmente solapadas;
4. parámetros nuisance fuera de distribución en test.

Las primeras ventanas context-target no se solaparán. El split se hará por trayectoria o por parámetros generativos, nunca mezclando ventanas casi idénticas entre train y test.

### Etapa 4 — Robustez y selección de modos

Estudiar:

- prediction-only frente a prediction + whitening;
- predictor aleatorio, identidad y ortogonal;
- distintas dimensiones latentes;
- distintos horizontes \(\Delta\);
- observación parcial y ruido;
- qué subconjuntos de modos sobreviven cuando la capacidad es insuficiente.

Para un horizonte \(\Delta\), el espectro esperado es \(\lambda_j^\Delta\). Esta propiedad funcionará como test adicional de consistencia y aliasing.

### Etapa 5 — Decaimiento y dinámica estocástica

Sólo después del caso periódico se introducirán eigenvalues con módulo menor que uno, por ejemplo mediante una cadena de Markov conocida o un sistema lineal latente con observación no lineal.

En esta etapa, los eigenmodes no serán predecibles trayectoria por trayectoria. Las evaluaciones distinguirán error respecto de realizaciones individuales y error respecto de la esperanza condicional.

## 7. Arquitectura inicial

### Encoder

Una CNN 1D pequeña con global pooling y proyección final a \(d=3\). Global pooling reduce la posibilidad de memorizar posiciones absolutas dentro de la ventana.

### Target encoder

Copia EMA del online encoder. Se registrará explícitamente la diferencia entre las representaciones online y target.

### Predictor

\[
\hat z_{t+1}=Mz_t,
\qquad
M\in\mathbb R^{3\times3},
\]

sin bias ni no linealidad.

### Losses

Condición base:

\[
\mathcal L_{pred}
=
\mathbb E\left[
\|M f_\theta(X_t)-\operatorname{sg}(f_{EMA}(X_{t+1}))\|_2^2
\right].
\]

Condición no colapsada:

\[
\mathcal L
=
\mathcal L_{pred}
+\alpha\mathcal L_{mean}
+\beta\mathcal L_{cov},
\]

donde los términos adicionales centran la representación y aproximan su covarianza a la identidad. No contendrán información sobre \(P\) ni sobre sus eigenvalues.

## 8. Evaluación

### 8.1 Colapso y rango efectivo

Reportar:

- singular values de los embeddings;
- eigenvalues de la covarianza;
- rango efectivo;
- varianza intra-fase y entre fases;
- norma y distribución de los embeddings.

### 8.2 Recuperación del subespacio

Ajustar en train una transformación lineal \(A\) entre las features verdaderas y aprendidas:

\[
z_t\approx A\psi_c(r_t).
\]

Evaluar en test:

- error de proyección;
- canonical correlations o principal angles;
- \(R^2\) de la alineación;
- generalización a nuisances no vistos.

### 8.3 Cierre dinámico

Medir

\[
E_{int}
=
\frac{\|MA-AK_c\|_F}{\|AK_c\|_F}.
\]

Esta será una de las métricas principales.

### 8.4 Espectro

Comparar el espectro del predictor restringido al subespacio activo mediante matching óptimo:

\[
E_{spec}
=
\min_\pi
\frac1q\sum_{j=1}^{q}
|\hat\lambda_j-\lambda_{\pi(j)}|.
\]

No se usará el espectro completo de una matriz sobrecompleta, ya que su acción fuera del span de los datos no está identificada.

### 8.5 Eigenfunctions

Si

\[
w_j^\top M=\hat\lambda_jw_j^\top,
\]

entonces la eigenfunction latente candidata es

\[
\hat\phi_j(X)=w_j^\top f(X).
\]

Compararemos estas funciones con los modos Fourier verdaderos de la fase y mediremos el residual

\[
\hat\phi_j(X_{t+1})-\hat\lambda_j\hat\phi_j(X_t).
\]

### 8.6 Predicción multi-step

\[
E_k
=
\frac{\|Z_{t+k}-M^kZ_t\|_F}{\|Z_{t+k}\|_F},
\qquad k=1,\ldots,8.
\]

En sistemas estocásticos también se comparará contra la esperanza condicional o los centroides de fase.

### 8.7 Operador post hoc

Al terminar el entrenamiento, se congelará un solo encoder y se ajustará

\[
K_{post}=Z_+Z^\dagger.
\]

Se comparará \(K_{post}\) con el predictor entrenado. Una gran discrepancia indicará que el desfase entre online y EMA impide interpretar directamente \(M\) como un endomorfismo Koopman.

### 8.8 Estadística

Los resultados principales usarán al menos 10 seeds. Se reportarán mediana, dispersión y tasa de ejecuciones exitosas, no sólo el mejor modelo.

## 9. Baselines

Orden inicial:

1. features de fase verdaderas + least squares;
2. DMD/EDMD sobre observaciones;
3. red con residual Koopman directo, estilo dictionary learning/DeepDMD;
4. Koopman autoencoder;
5. VAMP o time-lagged CCA, si el alcance lo permite;
6. control con pares temporales barajados.

Un autoencoder convencional y un predictor no lineal pueden incluirse como controles secundarios, pero no sustituyen a los baselines Koopman.

## 10. Criterios de resultado positivo

Un resultado positivo fuerte requiere simultáneamente:

- ausencia de colapso y rango latente efectivo completo;
- alineación del latent con el subespacio verdadero en test;
- error de entrelazamiento bajo;
- espectro correcto sobre el subespacio activo en la mayoría de las seeds;
- eigenfunctions latentes alineadas con los modos Fourier;
- rollouts multi-step coherentes;
- comportamiento correcto en las condiciones estática, cíclica e independiente;
- acuerdo razonable entre el predictor entrenado y el operador post hoc.

No será evidencia suficiente:

- un círculo en PCA;
- clusters o buen linear probe;
- t-SNE visualmente atractivo;
- baja prediction loss;
- periodicidad latente sin cierre lineal;
- eigenvalues correctos en dimensiones que los datos no utilizan.

## 11. Interpretación de posibles resultados

### Positivo fuerte

JEPA recupera consistentemente el subespacio, sus eigenfunctions y su dinámica espectral sin una loss Koopman explícita.

### Positivo parcial

JEPA recupera sólo algunos modos. La pregunta pasa a ser qué sesgo de arquitectura u optimización determina la selección modal.

### Dependiente del anti-colapso

Prediction-only colapsa, pero una restricción estadística genérica permite recuperar el espectro. Esto indicaría que predictibilidad y diversidad cumplen papeles complementarios.

### Negativo informativo

El modelo predice bien pero su subespacio o espectro no es estable entre seeds, condiciones o nuisances. Esto mostraría que el éxito para \(\lambda=1\) no se generaliza automáticamente a modos no triviales.

## 12. Alcance y viabilidad

La replicación reducida y el ciclo de cuatro fases pueden ejecutarse en una MacBook con Apple Silicon. No requieren un foundation model ni TS-JEPA completo. Una GPU externa sólo sería útil posteriormente para grandes barridos de hiperparámetros.

El toy cíclico debe considerarse un test de mecanismo, no una contribución final suficiente. El valor científico mayor estará en caracterizar colapso, selección de modos, generalización y la transición posterior a eigenvalues de decaimiento.

## 13. Decisión antes de implementar

El orden acordado propuesto es:

1. replicación mecanística reducida del caso \(\lambda=1\);
2. unit tests espectrales con features verdaderas;
3. control estático/cíclico/independiente con el mismo generador;
4. fase oculta tras ventanas temporales;
5. robustez, baselines y dinámica estocástica.

No se comenzará la implementación del experimento principal hasta cerrar la configuración de la Etapa 0 y sus criterios de avance.
