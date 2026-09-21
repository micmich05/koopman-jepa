# Protocolo congelado: generalización a dinámica amortiguada

Estado: **definido antes de entrenar**. Este documento fija qué vamos a medir y
qué contará como evidencia; no contiene resultados.

## Pregunta

El experimento anterior distinguió tres operadores extremos: identidad, ciclo e
independencia. Ahora preguntamos algo más fino:

> ¿Puede un JEPA aprender no sólo que existe una oscilación, sino también cuánto
> de esa dinámica persiste de un paso al siguiente?

Usamos una familia de transiciones estocásticas entre cuatro fases:

\[
P_\rho = \rho C + (1-\rho)U,
\qquad
\rho\in\{0,\tfrac14,\tfrac12,\tfrac34,1\},
\]

donde \(C\) avanza una fase de forma cíclica y \(U\) elige la fase futura de
forma uniforme e independiente. En el subespacio centrado de dimensión tres,
el modo constante desaparece y el operador esperado es

\[
K_\rho=\rho C,
\qquad
\operatorname{spec}(K_\rho)=\rho\{-1,i,-i\}.
\]

Así, el ángulo de los autovalores codifica la oscilación y su módulo \(\rho\)
codifica la memoria o amortiguamiento. Los extremos ya tienen una interpretación
conocida: \(\rho=1\) es el ciclo determinista y \(\rho=0\) es independencia.

## Control experimental

- Las ventanas observadas, su ruido y sus nuisance variables son exactamente los
  mismos para todos los valores de \(\rho\); sólo cambia el emparejamiento
  presente--futuro.
- Cada condición tiene marginales uniformes idénticas en presente y futuro.
- Para cada fase presente construimos \(4r\) pares. El sucesor cíclico aparece
  \(r(1+3\rho)\) veces y cada una de las otras fases, \(r(1-\rho)\) veces. Con
  los \(\rho\) elegidos y \(r=64\) en train / \(r=16\) en validation, todos los
  conteos son enteros y no hay muestreo aproximado de la matriz de transición.
- Se mantienen sin cambios la arquitectura, la loss, el optimizador y la receta
  de entrenamiento del control de tres dinámicas.
- Se entrenan 10 seeds independientes por cada uno de los cinco valores de
  \(\rho\). No se construye un conjunto de test oculto: ésta es una extensión de
  desarrollo, no una confirmación final.

## Qué se evalúa

Para cada modelo estimamos el mapa de alineamiento \(A\) entre indicadores de
fase centrados y el latente. Comparamos el predictor aprendido \(M\) contra los
cinco candidatos mediante

\[
e_{\text{acción}}(\rho)
=\frac{\lVert MA-AK_\rho\rVert_F}{\lVert A\rVert_F}.
\]

El denominador es común a todos los candidatos, por lo que elegir el menor error
no favorece a operadores de mayor o menor norma. También restringimos \(M\) al
span activo de \(A\) y comparamos sus tres autovalores con
\(\rho\{-1,i,-i\}\), usando el apareamiento de mínimo error total.

El resultado principal será:

1. identificación por acción: qué \(\rho\) minimiza
   \(e_{\text{acción}}\);
2. identificación por espectro: qué \(\rho\) minimiza el error espectral medio.

Una condición cuenta como correctamente identificada sólo si su span activo
tiene rango tres. El criterio agregado, fijado antes de correr, es **al menos
8/10 seeds correctas, por ambos métodos, para cada uno de los cinco valores de
\(\rho\)**. Bajo elección uniforme entre cinco candidatos, la probabilidad
binomial de obtener al menos 8/10 aciertos es \(7.79264\times10^{-5}\) por
condición.

## Diagnósticos, no umbrales adicionales

Además mostraremos:

- autovalores aprendidos en el plano complejo;
- módulo espectral aprendido frente a \(\rho\), como curva de calibración;
- errores de rollout a horizontes \(1,2,3,4,8\);
- error correcto, margen respecto del segundo candidato, rango activo y probe de
  fase;
- curvas de loss sólo como diagnóstico de optimización.

No habrá umbral de loss ni un corte post hoc sobre el error continuo. Si la
identificación falla, reportaremos dónde y cómo falla en lugar de redefinir el
criterio.

## Alcance de una conclusión positiva

Un resultado positivo demostraría interpolación dentro de una familia controlada
de Koopman: el JEPA recupera de los pares temporales tanto la estructura cíclica
como una tasa de decaimiento no trivial. Todavía no demostraría generalización a
otro tipo de observación, más fases, dinámica no lineal, ni datos reales.
