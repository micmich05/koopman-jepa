# Koopman-JEPA

Experimentos controlados para estudiar si una arquitectura JEPA temporal puede aprender
subespacios finitos invariantes bajo Koopman y sus dinámicas espectrales.

## Por dónde empezar

- [Research brief](RESEARCH_BRIEF.md): pregunta, matemática, hipótesis y gates.
- [Índice de notebooks](notebooks/README.md): recorrido recomendado y estado de
  cada experimento.
- [Especificación de replicación](docs/PAPER_REPLICATION_SPEC.md): qué publica
  el paper y qué supuestos locales fueron necesarios.

El código reutilizable vive en [`src/koopman_jepa`](src/koopman_jepa), los
protocolos congelados en [`configs`](configs) y sus contratos en [`tests`](tests).

## Progreso hacia Koopman no trivial

La Etapa 1 está validada en el notebook ejecutado
[`stage1_four_phase_koopman_oracle.ipynb`](notebooks/stage1_four_phase_koopman_oracle.ipynb).
Usando directamente las indicadoras centradas del ciclo de cuatro fases, el
ajuste por mínimos cuadrados recupera el span activo tridimensional y el
espectro `{-1, i, -i}`. El error espectral medio es `1.14e-15`, el error de
entrelazamiento `1.59e-15` y el máximo error de rollout hasta ocho pasos
`9.62e-15`. Esto valida orientación de matrices, restricción al subespacio
activo, matching espectral y left eigenvectors antes de introducir un encoder.

La Etapa 2A oracle también está validada en
[`stage2_phase_dynamics_oracle.ipynb`](notebooks/stage2_phase_dynamics_oracle.ipynb).
Con 1024 transiciones balanceadas por condición, las dinámicas estática,
cíclica e independiente tienen las mismas marginales pero recuperan
respectivamente los espectros `{1,1,1}`, `{-1,i,-i}` y `{0,0,0}`. El peor error
espectral es `3.33e-15`. En la condición independiente, el error contra una
realización es `1.00`, mientras que el error contra la media condicional es
`3.16e-16`; esa diferencia es incertidumbre irreducible, no un fallo.

La Etapa 2B también está validada en
[`stage2_phase_observation_audit.ipynb`](notebooks/stage2_phase_observation_audit.ipynb).
Las tres condiciones reutilizan exactamente las mismas ventanas marginales,
con diferencia máxima `0.00`, mientras el decoder de fase oracle alcanza
`100%` y recupera las tres matrices de transición sin error. Esto confirma que
la fase es observable pero la dinámica sólo se revela mediante el par temporal.

El primer smoke neuronal está ejecutado en
[`stage3_cyclic_neural_smoke.ipynb`](notebooks/stage3_cyclic_neural_smoke.ipynb)
y dio `FAIL`. El encoder sí aprendió fase: probe `100%`, rango efectivo `2.846`
y error de alineación `0.153`. Sin embargo, el predictor obtuvo error de
entrelazamiento `1.278` y error espectral máximo `1.074`.

El diagnóstico
[`stage3_cyclic_neural_smoke_diagnostic.ipynb`](notebooks/stage3_cyclic_neural_smoke_diagnostic.ipynb)
localiza la brecha. El operador post-hoc online→online recupera el espectro con
error máximo `0.037`, mientras las bases online y EMA difieren `0.812`. La
representación contiene la dinámica, pero el predictor del checkpoint temprano
no funciona como endomorfismo Koopman. El siguiente control reducirá únicamente
el momentum EMA para comprobar si el target rezagado causa esa discrepancia.

Ese control está ejecutado en
[`stage3_cyclic_neural_ema_fast_smoke.ipynb`](notebooks/stage3_cyclic_neural_ema_fast_smoke.ipynb)
y también dio `FAIL`. Bajar EMA de `0.99` a `0.90` redujo el desacople de bases
de `0.812` a `0.182` y el error de entrelazamiento de `1.278` a `0.759`, pero el
error espectral del predictor apenas cambió (`1.074→1.059`). El post-hoc online
permanece correcto (`0.045`). EMA era parte del problema, pero no basta; el
siguiente diagnóstico instrumentará la optimización del predictor antes de
cambiar su learning rate.

El protocolo científico está documentado en
[RESEARCH_BRIEF.md](RESEARCH_BRIEF.md). La implementación cubre la **Fase 0**,
una replicación mecanística reducida del caso de invariantes de Koopman
(`lambda = 1`), y una reconstrucción auditable de los 18 regímenes sintéticos
presentados por Ruiz-Morales et al. en AAAI 2026.

## Auditoría del dataset del paper

El notebook ejecutado
[`paper_dataset_audit.ipynb`](notebooks/paper_dataset_audit.ipynb) revisa la
geometría, espectros, autocorrelaciones, tendencias, pulsos y sensibilidades de
los 18 regímenes. La mecánica del generador pasa, pero el entrenamiento está
pausado por un hallazgo de identificabilidad: la estandarización por secuencia
convierte `Sine_MedFreq` y `Sine_LowAmp` en la misma distribución observable.

La especificación, los parámetros publicados y cada supuesto local están en
[`PAPER_REPLICATION_SPEC.md`](docs/PAPER_REPLICATION_SPEC.md).

También está implementada la arquitectura temporal publicada, con tests para
su geometría convolucional, predictor y actualización EMA. Como el texto y el
apéndice se contradicen en la proyección del encoder y la profundidad del MLP,
[`paper_model.py`](src/koopman_jepa/paper_model.py) conserva cada interpretación
como una variante con nombre. El paso mínimo de entrenamiento también está
validado: el loss euclídeo, los gradientes, la actualización del optimizer y el
EMA funcionan juntos sobre un batch. Aún no hay un resultado de entrenamiento
del modelo completo. La configuración local de sobreajuste y sus umbrales de
no-colapso ya están congelados en
[`paper_overfit_smoke.yaml`](configs/paper_overfit_smoke.yaml). El notebook
ejecutado [`paper_overfit_smoke.ipynb`](notebooks/paper_overfit_smoke.ipynb)
pasó el gate: la loss final fue 0.0252% de la inicial, la dispersión retuvo
99.1% y el rango efectivo final fue 12.08. Es una prueba de integración sobre
un batch memorizable, no un resultado comparable con el paper. La condición
corta con splits separados está en
[`paper_train_validation_smoke.yaml`](configs/paper_train_validation_smoke.yaml),
y el notebook ejecutado
[`paper_train_validation_smoke.ipynb`](notebooks/paper_train_validation_smoke.ipynb)
pasa el gate agregado: validation baja al 24.6% de su baseline y conserva rango
efectivo 20.93. Sin embargo, la brecha individual validation/train llega a
4.467 en la época 10. Por eso test sigue reservado. La política siguiente ya
está congelada en
[`paper_seed_stability_smoke.yaml`](configs/paper_seed_stability_smoke.yaml):
selecciona la menor validation loss que todavía respeta los controles de brecha
y no-colapso. El notebook ejecutado
[`paper_seed_stability_smoke.ipynb`](notebooks/paper_seed_stability_smoke.ipynb)
encuentra checkpoints válidos para seeds 0–4, pero el gate global falla: el CV
de las validation loss absolutas es 0.288 frente al límite 0.25. Los demás
criterios pasan. Como diagnóstico, el CV de los ratios validation/baseline es
0.210, pero no estaba predefinido y no convierte el resultado en PASS. Test
permanece cerrado. El protocolo siguiente ya está congelado en
[`paper_seed_stability_scale_invariant.yaml`](configs/paper_seed_stability_scale_invariant.yaml):
repite la misma condición con seeds nuevas 5–9 y usa el CV de los ratios
validation/baseline como criterio de variabilidad, manteniendo todos los demás
límites. La loss absoluta seguirá reportándose sólo como diagnóstico. Este
protocolo fue ejecutado en
[`paper_seed_stability_scale_invariant.ipynb`](notebooks/paper_seed_stability_scale_invariant.ipynb)
y pasa por margen estrecho: CV relativo 0.239 frente al máximo 0.25, con CV
absoluto diagnóstico 0.251. Las cinco seeds producen checkpoints elegibles y
los controles de brecha, dispersión y rango pasan. Esto habilita diseñar y
congelar la evaluación held-out. El protocolo lineal-identidad ya está fijado
en
[`paper_linear_identity_heldout_smoke.yaml`](configs/paper_linear_identity_heldout_smoke.yaml),
y la captura/reproducción exacta del checkpoint ya está implementada y cubierta
por tests. Las métricas del operador y el gate conjunto también están
implementados y validados con casos sintéticos. El notebook de evaluación ya
fue ejecutado en
[`paper_linear_identity_heldout_smoke.ipynb`](notebooks/paper_linear_identity_heldout_smoke.ipynb).
Los cinco replays fueron exactos y el gate held-out pasó en las cinco seeds:
peor error a identidad 1.853%, peor antisimetría 0.740%, peor acción sobre
centroides 1.957%, mínimo 31 autovalores cerca de 1 y rango efectivo mínimo
17.963. La acción sobre centroides pasa por margen estrecho y test queda
consumido para este protocolo. Este smoke test todavía no reproduce la escala
ni todos los experimentos del paper.

El siguiente control pareado fue congelado y ejecutado en
[`paper_linear_random_control_smoke.yaml`](configs/paper_linear_random_control_smoke.yaml):
repite seeds 5–9 con encoder y minibatches idénticos, cambiando únicamente
`M_0` de identidad a Xavier uniforme. El notebook train/validation ejecutado
[`paper_linear_random_control_smoke.ipynb`](notebooks/paper_linear_random_control_smoke.ipynb)
reprodujo exactamente los cinco checkpoints identidad, verificó igualdad
exacta de los encoders iniciales entre condiciones y seleccionó epoch 10 para
las cinco corridas Xavier. El gate de desarrollo pasó: CV relativo 0.204,
peor validation/baseline 0.061, peor brecha validation/train 2.059, rango
efectivo mínimo 12.29 y retención de dispersión mínima 0.704. Frente a
identidad, el peor factor de mejora relativa fue 0.229, aunque la loss absoluta
random fue entre 1.129x y 1.291x mayor. Las matrices random quedaron lejos de
identidad (error relativo mínimo 138.4%) y densas (fracción off-diagonal mínima
98.2%). Test no fue construido ni consultado. El resultado permite congelar
estos epochs para una comparación posterior sobre el test smoke ya consumido;
no constituye todavía evidencia held-out del control ni una reproducción a
escala completa.

El protocolo held-out pareado fue congelado y ejecutado en
[`paper_linear_random_heldout_smoke.yaml`](configs/paper_linear_random_heldout_smoke.yaml).
Obliga a reproducir los diez checkpoints antes de reconstruir test y compara
las dos condiciones mediante error predictivo normalizado, pureza K-means y
rango efectivo. Los thresholds se fijaron antes de observar el test random. La
partición ya fue consumida por la evaluación identidad, por lo que este control
es predeclarado pero no una segunda prueba ciega independiente. El notebook
ejecutado está en
[`paper_linear_random_heldout_smoke.ipynb`](notebooks/paper_linear_random_heldout_smoke.ipynb).
Los diez replays y el pareo inicial pasaron, pero el gate global dio `FAIL`:
seed 7 obtuvo pureza random `47.92%` frente al mínimo absoluto `50%` (69 de 144
asignaciones, tres menos que el corte). Todos los demás criterios pasaron: peor
ratio de error predictivo 1.470, mínima retención de pureza 0.974, rango random
mínimo 12.25 y mínima retención de rango 0.682. La pureza random media fue
50.28%, ligeramente mayor que el 49.44% de identidad, pero el threshold
predeclarado exige que todas las seeds pasen y no se modifica después de ver
test. La partición queda consumida también para random.

El análisis exploratorio posterior está ejecutado en
[`paper_linear_random_heldout_diagnostic.ipynb`](notebooks/paper_linear_random_heldout_diagnostic.ipynb).
Reproduce exactamente los conteos del gate y muestra que la pureza depende de
la inicialización de K-means incluso con `n_init=20`: sobre las condiciones y
seeds observadas, identidad abarca 43.75–55.56% y random 44.44–56.94%. Los
regímenes random con menor recall medio son `ar_pos_strong` (10%), `ma_pos`
(17.5%) y `pulses_sparse` (22.5%); tendencias y ondas cuadradas llegan a 100%.
La pareja de senos indistinguible permanece dentro del par sólo 53.8% de las
veces en random, así que explica parte, pero no todo, del clustering débil. Este
diagnóstico no modifica el `FAIL` ni crea una nueva decisión sobre test.

El siguiente experimento está congelado, pero todavía no ejecutado, en
[`paper_mlp_clustering_development.yaml`](configs/paper_mlp_clustering_development.yaml).
Usa una realización nueva (`base_seed=1`), 64/32 ejemplos train/validation por
régimen, seeds de modelo 10–14 y el MLP `32→64→64→32`. La pureza de validation
se promediará sobre 20 `random_state` de K-means, todos con `n_init=20`. El gate
de desarrollo exige media global ≥60%, peor seed ≥55%, CV entre seeds ≤10% y
desviación intra-seed ≤3 puntos. Test no se construirá en esta etapa.
Las métricas agregadas y el gate están implementados y cubiertos por casos
sintéticos. El notebook de desarrollo MLP fue ejecutado sólo con
`train`/`validation` y dio `FAIL` antes de clustering: seeds 10, 13 y 14
seleccionaron épocas 5, 4 y 5, mientras que seeds 11 y 12 no alcanzaron el rango
efectivo mínimo `4.0` (mejores valores `2.71` y `3.48`). Predicción, brecha y
dispersión sí pasaron. Para no sesgar el resultado, K-means fue omitido en vez de
medirse sólo sobre las tres seeds aceptadas. Test no fue construido ni
consultado. La próxima condición ya está congelada en
[`paper_mlp_one_hidden_development.yaml`](configs/paper_mlp_one_hidden_development.yaml):
cambia únicamente el predictor a la lectura de una capa oculta `32→64→32`; no
relaja el gate observado. El notebook
[`paper_mlp_one_hidden_development.ipynb`](notebooks/paper_mlp_one_hidden_development.ipynb)
ya fue ejecutado. Mejoró de 3/5 a 4/5 checkpoints elegibles, pero seed 11
quedó en rango efectivo `2.10` pese a reducir el error a `0.8%` del inicial; el
gate predictivo continuó en `FAIL` y clustering fue omitido. El próximo paso es
un diagnóstico post-hoc de pureza en validation usando el checkpoint de mínimo
error de cada seed; no habilita test ni cambia el fallo formal. Está ejecutado en
[`paper_mlp_one_hidden_clustering_diagnostic.ipynb`](notebooks/paper_mlp_one_hidden_clustering_diagnostic.ipynb).
El diagnóstico ya ejecutado obtiene pureza global `50.76%` (rango por seed
`47.60–53.59%`), lejos del `60%` local y del `65.48%` publicado. La variación es
baja (CV entre seeds `0.049`, peor sd de K-means `1.35%`), por lo que no parece
un accidente de inicialización. Seed 11, aun con rango efectivo `2.10`, es la
segunda mejor en pureza (`52.62%`): el filtro de rango no ocultaba una
reproducción exitosa.

La siguiente sensibilidad está congelada en
[`paper_mlp_two_stage_one_hidden_development.yaml`](configs/paper_mlp_two_stage_one_hidden_development.yaml).
Cambia únicamente el encoder directo `6144→32` por la lectura reconciliada del
apéndice `6144→64→32`. El notebook
[`paper_mlp_two_stage_one_hidden_development.ipynb`](notebooks/paper_mlp_two_stage_one_hidden_development.ipynb)
y ya fue ejecutado. Dio `FAIL` antes de clustering: sólo seed 10 alcanzó el
rango mínimo (checkpoint en época 4, rango `4.53`); las otras cuatro quedaron
en `1.22–2.91`. El encoder adicional empeoró la concentración dimensional. El
diagnóstico de pureza para las cinco seeds está ejecutado en
[`paper_mlp_two_stage_clustering_diagnostic.ipynb`](notebooks/paper_mlp_two_stage_clustering_diagnostic.ipynb).
Obtiene sólo `47.90%` de pureza global, `2.86` puntos
menos que el encoder directo; el rango de las medias por seed es
`45.23–51.22%`. La lectura de dos etapas queda descartada como explicación de
la brecha con el paper.

La siguiente sensibilidad está congelada en
[`paper_mlp_low_lr_development.yaml`](configs/paper_mlp_low_lr_development.yaml):
vuelve al mejor encoder directo y cambia únicamente el learning rate de
`3e-4` a `1e-4`. El notebook
[`paper_mlp_low_lr_development.ipynb`](notebooks/paper_mlp_low_lr_development.ipynb)
y ya fue ejecutado. La pureza sube sólo de `50.76%` a `51.13%`, mientras el
gate predictivo sigue en `FAIL` y sólo una seed supera rango 4. El learning rate
menor no explica la brecha con el paper.

El próximo piloto está congelado en
[`paper_mlp_medium_scale_development.yaml`](configs/paper_mlp_medium_scale_development.yaml):
usa la mejor configuración base y escala de 64/32 a 256/64 secuencias
train/validation por régimen. Es un diagnóstico intermedio de escala, todavía
lejos de los miles de ejemplos por régimen del paper. Su notebook
[`paper_mlp_medium_scale_development.ipynb`](notebooks/paper_mlp_medium_scale_development.ipynb)
y ya fue ejecutado. Con 4× más train, la pureza queda en `49.92%`, sin mejora
frente al `50.76%` pequeño. Los checkpoints óptimos ocurren tras un número de
updates similar, pero el error explota después; esto apunta a una receta de
optimización incompleta más que a falta de datos.

Una nueva auditoría de las fuentes oficiales (AAAI y arXiv v2, 2026-09-20) no
encontró un enlace a código ni detalles sobre optimizer, schedule, clipping o
presupuesto de updates. Un diagnóstico repetido de seed 10 en escala media
confirmó explosión de gradientes: las normas online/predictor pasan de
`0.037/0.076` en época 2 a `1655.7/3774.8` en época 20, mientras el error de
validation pasa de `0.015×` a `1014×` el baseline. El próximo experimento será
una estabilización local predeclarada con clipping global de norma `1.0`; no se
presentará como receta literal del paper y seguirá sin consultar test.

Ese probe ya fue ejecutado en
[`paper_mlp_gradient_clip_probe.ipynb`](notebooks/paper_mlp_gradient_clip_probe.ipynb)
y dio `FAIL`. El clip se activó, pero el error final llegó a `1843×` el
baseline, peor que `1014×` sin clipping. El checkpoint temprano conserva rango
`7.88` y pureza `51.00%`, prácticamente idéntica al `51.07%` previo. No se
expandirá a cinco seeds. El próximo probe reducirá el learning rate después del
mínimo temprano; seguirá siendo una estabilización local, no una receta
atribuida al paper.

El probe de step decay también está ejecutado en
[`paper_mlp_step_decay_probe.ipynb`](notebooks/paper_mlp_step_decay_probe.ipynb)
y dio `FAIL`. Bajar de `3e-4` a `3e-5` después de la época 2 reduce la explosión
final de `1014×` a `716×`, pero no la elimina. El checkpoint sigue en época 2,
con rango `7.87` y pureza `51.07%`: estabilizar parcialmente la cola no recupera
los `65.48%` publicados. No se expandirá a cinco seeds ni se seguirá ajustando
stabilizers como si fueran una explicación del clustering faltante.

El control supervisado ejecutado
[`paper_supervised_separability_probe.ipynb`](notebooks/paper_supervised_separability_probe.ipynb)
sí dio `PASS`: `91.49%` de accuracy, `96.44%` al fusionar el par de senos
observacionalmente idéntico y `88.05% ± 0.43%` de pureza K-means en los
embeddings supervisados. Por lo tanto, el dataset reconstruido y el encoder
convolucional tienen capacidad suficiente; la brecha del JEPA local (~`51%`)
queda localizada en la receta/objetivo autosupervisado no especificado o en
otro detalle de implementación ausente. Esto no convierte el control
supervisado en una reproducción: sólo descarta separabilidad y capacidad como
explicaciones principales.

Las dos condiciones de preprocesamiento quedaron congeladas como:

- [`paper_literal.yaml`](configs/paper_literal.yaml): estandarización por
  secuencia, fiel al texto publicado;
- [`paper_amplitude_preserving.yaml`](configs/paper_amplitude_preserving.yaml):
  una sensibilidad con media y desviación globales ajustadas sólo sobre train,
  que conserva diferencias relativas de amplitud.

## Qué comprueba la Fase 0

- Regímenes sintéticos temporalmente persistentes e inmiscibles.
- Encoder temporal online y target encoder actualizado por EMA.
- Predictor lineal con inicialización identidad o aleatoria.
- Diagnósticos de colapso y rango efectivo.
- Separación de regímenes en test.
- Acción y espectro del predictor sobre el span de los centroides latentes.

No se considera suficiente una loss predictiva baja ni una visualización t-SNE.

## Instalación

El proyecto usa [uv](https://docs.astral.sh/uv/) para fijar un entorno reproducible:

```bash
uv sync --extra dev
```

## Tests

```bash
uv run pytest
uv run ruff check .
```

## Ejecutar la Fase 0

Predictor inicializado en identidad:

```bash
uv run koopman-jepa-phase0 \
  --config configs/phase0_smoke.yaml \
  --predictor-init identity \
  --seed 0
```

Control con inicialización aleatoria:

```bash
uv run koopman-jepa-phase0 \
  --config configs/phase0_smoke.yaml \
  --predictor-init random \
  --seed 0
```

Cada ejecución crea un directorio bajo `runs/phase0/` con configuración, historia,
métricas, checkpoint y figuras. `runs/` se mantiene fuera de Git para no versionar
artefactos pesados.

## Gate 0

No se avanza al ciclo de cuatro fases hasta comprobar que:

1. la representación no colapsa;
2. los regímenes son separables fuera de muestra;
3. el predictor identidad actúa cerca de la identidad sobre el span latente activo;
4. entendemos cómo cambia el resultado con inicialización aleatoria.
