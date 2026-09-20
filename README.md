# Koopman-JEPA

Experimentos controlados para estudiar si una arquitectura JEPA temporal puede aprender
subespacios finitos invariantes bajo Koopman y sus dinámicas espectrales.

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
un batch memorizable, no un resultado comparable con el paper. El próximo paso
es una corrida corta con train y validation separados.

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
