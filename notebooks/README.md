# Índice de experimentos

Este directorio conserva el orden experimental y los resultados completos. Los
notebooks ejecutados incluyen sus outputs, gráficos y un análisis escrito. Un
`PASS` o `FAIL` siempre se refiere al gate predeclarado de ese notebook; no
compara directamente protocolos distintos.

## Recorrido recomendado

1. [`phase0_identity_reproduction.ipynb`](phase0_identity_reproduction.ipynb):
   reproducción mínima del mecanismo invariante.
2. [`paper_dataset_audit.ipynb`](paper_dataset_audit.ipynb): reconstrucción y
   límites de identificabilidad del dataset del paper.
3. [`paper_linear_identity_heldout_smoke.ipynb`](paper_linear_identity_heldout_smoke.ipynb)
   y [`paper_linear_random_heldout_smoke.ipynb`](paper_linear_random_heldout_smoke.ipynb):
   comparación held-out de predictores lineales.
4. [`paper_supervised_separability_probe.ipynb`](paper_supervised_separability_probe.ipynb):
   control que localiza la brecha MLP en el entrenamiento autosupervisado.
5. [`stage1_four_phase_koopman_oracle.ipynb`](stage1_four_phase_koopman_oracle.ipynb):
   validación algebraica del ciclo de cuatro fases.
6. [`stage2_phase_dynamics_oracle.ipynb`](stage2_phase_dynamics_oracle.ipynb):
   comparación estática/cíclica/independiente con marginales idénticas.

## Registro completo

### Etapa 0 — Reproducción y diagnóstico del paper

| Notebook | Estado | Pregunta principal |
|---|---:|---|
| [`phase0_identity_reproduction.ipynb`](phase0_identity_reproduction.ipynb) | PASS mecanístico | ¿Aparece la solución invariante `λ=1` en el toy reducido? |
| [`paper_dataset_audit.ipynb`](paper_dataset_audit.ipynb) | Auditoría | ¿La reconstrucción de los 18 regímenes respeta la geometría publicada? |
| [`paper_overfit_smoke.ipynb`](paper_overfit_smoke.ipynb) | PASS | ¿Loss, optimizer y EMA pueden memorizar un batch sin colapsar? |
| [`paper_train_validation_smoke.ipynb`](paper_train_validation_smoke.ipynb) | PASS agregado | ¿La señal predictiva generaliza en una corrida corta? |
| [`paper_seed_stability_smoke.ipynb`](paper_seed_stability_smoke.ipynb) | FAIL | ¿Es estable la loss absoluta entre seeds? |
| [`paper_seed_stability_scale_invariant.ipynb`](paper_seed_stability_scale_invariant.ipynb) | PASS estrecho | ¿Es estable la mejora relativa a su baseline? |
| [`paper_linear_identity_heldout_smoke.ipynb`](paper_linear_identity_heldout_smoke.ipynb) | PASS | ¿El predictor identidad conserva su mecanismo en held-out? |
| [`paper_linear_random_control_smoke.ipynb`](paper_linear_random_control_smoke.ipynb) | PASS desarrollo | ¿El control Xavier entrena con el mismo encoder y batches? |
| [`paper_linear_random_heldout_smoke.ipynb`](paper_linear_random_heldout_smoke.ipynb) | FAIL formal | ¿El control aleatorio supera todos los umbrales held-out? |
| [`paper_linear_random_heldout_diagnostic.ipynb`](paper_linear_random_heldout_diagnostic.ipynb) | Diagnóstico | ¿Cuánto depende la pureza de K-means y qué clases fallan? |
| [`paper_mlp_clustering_development.ipynb`](paper_mlp_clustering_development.ipynb) | FAIL | ¿El MLP publicado aproximado conserva rango antes de clustering? |
| [`paper_mlp_one_hidden_development.ipynb`](paper_mlp_one_hidden_development.ipynb) | FAIL | ¿Una sola capa oculta corrige el rango? |
| [`paper_mlp_one_hidden_clustering_diagnostic.ipynb`](paper_mlp_one_hidden_clustering_diagnostic.ipynb) | Diagnóstico | ¿El checkpoint de mínimo error recupera la pureza publicada? |
| [`paper_mlp_two_stage_one_hidden_development.ipynb`](paper_mlp_two_stage_one_hidden_development.ipynb) | FAIL | ¿La lectura alternativa `6144→64→32` corrige el rango? |
| [`paper_mlp_two_stage_clustering_diagnostic.ipynb`](paper_mlp_two_stage_clustering_diagnostic.ipynb) | Diagnóstico | ¿La lectura de dos etapas mejora la pureza? |
| [`paper_mlp_low_lr_development.ipynb`](paper_mlp_low_lr_development.ipynb) | FAIL | ¿Un learning rate menor explica la brecha? |
| [`paper_mlp_medium_scale_development.ipynb`](paper_mlp_medium_scale_development.ipynb) | FAIL | ¿Cuatro veces más datos mejoran pureza y estabilidad? |
| [`paper_mlp_gradient_clip_probe.ipynb`](paper_mlp_gradient_clip_probe.ipynb) | FAIL | ¿Clipping global evita la divergencia tardía? |
| [`paper_mlp_step_decay_probe.ipynb`](paper_mlp_step_decay_probe.ipynb) | FAIL | ¿Step decay evita la divergencia tardía? |
| [`paper_supervised_separability_probe.ipynb`](paper_supervised_separability_probe.ipynb) | PASS control | ¿Dataset y CNN contienen señal separable suficiente? |

Conclusión de Etapa 0: el mecanismo lineal identidad se reproduce, pero no la
pureza MLP `65.48%` del paper. El control supervisado llega a `91.49%` de
accuracy y `88.05%` de pureza, por lo que la brecha queda localizada en la
receta u objetivo autosupervisado no publicado, no en ausencia de señal.

### Extensión Koopman no trivial

| Notebook | Estado | Resultado |
|---|---:|---|
| [`stage1_four_phase_koopman_oracle.ipynb`](stage1_four_phase_koopman_oracle.ipynb) | PASS | Recupera rango activo 3 y espectro `{-1,i,-i}` con error espectral medio `1.14e-15`. |
| [`stage2_phase_dynamics_oracle.ipynb`](stage2_phase_dynamics_oracle.ipynb) | PASS oracle | Con marginales idénticas recupera `{1,1,1}`, `{-1,i,-i}` y `{0,0,0}`; distingue error por muestra de error contra la media condicional. |

## Convenciones

- `paper_*`: reproducción, sensibilidades o controles asociados al trabajo
  AAAI 2026.
- `stage*`: extensión propia hacia modos de Koopman no triviales.
- `*_development`: sólo usa train/validation.
- `*_heldout_*`: consume una partición de test; sus thresholds no se cambian
  después de observarla.
- `*_diagnostic` y `*_probe`: análisis post-hoc o sensibilidad; no convierten
  un `FAIL` previo en `PASS`.

Los parámetros congelados están en [`../configs`](../configs), la lógica
reutilizable en [`../src/koopman_jepa`](../src/koopman_jepa) y los contratos de
reproducibilidad en [`../tests`](../tests).
