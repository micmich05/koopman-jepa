# Configuraciones

## Protocolo principal actual

[`stage3_three_dynamics_control.yaml`](stage3_three_dynamics_control.yaml)
define la comparación neuronal entre dinámica estática, cíclica e independiente:
misma arquitectura, mismas marginales, 10 seeds emparejadas y decisión basada
en identificación de acción y espectro.

## Soporte directo

- [`stage2_phase_observation_audit.yaml`](stage2_phase_observation_audit.yaml):
  auditoría del generador de observaciones compartidas.
- [`stage3_cyclic_multiseed_development.yaml`](stage3_cyclic_multiseed_development.yaml):
  robustez de la receta neuronal en el ciclo.
- [`stage3_cyclic_heldout.yaml`](stage3_cyclic_heldout.yaml): evaluación ya
  consumida del caso cíclico sobre nuevas emisiones.

## Historial técnico

Los demás archivos `stage3_cyclic_*` conservan cambios de un solo factor usados
para diagnosticar EMA, velocidad del predictor, checkpoint e inestabilidad de
escala. No son hipótesis científicas independientes.

Los archivos `paper_*` pertenecen a la reproducción parcial del antecedente de
invariantes con eigenvalue `1`; no forman parte del resultado principal sobre
operadores no triviales.
