# Notebooks

La lectura principal tiene cinco notebooks. Juntos responden una sola pregunta:
¿un JEPA temporal puede aprender un predictor lineal cuya acción y espectro
cambien con la dinámica, aun cuando las observaciones marginales sean iguales?

| Evidencia | Notebook | Resultado |
|---|---|---|
| La evaluación matemática funciona | [`stage1_four_phase_koopman_oracle.ipynb`](stage1_four_phase_koopman_oracle.ipynb) | Con las fases verdaderas recupera rango 3, \(MA=AK\), espectro `{-1,i,-i}` y rollouts hasta 8 pasos. |
| La dinámica es identificable sin cambiar las observaciones | [`stage2_phase_dynamics_oracle.ipynb`](stage2_phase_dynamics_oracle.ipynb) | Distingue estática, cíclica e independiente con espectros `{1,1,1}`, `{-1,i,-i}` y `{0,0,0}`. |
| El generador no filtra la respuesta por los marginales | [`stage2_phase_observation_audit.ipynb`](stage2_phase_observation_audit.ipynb) | Las tres condiciones reutilizan exactamente las mismas ventanas; sólo cambia el pairing temporal. |
| El caso neuronal cíclico es reproducible | [`stage3_cyclic_multiseed_development.ipynb`](stage3_cyclic_multiseed_development.ipynb) | La representación y el predictor recuperan el ciclo en 8/10 seeds bajo los criterios absolutos originales. |
| El predictor neuronal sigue la dinámica | [`stage3_three_dynamics_control.ipynb`](stage3_three_dynamics_control.ipynb) | Acción y espectro identifican la condición correcta en 10/10 seeds para cada una de las tres dinámicas. |

El último notebook contiene las figuras y la conclusión principal. Es un
experimento de desarrollo sobre validation; todavía no es una evaluación ciega
de todo el control de tres condiciones.

## Material secundario

- [`stage3_cyclic_heldout.ipynb`](stage3_cyclic_heldout.ipynb) confirma sobre
  muestras nuevas que las métricas del operador cíclico se mantienen en 8/10
  seeds. Su antiguo `FAIL` agregado corresponde a un umbral auxiliar de loss y
  no se usa para responder la hipótesis Koopman.
- [`phase0_identity_reproduction.ipynb`](phase0_identity_reproduction.ipynb) y
  los notebooks `paper_*` documentan la reproducción parcial del antecedente
  para eigenvalue `1`. Son contexto, no evidencia principal de la extensión.
- [Bitácora técnica completa](EXPERIMENT_LOG.md): todos los smokes,
  diagnósticos, gates históricos y decisiones de optimización.

Las configuraciones congeladas están en [`../configs`](../configs), la lógica
reutilizable en [`../src/koopman_jepa`](../src/koopman_jepa) y los tests de
reproducibilidad en [`../tests`](../tests).
