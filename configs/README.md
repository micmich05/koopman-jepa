# Configuraciones

- [`observation_audit.yaml`](observation_audit.yaml): emisión y checks de
  integridad de los datos.
- [`three_dynamics.yaml`](three_dynamics.yaml): prueba principal de los tres
  operadores, 10 seeds por condición.
- [`koopman_decay_generalization.yaml`](koopman_decay_generalization.yaml):
  familia $K_\rho=\rho C$, cinco valores de $\rho$ y 10 seeds.
- [`koopman_decay_baselines.yaml`](koopman_decay_baselines.yaml): métodos de la
  comparación exploratoria; sólo se ejecutó la seed 101 sobre validación.

Los umbrales de identificación se fijaron antes de ejecutar cada comparación.
No hay umbrales de loss usados para decidir éxito o fracaso.
