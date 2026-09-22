# Configurations

- [`observation_audit.yaml`](observation_audit.yaml) defines the observation
  process and dataset-integrity checks.
- [`three_dynamics.yaml`](three_dynamics.yaml) defines the static, cyclic, and
  independent JEPA experiment with 10 seeds per condition.
- [`koopman_decay_generalization.yaml`](koopman_decay_generalization.yaml)
  defines the family $K_\rho=\rho C$ at five persistence values with 10 seeds
  per value.

Identification criteria were fixed before each experiment. Training loss is
never used as an operator-identification threshold.
