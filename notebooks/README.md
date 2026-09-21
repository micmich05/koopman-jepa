# Notebooks

Los cinco notebooks están ejecutados y cuentan una sola historia. Orden sugerido:

1. [`koopman_oracle.ipynb`](koopman_oracle.ipynb) — verifica exactamente
   $MA=AK$ y el espectro $\{-1,i,-i\}$ usando la fase verdadera.
2. [`observation_audit.ipynb`](observation_audit.ipynb) — confirma que las
   condiciones comparten las mismas ventanas y sólo difieren en el pairing.
3. [`three_dynamics_experiment.ipynb`](three_dynamics_experiment.ipynb) —
   resultado principal: estática, cíclica e independiente, 10 seeds cada una.
4. [`koopman_decay_generalization.ipynb`](koopman_decay_generalization.ipynb) —
   estima la intensidad continua $\rho$ en $K_\rho=\rho C$.
5. [`koopman_decay_baselines_validation.ipynb`](koopman_decay_baselines_validation.ipynb) —
   contextualiza el resultado contra seis referencias sobre una seed de validación.

El último notebook es exploratorio: el held-out no fue materializado y no se
presenta como comparación estadística final.
