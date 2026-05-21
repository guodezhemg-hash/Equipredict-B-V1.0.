# EquiPredict-B:V1.0

This repository contains a compact Python implementation of the calculation core used in EquiPredict-B, a chemical reaction network model for estimating equilibrium flavor-compound concentrations in Chinese Baijiu.

The model describes the transition from an initial concentration state to a thermodynamic equilibrium state through a reaction-extent formulation. Given an initial concentration vector, a stoichiometric coefficient matrix, and reaction equilibrium constants, the calculation updates compound concentrations, evaluates reaction quotients, and solves the reaction extents by nonlinear least-squares minimization.



## Model Formulation

For initial concentration vector `C0`, stoichiometric coefficient matrix `V`, and reaction extent vector `xi`, the predicted concentration vector is calculated as:

```python
C = C0 + V @ xi
```

The logarithmic reaction quotient is evaluated from the current concentration vector:

```python
logQ = V.T @ np.log(C)
```

The thermodynamic residual used for least-squares optimization is:

```python
residual = logQ - logK
```

where `logK` is the logarithmic form of the apparent equilibrium constants.

To improve numerical efficiency, the analytical Jacobian of the residual vector with respect to the reaction extent vector is supplied:

```python
J = V.T @ (V / C[:, None])
```

These expressions are implemented in [calculation_core.py](calculation_core.py).

## Least-Squares Example

The repository includes a small artificial example showing how the residual and analytical Jacobian can be passed to `scipy.optimize.least_squares`:

```python
result = least_squares(
    residual_fn,
    x0=initial_xi,
    jac=jacobian_fn,
)
```

The example uses generic arrays only. It is intended to demonstrate the reaction-extent-guided residual minimization procedure.

## Included

- concentration update from reaction extent,
- logarithmic reaction quotient calculation,
- thermodynamic residual construction,
- analytical Jacobian construction,
- a toy nonlinear least-squares example.


## Quick Check

Use Python 3.10 or later.

```bash
pip install -r requirements.txt
python calculation_core.py
```

Expected output includes:

```text
solver success = True
solver xi = [0.72828584]
```

Small numerical differences may occur across Python, NumPy, or SciPy versions.



## Citation

If this repository is referenced, please cite the associated EquiPredict-B manuscript after publication.

## Reuse

The code is provided to describe the calculation strategy. For reuse beyond viewing this repository, please contact the authors.
