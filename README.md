# Regularized Logistic Regression — Microchip Quality Classification

A from-scratch implementation of regularized logistic regression using only NumPy and Matplotlib. Built to develop a rigorous understanding of ML mathematics rather than rely on high-level libraries.

---

## Overview

Binary classification task: predicting whether a microchip passes quality assurance based on two test scores. Since the classes are not linearly separable, the model uses **degree-6 polynomial feature mapping** to learn a nonlinear decision boundary, combined with **L2 regularization** to prevent overfitting.

---

## Math

**Sigmoid:**
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Regularized cost function:**
$$J(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right] + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$$

The bias $b$ is intentionally excluded from the regularization term.

**Gradients:**
$$\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{m} \mathbf{X}^\top (\hat{\mathbf{y}} - \mathbf{y}) + \frac{\lambda}{m} \mathbf{w}, \qquad \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$

---

## Key Implementation Details

| Component | Detail |
|---|---|
| Feature mapping | Degree-6 polynomial → 27 features, bias handled separately |
| Regularization | L2 on weights only; $\lambda$ controls overfitting vs. underfitting |
| Optimization | Batch gradient descent, manually implemented |
| Decision boundary | Meshgrid → polynomial map → contour at $\hat{y} = 0.5$ |

---

## Project Structure

```
chip_quality_classifier/
├── chip_quality_classifier.ipynb   # Full implementation and results
├── data/                           # Microchip test dataset  link:https://drive.google.com/file/d/1QnGOWBsA_R7_bakwKyZSriGKcJciI_M4/view?usp=sharing  
|__utils.py                         #feature engineering and sigmoid
└── README.md
```

---

## Installation & Usage

```bash
pip install numpy matplotlib jupyter
jupyter notebook chip_quality_classifier.ipynb
```

---

## Results

The model learns a smooth, nonlinear boundary that correctly separates accepted from rejected chips. Training cost decreases monotonically, confirming stable gradient descent. The regularized boundary generalizes well without overfitting to the training set.

---

## Future Improvements

- Hyperparameter grid search over $\lambda$ and learning rate
- Train/validation/test split for formal generalization metrics
- Adaptive convergence stopping criterion
- Feature scaling prior to polynomial mapping
- Experiment with polynomial degrees to explore the bias–variance tradeoff

---

## License

MIT
