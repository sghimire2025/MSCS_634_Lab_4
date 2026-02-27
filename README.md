# MSCS 634 - Lab 4: Regression Analysis with Regularization Techniques (Diabetes Dataset)

**Name:** Suresh Ghimire  
**Course:** MSCS 634 - Advanced Big Data and Data Mining  
**Professor:** Satish Penmatsa  
**Dataset:** `sklearn.datasets.load_diabetes()` (Diabetes dataset)

## Overview
This lab explores regression techniques and regularization methods using the Diabetes dataset from scikit-learn. The notebook implements and compares:

- Simple Linear Regression (single feature: `bmi`)
- Multiple Linear Regression (all features)
- Polynomial Regression (degrees 1-5 on `bmi`)
- Ridge Regression
- Lasso Regression

Models are evaluated using:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R2 (R-squared)

---

## Project Structure

```text
.
├── MSCS_634_Lab_4.ipynb
├── README.md
└── images/
    ├── step2_simple_lr_bmi_vs_target.png
    ├── step2_actual_vs_pred_simple.png
    ├── step3_actual_vs_pred_multiple.png
    ├── step4_polynomial_curve_degree_2.png
    ├── step4_polynomial_fits_different_degrees.png
    ├── step5_ridge_actual_vs_pred.png
    ├── step5_lasso_actual_vs_pred.png
    ├── step6_model_comparison_r2.png
    └── step6_model_comparison_rmse.png
```

---

## Setup and Run

### 1) Install dependencies
If using `uv`:

```bash
uv sync
```

Then run Jupyter Lab:

```bash
uv run jupyter lab
```

### 2) Open the notebook
Open `MSCS_634_Lab_4.ipynb` and run the cells in order.

---

## Lab Steps Summary

### Step 1: Data Preparation
- Loaded the Diabetes dataset from `sklearn.datasets`
- Converted data into a pandas DataFrame
- Explored dataset structure, summary statistics, and feature distribution
- Checked for missing values (none found)
- Examined correlations to support feature selection

**Key note:** The dataset contains 442 samples, 10 features, and 1 target variable.

---

### Step 2: Simple Linear Regression (BMI only)
- Used `bmi` as the independent variable
- Split data into training and testing sets
- Trained a Simple Linear Regression model
- Evaluated using MAE, MSE, RMSE, and R2
- Visualized:
  - BMI vs target with regression line
  - Actual vs Predicted scatter plot

**Short finding:** BMI shows a positive relationship with the target, but using BMI alone leaves noticeable prediction error.

#### Image placeholders

![Step 2 - Simple Linear Regression (BMI vs Target)](images/step2_simple_lr_bmi_vs_target.png)

![Step 2 - Actual vs Predicted (Simple Linear Regression)](images/step2_actual_vs_pred_simple.png)

---

### Step 3: Multiple Linear Regression (All Features)
- Used all 10 input features
- Trained a Multiple Linear Regression model
- Compared performance against the simple linear model
- Visualized actual vs predicted and residual patterns

**Short finding:** Multiple Linear Regression improved performance compared to the single-feature model because it used more information from the dataset.

#### Image placeholders
![Step 3 - Actual vs Predicted (Multiple Linear Regression)](images/step3_actual_vs_pred_multiple.png)

<!-- Optional: Add residual plot screenshot here if you saved it -->
<!-- ![Step 3 - Residual Plot (Multiple Linear Regression)](images/step3_residual_plot_multiple.png) -->

---

### Step 4: Polynomial Regression
- Extended the `bmi` feature using polynomial features
- Trained polynomial regression models for multiple degrees (1-5)
- Compared performance to show underfitting vs overfitting behavior
- Visualized the degree-2 polynomial curve and polynomial fits for different degrees

**Short finding:** Polynomial regression increased flexibility, but using only BMI still limited performance. Higher degrees increased complexity and overfitting risk.

#### Image placeholders
![Step 4 - Polynomial Curve (Degree 2)](images/step4_polynomial_curve_degree_2.png)

![Step 4 - Polynomial Fits for Different Degrees](images/step4_polynomial_fits_different_degrees.png)

---

### Step 5: Regularization with Ridge and Lasso Regression
- Applied Ridge and Lasso regression using all features
- Used standardized inputs (`StandardScaler`) in a pipeline
- Compared Ridge vs Lasso coefficients
- Tested multiple alpha values
- Visualized actual vs predicted plots for Ridge and Lasso
- Selected the best Ridge/Lasso results and added them to the final comparison table

**Short findings:**
- Lasso (alpha = 1.0) performed slightly better than Ridge (alpha = 1.0) in the reported metrics.
- Ridge shrinks all coefficients, while Lasso can set coefficients to zero (feature selection behavior).
- Very large alpha values can reduce performance, especially for Lasso.

#### Image placeholders
![Step 5 - Actual vs Predicted (Ridge)](images/step5_ridge_actual_vs_pred.png)

![Step 5 - Actual vs Predicted (Lasso)](images/step5_lasso_actual_vs_pred.png)

---

### Step 6: Model Comparison and Analysis
- Combined all model results in a final comparison table
- Compared models using R2 and RMSE
- Visualized model comparison with bar charts
- Summarized observations about performance and regularization

**Final conclusion (from notebook results):**
- **Best model:** Lasso Regression (`alpha=1.0`)
- **Next best:** Ridge Regression (`alpha=100.0`) and Multiple Linear Regression
- **Lower-performing models:** Simple Linear Regression and Polynomial Regression using only BMI

#### Image placeholders
![Step 6 - Model Comparison by R2](images/step6_model_comparison_r2.png)

![Step 6 - Model Comparison by RMSE](images/step6_model_comparison_rmse.png)

---

## Final Results (Notebook Summary)

| Model | MAE | MSE | RMSE | R2 |
|---|---:|---:|---:|---:|
| Simple Linear Regression (bmi) | 52.259976 | 4061.825928 | 63.732456 | 0.233350 |
| Multiple Linear Regression (all features) | 42.794095 | 2900.193628 | 53.853446 | 0.452603 |
| Polynomial Regression (bmi, degree=1) | 52.259976 | 4061.825928 | 63.732456 | 0.233350 |
| Ridge Regression (alpha=100.0) | 43.250653 | 2858.224287 | 53.462363 | 0.460524 |
| Lasso Regression (alpha=1.0) | 42.802984 | 2824.568094 | 53.146666 | 0.466877 |

---

## Notes
- The Diabetes dataset used in scikit-learn is already preprocessed/scaled, but standardization was still applied for Ridge/Lasso as part of best practice for regularization.
- Some plot cells in the notebook include `plt.savefig(...)` to automatically save screenshots into the `images/` folder.
- If an image is not available yet, keep the placeholder and add the screenshot later before submission.

---

## How to Save Plot Images in Notebook (Reminder)
Use this pattern before `plt.show()`:

```python
plt.tight_layout()
plt.savefig("images/your_plot_name.png", dpi=300, bbox_inches="tight")
plt.show()
```

Create the folder once near the top of the notebook:

```python
import os
os.makedirs("images", exist_ok=True)
```
