import numpy as np
import pandas as pd
from pathlib import Path

base_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds")
audit_dir = base_dir / "Stage4_2_Independent_Model_Audit"
audit_dir.mkdir(parents=True, exist_ok=True)

print("Running Task 2: Focal Loss Gradient and Hessian Numerical Audit...")

# 1. Implemented objective function from train_heavy_rain_models.py
def focal_loss_implemented(y_true, z, gamma=2.0, alpha=0.75):
    p = 1.0 / (1.0 + np.exp(-z))
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    pt = np.where(y_true == 1, p, 1.0 - p)
    alpha_t = np.where(y_true == 1, alpha, 1.0 - alpha)
    focal_weight = alpha_t * ((1.0 - pt) ** gamma)
    grad = focal_weight * (p - y_true)
    hess = focal_weight * p * (1.0 - p)
    hess = np.clip(hess, 1e-5, None)
    return float(grad), float(hess)

# 2. True mathematical focal loss function
def true_focal_loss(y_true, z, gamma=2.0, alpha=0.75):
    p = 1.0 / (1.0 + np.exp(-z))
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    pt = p if y_true == 1 else (1.0 - p)
    alpha_t = alpha if y_true == 1 else (1.0 - alpha)
    return - alpha_t * ((1.0 - pt) ** gamma) * np.log(pt)

# 3. Exact analytical gradient of true focal loss with respect to logit z
def true_focal_grad_analytical(y_true, z, gamma=2.0, alpha=0.75):
    p = 1.0 / (1.0 + np.exp(-z))
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    if y_true == 1:
        # L1 = -alpha * (1-p)^gamma * ln(p)
        # dL1/dz = alpha * (1-p)^gamma * [ gamma * p * ln(p) + p - 1 ]
        return float(alpha * ((1.0 - p) ** gamma) * (gamma * p * np.log(p) + p - 1.0))
    else:
        # L0 = -(1-alpha) * p^gamma * ln(1-p)
        # dL0/dz = (1-alpha) * p^gamma * [ p - gamma * (1-p) * ln(1-p) ]
        return float((1.0 - alpha) * (p ** gamma) * (p - gamma * (1.0 - p) * np.log(1.0 - p)))

# 4. Exact analytical Hessian of true focal loss with respect to logit z
def true_focal_hess_analytical(y_true, z, gamma=2.0, alpha=0.75):
    p = 1.0 / (1.0 + np.exp(-z))
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    dp_dz = p * (1.0 - p)
    
    if y_true == 1:
        # G1 = alpha * (1-p)^gamma * [ gamma * p * ln(p) + p - 1 ]
        # Let u = (1-p)^gamma, v = gamma * p * ln(p) + p - 1
        # du/dp = -gamma * (1-p)^(gamma-1)
        # dv/dp = gamma * (ln(p) + 1) + 1 = gamma*ln(p) + gamma + 1
        # dG1/dp = alpha * [ du/dp * v + u * dv/dp ]
        u = (1.0 - p) ** gamma
        du_dp = -gamma * ((1.0 - p) ** (gamma - 1.0))
        v = gamma * p * np.log(p) + p - 1.0
        dv_dp = gamma * np.log(p) + gamma + 1.0
        dG1_dp = alpha * (du_dp * v + u * dv_dp)
        return float(dG1_dp * dp_dz)
    else:
        # G0 = (1-alpha) * p^gamma * [ p - gamma * (1-p) * ln(1-p) ]
        # Let u = p^gamma, v = p - gamma * (1-p) * ln(1-p)
        # du/dp = gamma * p^(gamma-1)
        # dv/dp = 1 - gamma * [ -ln(1-p) - 1 ] = 1 + gamma*ln(1-p) + gamma
        u = p ** gamma
        du_dp = gamma * (p ** (gamma - 1.0))
        v = p - gamma * (1.0 - p) * np.log(1.0 - p)
        dv_dp = 1.0 + gamma * np.log(1.0 - p) + gamma
        dG0_dp = (1.0 - alpha) * (du_dp * v + u * dv_dp)
        return float(dG0_dp * dp_dz)

# 5. Numerical finite-difference evaluation (Central Difference)
def numerical_grad_hess(y_true, z, eps=1e-5):
    l_plus = true_focal_loss(y_true, z + eps)
    l_minus = true_focal_loss(y_true, z - eps)
    l_center = true_focal_loss(y_true, z)
    
    num_grad = (l_plus - l_minus) / (2.0 * eps)
    num_hess = (l_plus - 2.0 * l_center + l_minus) / (eps ** 2)
    return float(num_grad), float(num_hess)

# Run test grid across diverse logit margins z
test_grid_z = [-8.0, -6.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
audit_records = []

for y in [1, 0]:
    for z in test_grid_z:
        p = 1.0 / (1.0 + np.exp(-z))
        
        g_impl, h_impl = focal_loss_implemented(y, z)
        g_ana = true_focal_grad_analytical(y, z)
        h_ana = true_focal_hess_analytical(y, z)
        g_num, h_num = numerical_grad_hess(y, z)
        
        diff_grad_num_ana = abs(g_num - g_ana)
        diff_hess_num_ana = abs(h_num - h_ana)
        discrepancy_impl_vs_true_grad = abs(g_impl - g_ana)
        discrepancy_impl_vs_true_hess = abs(h_impl - h_ana)
        
        audit_records.append({
            "label_y": y,
            "raw_margin_z": z,
            "probability_p": round(p, 6),
            "implemented_grad": round(g_impl, 8),
            "implemented_hess": round(h_impl, 8),
            "true_analytical_grad": round(g_ana, 8),
            "true_analytical_hess": round(h_ana, 8),
            "numerical_diff_grad": round(g_num, 8),
            "numerical_diff_hess": round(h_num, 8),
            "diff_ana_vs_num_grad": round(diff_grad_num_ana, 10),
            "diff_ana_vs_num_hess": round(diff_hess_num_ana, 10),
            "analytical_verification_status": "PASSED" if diff_grad_num_ana < 1e-4 and diff_hess_num_ana < 1e-3 else "FAILED",
            "impl_formulation_type": "Pseudo_Focal_Surrogate" if discrepancy_impl_vs_true_grad > 1e-5 else "Exact_Focal",
            "is_true_hessian_negative": True if h_ana < 0.0 else False,
            "is_implemented_hessian_strictly_positive": True if h_impl > 0.0 else False
        })

df_audit = pd.DataFrame(audit_records)
df_audit.to_csv(audit_dir / "focal_loss_gradient_hessian_tests.csv", index=False)
print("Saved focal_loss_gradient_hessian_tests.csv")
print("\nSample Rows:")
print(df_audit[["label_y", "raw_margin_z", "probability_p", "implemented_grad", "true_analytical_grad", "implemented_hess", "true_analytical_hess", "is_true_hessian_negative", "is_implemented_hessian_strictly_positive"]].to_string())

