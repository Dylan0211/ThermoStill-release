import math
import numpy as np
import importlib

def mae(pred: np.ndarray, target: np.ndarray):
    """Mean Absolute Error (MAE) for arrays of shape (B, L, C)."""
    return np.mean(np.abs(pred - target))

def mse(pred: np.ndarray, target: np.ndarray):
    """Mean Squared Error (MSE) for arrays of shape (B, L, C)."""
    return np.mean((pred - target) ** 2)

def rmse(pred: np.ndarray, target: np.ndarray):
    """Root Mean Squared Error (RMSE) for arrays of any shape."""
    return np.sqrt(np.mean((pred - target) ** 2))

def cv_rmse(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8):
    """
    Coefficient of Variation of the Root Mean Squared Error (CV-RMSE).
    CV-RMSE = RMSE / mean(target) (%)
    """
    rmse = np.sqrt(np.mean((pred - target) ** 2))
    mean_target = np.mean(target)
    return rmse / (mean_target + eps) * 100

def select_rc_model(args):
    module_name_map = {
        "R1C1": "rc_1r1c",
        "R2C1": "rc_2r1c",
        "R2C2": "rc_2r2c",
    }
    module_name = module_name_map[args.rc_model]
    module = importlib.import_module(f"models.grey_box.{module_name}")
    class_name = "R1C1" if "R1C1" in args.rc_model else ("R2C1" if "R2C1" in args.rc_model else "R2C2")
    model_cls = getattr(module, class_name)
    return model_cls(args)

def adjust_learning_rate(optimizer, epoch, lr, max_epochs, lradj="type1"):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if lradj == 'type1':
        # lr_adjust = {epoch: lr * (0.5 ** ((epoch - 1) // 50))}
        lr_arr = np.linspace(lr, lr * 0.1, max_epochs)
        lr_adjust = {epoch: lr for epoch, lr in enumerate(lr_arr)}
    elif lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif lradj == 'type3':
        lr_adjust = {epoch: lr if epoch < 3 else lr * (0.9 ** ((epoch - 3) // 1))}
    elif lradj == "cosine":
        lr_adjust = {epoch: lr /2 * (1 + math.cos(epoch / max_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
