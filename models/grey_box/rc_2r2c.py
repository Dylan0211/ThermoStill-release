"""2R2C RC model."""

import torch
import torch.nn as nn

from models.grey_box.rc_shared import bounded_value, make_bounded_parameter


class R2C2(nn.Module):
    PARAM_BOUNDS = {
        "Ri": (1e-4, 0.2),
        "Re": (1e-4, 0.2),
        "Ci": (1e5, 1e8),
        "Ce": (1e5, 1e8),
        "Ai": (0.0, 0.2),
        "Ae": (0.0, 0.2),
        "hvac_gain": (1.0, 2e4),
    }
    PARAM_INIT = {
        "Ri": 7e-3,
        "Re": 1.2e-2,
        "Ci": 3.0e6,
        "Ce": 6.0e6,
        "Ai": 0.01,
        "Ae": 0.005,
        "hvac_gain": 450.0,
    }

    def __init__(self, args):
        super().__init__()
        self.delta_t = float(args.delta_t)
        self.mode = "cool"

        self.raw_Ri = make_bounded_parameter(self.PARAM_INIT["Ri"], *self.PARAM_BOUNDS["Ri"])
        self.raw_Re = make_bounded_parameter(self.PARAM_INIT["Re"], *self.PARAM_BOUNDS["Re"])
        self.raw_Ci = make_bounded_parameter(self.PARAM_INIT["Ci"], *self.PARAM_BOUNDS["Ci"])
        self.raw_Ce = make_bounded_parameter(self.PARAM_INIT["Ce"], *self.PARAM_BOUNDS["Ce"])
        self.raw_Ai = make_bounded_parameter(self.PARAM_INIT["Ai"], *self.PARAM_BOUNDS["Ai"])
        self.raw_Ae = make_bounded_parameter(self.PARAM_INIT["Ae"], *self.PARAM_BOUNDS["Ae"])
        self.raw_hvac_gain = make_bounded_parameter(self.PARAM_INIT["hvac_gain"], *self.PARAM_BOUNDS["hvac_gain"])

    @property
    def Ri(self):
        return bounded_value(self.raw_Ri, *self.PARAM_BOUNDS["Ri"])

    @property
    def Re(self):
        return bounded_value(self.raw_Re, *self.PARAM_BOUNDS["Re"])

    @property
    def Ci(self):
        return bounded_value(self.raw_Ci, *self.PARAM_BOUNDS["Ci"])

    @property
    def Ce(self):
        return bounded_value(self.raw_Ce, *self.PARAM_BOUNDS["Ce"])

    @property
    def Ai(self):
        return bounded_value(self.raw_Ai, *self.PARAM_BOUNDS["Ai"])

    @property
    def Ae(self):
        return bounded_value(self.raw_Ae, *self.PARAM_BOUNDS["Ae"])

    @property
    def roxP_hvac(self):
        return bounded_value(self.raw_hvac_gain, *self.PARAM_BOUNDS["hvac_gain"])

    def _state_update(self, t_in, t_env, t_out, hvac_u, solar):
        cooling_sign = -1.0 if self.mode == "cool" else 1.0
        q_hvac = cooling_sign * self.roxP_hvac * hvac_u

        dtin = ((t_env - t_in) / self.Ri + self.Ai * solar + q_hvac) / self.Ci
        dtenv = ((t_in - t_env) / self.Ri + (t_out - t_env) / self.Re + self.Ae * solar) / self.Ce
        next_t_in = t_in + self.delta_t * dtin
        next_t_env = t_env + self.delta_t * dtenv
        return next_t_in, next_t_env

    def forward(self, input_vec):
        return self.onestep_predict(input_vec) if input_vec.shape[1] == 1 else self.multistep_predict(input_vec)

    def onestep_predict(self, input_vec):
        t_in = input_vec[:, :, 0]
        t_out = input_vec[:, :, 1]
        hvac_u = input_vec[:, :, 2]
        solar = input_vec[:, :, 3]
        t_env = 0.5 * (t_in + t_out)
        next_t_in, _ = self._state_update(t_in, t_env, t_out, hvac_u, solar)
        return next_t_in.unsqueeze(-1)

    def multistep_predict(self, input_vec):
        _, seq_len, _ = input_vec.shape
        t_in = input_vec[:, :, 0]
        t_out = input_vec[:, :, 1]
        hvac_u = input_vec[:, :, 2]
        solar = input_vec[:, :, 3]

        predictions = []
        t_in_prev = t_in[:, 0]
        t_env_prev = 0.5 * (t_in[:, 0] + t_out[:, 0])
        for step in range(seq_len):
            t_in_prev, t_env_prev = self._state_update(
                t_in_prev,
                t_env_prev,
                t_out[:, step],
                hvac_u[:, step],
                solar[:, step],
            )
            predictions.append(t_in_prev)
        return torch.stack(predictions, dim=1).unsqueeze(-1)

    def describe_model(self, precision: int = 6):
        desc = (
            f"2R2C system (mode={self.mode}):\n"
            f"Ci*dTin/dt = (Tenv - Tin)/Ri - hvac_gain*u + Ai*sol\n"
            f"Ce*dTenv/dt = (Tin - Tenv)/Ri + (Tout - Tenv)/Re + Ae*sol\n"
            f"(Ri={self.Ri.item():.{precision}f}, Re={self.Re.item():.{precision}f}, "
            f"Ci={self.Ci.item():.{precision}f}, Ce={self.Ce.item():.{precision}f}, "
            f"Ai={self.Ai.item():.{precision}f}, Ae={self.Ae.item():.{precision}f}, "
            f"hvac_gain={self.roxP_hvac.item():.{precision}f}, dt={self.delta_t})"
        )
        print(desc)
        return desc
