"""1R1C RC model."""

import torch
import torch.nn as nn

from models.grey_box.rc_shared import bounded_value, make_bounded_parameter


class R1C1(nn.Module):
    PARAM_BOUNDS = {
        "R": (1e-4, 0.2),
        "C": (1e5, 1e8),
        "A_eff": (0.0, 0.2),
        "hvac_gain": (1.0, 2e4),
    }
    PARAM_INIT = {
        "R": 7e-3,
        "C": 3.5e6,
        "A_eff": 0.01,
        "hvac_gain": 450.0,
    }

    def __init__(self, args):
        super().__init__()
        self.delta_t = float(args.delta_t)
        self.mode = "cool"

        self.raw_R = make_bounded_parameter(self.PARAM_INIT["R"], *self.PARAM_BOUNDS["R"])
        self.raw_C = make_bounded_parameter(self.PARAM_INIT["C"], *self.PARAM_BOUNDS["C"])
        self.raw_A_eff = make_bounded_parameter(self.PARAM_INIT["A_eff"], *self.PARAM_BOUNDS["A_eff"])
        self.raw_hvac_gain = make_bounded_parameter(self.PARAM_INIT["hvac_gain"], *self.PARAM_BOUNDS["hvac_gain"])

    @property
    def R(self):
        return bounded_value(self.raw_R, *self.PARAM_BOUNDS["R"])

    @property
    def C(self):
        return bounded_value(self.raw_C, *self.PARAM_BOUNDS["C"])

    @property
    def A_eff(self):
        return bounded_value(self.raw_A_eff, *self.PARAM_BOUNDS["A_eff"])

    @property
    def roxP_hvac(self):
        return bounded_value(self.raw_hvac_gain, *self.PARAM_BOUNDS["hvac_gain"])

    def _step(self, t_in, t_out, hvac_u, solar):
        cooling_sign = -1.0 if self.mode == "cool" else 1.0
        heat_flow = (t_out - t_in) / self.R
        hvac_flow = cooling_sign * self.roxP_hvac * hvac_u
        solar_flow = self.A_eff * solar
        return t_in + self.delta_t * (heat_flow + hvac_flow + solar_flow) / self.C

    def forward(self, input_vec):
        return self.onestep_predict(input_vec) if input_vec.shape[1] == 1 else self.multistep_predict(input_vec)

    def onestep_predict(self, input_vec):
        t_in = input_vec[:, :, 0]
        t_out = input_vec[:, :, 1]
        hvac_u = input_vec[:, :, 2]
        solar = input_vec[:, :, 3]
        return self._step(t_in, t_out, hvac_u, solar).unsqueeze(-1)

    def multistep_predict(self, input_vec):
        _, seq_len, _ = input_vec.shape
        t_in = input_vec[:, :, 0]
        t_out = input_vec[:, :, 1]
        hvac_u = input_vec[:, :, 2]
        solar = input_vec[:, :, 3]

        predictions = []
        t_prev = t_in[:, 0]
        for step in range(seq_len):
            t_prev = self._step(t_prev, t_out[:, step], hvac_u[:, step], solar[:, step])
            predictions.append(t_prev)
        return torch.stack(predictions, dim=1).unsqueeze(-1)

    def describe_model(self, precision: int = 6):
        desc = (
            f"1R1C system (mode={self.mode}):\n"
            f"dTin/dt = ((Tout - Tin)/R - hvac_gain*u + A_eff*sol) / C\n"
            f"(R={self.R.item():.{precision}f}, C={self.C.item():.{precision}f}, "
            f"A_eff={self.A_eff.item():.{precision}f}, hvac_gain={self.roxP_hvac.item():.{precision}f}, "
            f"dt={self.delta_t})"
        )
        print(desc)
        return desc
