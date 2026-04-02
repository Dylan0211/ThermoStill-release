"""2R1C RC model."""

import torch
import torch.nn as nn

from models.grey_box.rc_shared import bounded_value, make_bounded_parameter


class R2C1(nn.Module):
    PARAM_BOUNDS = {
        "Rm": (1e-4, 0.2),
        "Rout": (1e-4, 0.2),
        "C": (1e5, 1e8),
        "A_eff": (0.0, 0.2),
        "hvac_gain": (1.0, 2e4),
    }
    PARAM_INIT = {
        "Rm": 7e-3,
        "Rout": 1.2e-2,
        "C": 3.5e6,
        "A_eff": 0.01,
        "hvac_gain": 450.0,
    }

    def __init__(self, args):
        super().__init__()
        self.delta_t = float(args.delta_t)
        self.mode = "cool"

        self.raw_Rm = make_bounded_parameter(self.PARAM_INIT["Rm"], *self.PARAM_BOUNDS["Rm"])
        self.raw_Rout = make_bounded_parameter(self.PARAM_INIT["Rout"], *self.PARAM_BOUNDS["Rout"])
        self.raw_C = make_bounded_parameter(self.PARAM_INIT["C"], *self.PARAM_BOUNDS["C"])
        self.raw_A_eff = make_bounded_parameter(self.PARAM_INIT["A_eff"], *self.PARAM_BOUNDS["A_eff"])
        self.raw_hvac_gain = make_bounded_parameter(self.PARAM_INIT["hvac_gain"], *self.PARAM_BOUNDS["hvac_gain"])

    @property
    def Rm(self):
        return bounded_value(self.raw_Rm, *self.PARAM_BOUNDS["Rm"])

    @property
    def Rout(self):
        return bounded_value(self.raw_Rout, *self.PARAM_BOUNDS["Rout"])

    @property
    def Ri(self):
        return self.Rm

    @property
    def Re(self):
        return self.Rout

    @property
    def C(self):
        return bounded_value(self.raw_C, *self.PARAM_BOUNDS["C"])

    @property
    def A_eff(self):
        return bounded_value(self.raw_A_eff, *self.PARAM_BOUNDS["A_eff"])

    @property
    def roxP_hvac(self):
        return bounded_value(self.raw_hvac_gain, *self.PARAM_BOUNDS["hvac_gain"])

    def _state_update(self, t_in, t_mass, t_out, hvac_u, solar):
        cooling_sign = -1.0 if self.mode == "cool" else 1.0
        q_hvac = cooling_sign * self.roxP_hvac * hvac_u
        q_solar = self.A_eff * solar

        dtin = ((t_mass - t_in) / self.Rm + q_hvac + q_solar) / self.C
        dtmass = ((t_in - t_mass) / self.Rm + (t_out - t_mass) / self.Rout) / self.C
        next_t_in = t_in + self.delta_t * dtin
        next_t_mass = t_mass + self.delta_t * dtmass
        return next_t_in, next_t_mass

    def forward(self, input_vec):
        return self.onestep_predict(input_vec) if input_vec.shape[1] == 1 else self.multistep_predict(input_vec)

    def onestep_predict(self, input_vec):
        t_in = input_vec[:, :, 0]
        t_out = input_vec[:, :, 1]
        hvac_u = input_vec[:, :, 2]
        solar = input_vec[:, :, 3]
        t_mass = 0.5 * (t_in + t_out)
        next_t_in, _ = self._state_update(t_in, t_mass, t_out, hvac_u, solar)
        return next_t_in.unsqueeze(-1)

    def multistep_predict(self, input_vec):
        _, seq_len, _ = input_vec.shape
        t_in = input_vec[:, :, 0]
        t_out = input_vec[:, :, 1]
        hvac_u = input_vec[:, :, 2]
        solar = input_vec[:, :, 3]

        predictions = []
        t_in_prev = t_in[:, 0]
        t_mass_prev = 0.5 * (t_in[:, 0] + t_out[:, 0])
        for step in range(seq_len):
            t_in_prev, t_mass_prev = self._state_update(
                t_in_prev,
                t_mass_prev,
                t_out[:, step],
                hvac_u[:, step],
                solar[:, step],
            )
            predictions.append(t_in_prev)
        return torch.stack(predictions, dim=1).unsqueeze(-1)

    def describe_model(self, precision: int = 6):
        desc = (
            f"2R1C system (mode={self.mode}):\n"
            f"C*dTin/dt = (Tmass - Tin)/Rm - hvac_gain*u + A_eff*sol\n"
            f"C*dTmass/dt = (Tin - Tmass)/Rm + (Tout - Tmass)/Rout\n"
            f"(Rm={self.Rm.item():.{precision}f}, Rout={self.Rout.item():.{precision}f}, "
            f"C={self.C.item():.{precision}f}, A_eff={self.A_eff.item():.{precision}f}, "
            f"hvac_gain={self.roxP_hvac.item():.{precision}f}, dt={self.delta_t})"
        )
        print(desc)
        return desc
