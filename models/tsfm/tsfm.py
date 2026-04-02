"""Teacher model wrappers used by the ThermoStill training path."""

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler


class ChronosModel:
    def __init__(self, name="amazon/chronos-t5-large", device="cuda"):
        from chronos import ChronosPipeline

        self.model = ChronosPipeline.from_pretrained(
            name,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        self.scaler = MinMaxScaler()

    def __call__(self, data, prediction_length, num_samples=1):
        data = data[:, :, 0]
        n_sample, seq_len = data.shape[0], data.shape[1]
        normalized_data = self.scaler.fit_transform(np.array(data).reshape(-1, 1))
        model_input = torch.tensor(normalized_data.flatten(), dtype=torch.float32).reshape(n_sample, seq_len)

        forecast = self.model.predict(
            inputs=model_input,
            prediction_length=prediction_length,
            num_samples=num_samples,
            limit_prediction_length=False,
        )
        output = forecast[:, 0, :].numpy().reshape(-1, 1)
        return self.scaler.inverse_transform(output).reshape(n_sample, prediction_length)


class TimesFMModel:
    def __init__(self, name="google/timesfm-2.5-200m-pytorch"):
        import timesfm

        self.timesfm = timesfm
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(name)
        self.scaler = MinMaxScaler()

    def __call__(self, data, prediction_length):
        data = data[:, :, 0]
        n_sample, seq_len = data.shape[0], data.shape[1]
        normalized_data = self.scaler.fit_transform(np.array(data).reshape(-1, 1)).reshape(n_sample, seq_len)

        self.model.compile(
            self.timesfm.ForecastConfig(
                max_context=512,
                max_horizon=512,
                normalize_inputs=False,
            )
        )
        forecast_list = []
        for idx in range(normalized_data.shape[0]):
            point_forecast, _ = self.model.forecast(
                horizon=prediction_length,
                inputs=[normalized_data[idx]],
            )
            forecast_list.append(self.scaler.inverse_transform(point_forecast))
        return np.concatenate(forecast_list)


class TimeMoEModel:
    def __init__(self, name="Maple728/TimeMoE-50M", device="cuda"):
        from transformers import AutoModelForCausalLM

        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=name,
            device_map=device,
            trust_remote_code=True,
        )
        self.device = device
        self.scaler = MinMaxScaler()

    def __call__(self, data, prediction_length):
        data = data[:, :, 0]
        n_sample, seq_len = data.shape[0], data.shape[1]
        normalized_data = self.scaler.fit_transform(np.array(data).reshape(-1, 1)).reshape(n_sample, seq_len)
        model_input = torch.tensor(normalized_data.flatten(), dtype=torch.float32, device=self.device).reshape(
            n_sample, seq_len
        )

        forecast_output = self.model.generate(model_input, max_new_tokens=prediction_length)
        forecast_output = forecast_output[:, -prediction_length:].detach().cpu().numpy()
        return self.scaler.inverse_transform(forecast_output).reshape(n_sample, prediction_length)
