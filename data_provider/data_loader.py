import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from torch.utils.data import Dataset


class Dataset_Ecobee_1h(Dataset):
    def __init__(self, args, file_name, n_feat=4, flag="train", scale=True):
        self.args = args
        self.file_name = file_name
        self.state_dataset = args.state_dataset
        self.project_root = Path(args.project_root).resolve()
        self.tmp_data_dir = Path(args.tmp_data_dir)
        self.dataset_raw_dir = Path(args.dataset_raw_dir)

        self.n_feat = n_feat
        self.train_days = args.train_days
        self.val_days = args.val_days
        self.test_days = args.test_days
        self.context_length = args.context_length
        self.prediction_length = args.prediction_length

        # manually set the bound for indoor temperature
        self.max_intemp = 40.
        self.min_intemp = 0.

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.flag = flag
        self.set_type = type_map[self.flag]
        self.scale = scale

        self.teacher_model_names = list(args.tsfm_name_list)
        self.use_teacher_predictions = True

        self.__read_data__()

    def _resolve_under_project(self, path_like: Path) -> Path:
        return path_like if path_like.is_absolute() else (self.project_root / path_like).resolve()

    def _create_samples_for_teacher_model(self, data):
        sample_x, sample_y = [], []
        n_samples = (data.shape[0] - self.context_length) // self.prediction_length
        gap = data.shape[0] - (n_samples * self.prediction_length + self.context_length)
        if gap > 0:
            n_samples += 1
        for i in range(n_samples):
            idx = i * self.prediction_length
            sample_x.append(data[idx: idx + self.context_length, :])
            if idx + self.context_length + self.prediction_length > data.shape[0]:
                this_data = data[idx + self.context_length: idx + self.context_length + self.prediction_length, 0]
                this_data = np.pad(this_data, (0, self.prediction_length - this_data.shape[0]), mode='constant', constant_values=0.)
                sample_y.append(this_data)
            else:
                sample_y.append(data[idx + self.context_length: idx + self.context_length + self.prediction_length, 0])
        return np.array(sample_x), np.array(sample_y)

    def _build_teacher_model(self, teacher_name):
        from models.tsfm.tsfm import ChronosModel, TimesFMModel, TimeMoEModel

        teacher_map = {
            "chronos": lambda: ChronosModel(name="amazon/chronos-t5-large", device=self.args.device),
            "timesfm": lambda: TimesFMModel(name="google/timesfm-2.5-200m-pytorch"),
            "timemoe": lambda: TimeMoEModel(name="Maple728/TimeMoE-200M", device=self.args.device),
        }
        return teacher_map[teacher_name]()

    def _load_or_compute_teacher_predictions(self, data, cache_name):
        cache_root = self._resolve_under_project(self.tmp_data_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_path = cache_root / cache_name

        if cache_path.exists():
            with open(cache_path, "rb") as file:
                return pickle.load(file)

        teacher_name = cache_name.split("_train_days_")[0].split(f"{self.file_name}_", 1)[1]
        teacher_model = self._build_teacher_model(teacher_name)
        teacher_pred = teacher_model(data=data, prediction_length=self.prediction_length)
        with open(cache_path, "wb") as file:
            pickle.dump(teacher_pred, file)
        return teacher_pred

    def __read_data__(self):
        dataset_root = self._resolve_under_project(self.dataset_raw_dir)
        data_path = dataset_root / self.state_dataset / self.file_name
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        df_raw = pd.read_csv(data_path)

        # preprocessing
        df = df_raw[['time', 'T01_TEMP', 'Text', 'duty_cycle', 'GHI']].copy()
        df.rename(columns={'T01_TEMP': 'Tin', 'duty_cycle': 'u', 'GHI': 'sol'}, inplace=True)
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')

        # borders for train, val, and test set
        len_data = df.shape[0]

        # reserve the second half data for evaluation
        train_start_idx, train_end_idx = 0, 24 * self.train_days
        val_start_idx, val_end_idx = 24 * self.train_days - self.context_length, 24 * (self.train_days + self.val_days)
        test_start_idx, test_end_idx = len_data - 24 * self.test_days - self.context_length, len_data
        border1s = [train_start_idx, val_start_idx, test_start_idx]
        border2s = [train_end_idx, val_end_idx, test_end_idx]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # create data for current flag
        self.df = df.iloc[border1: border2]
        self.df.reset_index(inplace=True)

        # get teacher outputs for training data
        if self.flag == "train" and self.use_teacher_predictions:
            train_data = df.iloc[border1s[0]: border2s[0]].values
            train_x, train_y = self._create_samples_for_teacher_model(train_data)

            for tsfm_name in self.teacher_model_names:
                cache_name = f"{self.file_name}_{tsfm_name}_train_days_{self.train_days}.pkl"
                this_tsfm_pred_y = self._load_or_compute_teacher_predictions(train_x, cache_name)
                this_tsfm_pred_y = this_tsfm_pred_y.reshape(-1)
                tsfm_pred_len = train_data.shape[0] - self.context_length
                this_tsfm_pred_y = this_tsfm_pred_y[:tsfm_pred_len]
                this_tsfm_pred_y = np.pad(
                    this_tsfm_pred_y,
                    (self.context_length, 0),
                    mode='constant',
                    constant_values=0.,
                )
                self.df[f"intemp_{tsfm_name}"] = this_tsfm_pred_y

        if self.flag == "test" and self.use_teacher_predictions:
            test_data = df.iloc[border1s[2]: border2s[2]].values
            test_x, test_y = self._create_samples_for_teacher_model(test_data)

            for tsfm_name in self.teacher_model_names:
                cache_name = f"{self.file_name}_{tsfm_name}_train_days_{self.train_days}_test_set.pkl"
                this_tsfm_pred_y = self._load_or_compute_teacher_predictions(test_x, cache_name)
                this_tsfm_pred_y = this_tsfm_pred_y.reshape(-1)
                tsfm_pred_len = test_data.shape[0] - self.context_length
                this_tsfm_pred_y = this_tsfm_pred_y[:tsfm_pred_len]
                this_tsfm_pred_y = np.pad(
                    this_tsfm_pred_y,
                    (self.context_length, 0),
                    mode='constant',
                    constant_values=0.,
                )
                self.df[f"intemp_{tsfm_name}"] = this_tsfm_pred_y

        self.data = self.df.iloc[:, 1:].values  # remove timestmap

    def __getitem__(self, idx):
        # full historical context for the ThermoStill state encoder
        data_x_ctx = self.data[idx: idx + self.context_length, :self.n_feat]

        # single-step input and output
        # data_x_s: (1, n_variate)
        # data_y_s: (1, 1)
        # data_y_tsfm_s: (1, n_teacher)
        data_x_s = self.data[idx + self.context_length - 1: idx + self.context_length, :self.n_feat]
        data_y_s = self.data[idx + self.context_length: idx + self.context_length + 1, :1]
        data_y_tsfm_s = self.data[idx + self.context_length: idx + self.context_length + 1, self.n_feat:]
        # multi-step input and output
        # data_x_m: (prediction_length, n_variate)
        # data_y_m: (prediction_length, 1)
        # data_y_tsfm_m: (prediction_length, n_teacher)
        data_x_m = self.data[idx + self.context_length - 1: idx + self.context_length + self.prediction_length - 1, :self.n_feat]
        data_y_m = self.data[idx + self.context_length: idx + self.context_length + self.prediction_length, :1]
        data_y_tsfm_m = self.data[idx + self.context_length: idx + self.context_length + self.prediction_length, self.n_feat:]

        return (data_x_ctx,
                data_x_s, data_y_s, data_y_tsfm_s,
                data_x_m, data_y_m, data_y_tsfm_m)

    def __len__(self):
        return self.data.shape[0] - self.context_length - self.prediction_length + 1

    def inverse_transform(self, data):
        return (data * (self.norm_stat_dict["intemp"][1] - self.norm_stat_dict["intemp"][0])
                + self.norm_stat_dict["intemp"][0])
