from pathlib import Path

from data_provider.data_loader import Dataset_Ecobee_1h
from torch.utils.data import DataLoader


def data_provider(args, flag):
    project_root = Path(args.project_root).resolve()
    dataset_raw_dir = Path(args.dataset_raw_dir)
    if not dataset_raw_dir.is_absolute():
        dataset_raw_dir = (project_root / dataset_raw_dir).resolve()

    state_dir = dataset_raw_dir / args.state_dataset
    if not state_dir.exists():
        raise FileNotFoundError(f"State dataset directory not found: {state_dir}")

    csv_files = sorted(path.name for path in state_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {state_dir}")
    if args.file_name not in csv_files:
        preview = ", ".join(csv_files[:5])
        raise FileNotFoundError(
            f"File '{args.file_name}' not found in '{state_dir}'. "
            f"Available examples: {preview}"
        )

    shuffle_flag = False if (flag == 'test') else True
    drop_last = True
    batch_size = args.batch_size

    data_set = Dataset_Ecobee_1h(
        args=args,
        file_name=args.file_name,
        flag=flag,
        scale=args.scale,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        dataset=data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last,
    )
    return data_set, data_loader
