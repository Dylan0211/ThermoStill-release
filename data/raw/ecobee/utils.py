"""Utility helpers for ecobee data preparation and analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def read_csvs_to_dfs(main_output_directory):
    """Load grouped ecobee CSV files into a nested dictionary."""
    all_houses_dict = {}
    main_output_directory = Path(main_output_directory)

    for subdirectory in main_output_directory.iterdir():
        if not subdirectory.is_dir():
            continue

        house_group = int(subdirectory.name.split("_")[-1])
        all_houses_dict.setdefault(house_group, {})

        for csv_path in subdirectory.glob("*.csv"):
            house_id = csv_path.stem.split("_")[-1]
            all_houses_dict[house_group][house_id] = pd.read_csv(csv_path)

    return all_houses_dict


def process_house_data(df):
    """Convert one raw ecobee house dataframe into the training-ready format."""
    df["duty_cycle"] = df["CoolingRunTime"] / 3600
    df.rename(columns={"Outdoor_Temperature": "Text"}, inplace=True)

    sensor_rename_map = {
        "Thermostat_Temperature": "T01_TEMP",
        "RemoteSensor1_Temperature": "T02_TEMP",
        "RemoteSensor2_Temperature": "T03_TEMP",
        "RemoteSensor3_Temperature": "T04_TEMP",
        "RemoteSensor4_Temperature": "T05_TEMP",
        "RemoteSensor5_Temperature": "T06_TEMP",
    }
    df.rename(columns=sensor_rename_map, inplace=True)

    temp_columns = [f"T0{i}_TEMP" for i in range(1, 7)] + ["Text"]
    for col in temp_columns:
        df[col] = (df[col] - 32) * 5 / 9 + 273.15

    columns_to_keep = ["time", "GHI", "duty_cycle"] + temp_columns
    df = df[columns_to_keep]
    df = df.ffill()
    return df


def print_optimization_statistics(optimization_results):
    """Print summary statistics for train/test RMSE grouped by sensor count."""
    for sensor_count, houses in optimization_results.items():
        rmse_train = [details["rmse_train"] for details in houses.values()]
        rmse_test = [details["rmse_test"] for details in houses.values()]

        print(f"Sensor Count: {sensor_count}")
        print("Training RMSE Statistics:")
        print(f"  Mean: {np.mean(rmse_train):.2f}")
        print(f"  Median: {np.median(rmse_train):.2f}")
        print(f"  Max: {np.max(rmse_train):.2f}")
        print(f"  Min: {np.min(rmse_train):.2f}")
        print(f"  Standard Deviation: {np.std(rmse_train):.2f}\n")

        print("Testing RMSE Statistics:")
        print(f"  Mean: {np.mean(rmse_test):.2f}")
        print(f"  Median: {np.median(rmse_test):.2f}")
        print(f"  Max: {np.max(rmse_test):.2f}")
        print(f"  Min: {np.min(rmse_test):.2f}")
        print(f"  Standard Deviation: {np.std(rmse_test):.2f}\n")


def plot_error_distribution(optimization_results):
    """Plot the distribution of test RMSE values for each sensor-count bucket."""
    data = {"Sensor Count": [], "Test RMSE": []}
    for sensor_count, houses in optimization_results.items():
        for results in houses.values():
            data["Sensor Count"].append(sensor_count)
            data["Test RMSE"].append(results["rmse_test"])

    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.ylim(0, 0.6)
    plt.title("Distribution of Test RMSE Errors by Sensor Count")
    sns.boxplot(x="Sensor Count", y="Test RMSE", data=df)
    plt.xlabel("Sensor Count")
    plt.ylabel("Test RMSE")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()
