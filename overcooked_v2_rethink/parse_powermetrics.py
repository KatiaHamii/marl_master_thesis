#!/usr/bin/env python3
"""Parse powermetrics.log and extract power consumption data."""

import re
import sys
from pathlib import Path

def parse_powermetrics(log_path):
    """Extract all power samples from powermetrics log."""
    with open(log_path, "r") as f:
        content = f.read()

    # Find all "Combined Power" lines
    pattern = r"Combined Power \(CPU \+ GPU \+ ANE\): (\d+) mW"
    matches = re.findall(pattern, content)

    if not matches:
        print(f"No power samples found in {log_path}")
        return None

    powers_mw = [int(m) for m in matches]
    powers_w = [p / 1000.0 for p in powers_mw]

    # Find elapsed time (in seconds) - powermetrics logs "1234.56ms elapsed"
    time_pattern = r"\((\d+\.\d+)ms elapsed\)"
    time_matches = re.findall(time_pattern, content)
    elapsed_ms = sum(float(m) for m in time_matches) if time_matches else None
    elapsed_sec = elapsed_ms / 1000.0 if elapsed_ms else None

    # Calculate stats
    avg_power_w = sum(powers_w) / len(powers_w)
    max_power_w = max(powers_w)
    min_power_w = min(powers_w)

    # Energy = Power × Time (in Wh if time is in hours)
    energy_wh = None
    if elapsed_sec:
        elapsed_hours = elapsed_sec / 3600
        energy_wh = avg_power_w * elapsed_hours

    return {
        "num_samples": len(powers_w),
        "avg_power_w": avg_power_w,
        "max_power_w": max_power_w,
        "min_power_w": min_power_w,
        "elapsed_seconds": elapsed_sec,
        "energy_wh": energy_wh,
        "powers_w": powers_w,
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_powermetrics.py <log_path> [--baseline <baseline_log>] [--carbon-intensity <kg_co2_per_kwh>]")
        sys.exit(1)

    log_path = Path(sys.argv[1])
    baseline_log = None
    carbon_intensity = 0.15  # kg CO2/kWh (EU average)

    # Parse optional arguments
    for i in range(2, len(sys.argv), 2):
        if sys.argv[i] == "--baseline" and i + 1 < len(sys.argv):
            baseline_log = Path(sys.argv[i + 1])
        elif sys.argv[i] == "--carbon-intensity" and i + 1 < len(sys.argv):
            carbon_intensity = float(sys.argv[i + 1])

    stats = parse_powermetrics(log_path)
    baseline_stats = None
    if baseline_log and baseline_log.exists():
        baseline_stats = parse_powermetrics(baseline_log)

    if stats:
        print(f"Power Metrics Summary")
        print(f"====================")
        print(f"Samples collected: {stats['num_samples']}")
        print(f"Average power:     {stats['avg_power_w']:.2f} W")
        print(f"Peak power:        {stats['max_power_w']:.2f} W")
        print(f"Min power:         {stats['min_power_w']:.2f} W")

        if baseline_stats:
            baseline_power = baseline_stats['avg_power_w']
            training_power = stats['avg_power_w'] - baseline_power
            print(f"\nBaseline subtraction:")
            print(f"Baseline power:    {baseline_power:.2f} W")
            print(f"Training power:    {training_power:.2f} W (total - baseline)")

        if stats['elapsed_seconds']:
            print(f"\nElapsed time:      {stats['elapsed_seconds']:.1f} sec ({stats['elapsed_seconds']/60:.1f} min)")

            # Use training power if baseline available, else total power
            power_w = (stats['avg_power_w'] - baseline_stats['avg_power_w']) if baseline_stats else stats['avg_power_w']
            energy_wh = power_w * (stats['elapsed_seconds'] / 3600)

            print(f"Total energy:      {energy_wh:.4f} Wh")

            # CO2 estimate
            co2_kg = energy_wh / 1000 * carbon_intensity
            print(f"Est. CO2:          {co2_kg:.6f} kg CO2 ({carbon_intensity} kg/kWh)")
