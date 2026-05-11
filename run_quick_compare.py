"""
HVAC ROM-Degradation Suite
Quick Effects Comparison Runner
================================
Runs every coupled physics effect one at a time, compares all four KPIs,
and prints a clean results table + saves CSV outputs.

Run:   python run_quick_compare.py

No arguments needed. All settings are defined inside this file.
Edit YEARS, APO_POP, APO_ITERS at the top to trade speed vs. accuracy.
"""

from __future__ import annotations
import warnings, copy, sys
from dataclasses import asdict
from pathlib import Path
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ─── Try importing engine ─────────────────────────────────────────────────
try:
    from hvac_v3_engine import BuildingSpec, HVACConfig, run_scenario_model
except ImportError as e:
    print(f"ERROR: Could not import engine — {e}")
    print("Make sure hvac_v3_engine.py is in the same folder and requirements are installed:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════
# SPEED SETTINGS — Edit these to control runtime
# ═══════════════════════════════════════════════════════════════════════════
YEARS       = 5       # increase to 20 for full PhD run (~5 min)
APO_POP     = 8       # S3 optimizer population. increase to 18 for publication
APO_ITERS   = 4       # S3 optimizer iterations. increase to 10 for publication
TIME_STEP   = 24.0    # hours. keep 24 (daily) — hourly is 24x slower
SEVERITY    = "Moderate"
CLIMATE     = "C0_Baseline"
OUTPUT_DIR  = "quick_compare_output"
RANDOM_SEED = 42

# ═══════════════════════════════════════════════════════════════════════════
# BUILDING — New Mansoura University defaults
# ═══════════════════════════════════════════════════════════════════════════
BLDG = BuildingSpec(
    building_type          = "Educational / University building",
    location               = "New Mansoura, Egypt",
    conditioned_area_m2    = 2000.0,
    floors                 = 3,
    n_spaces               = 20,
    occupancy_density_p_m2 = 0.08,
    lighting_w_m2          = 10.0,
    equipment_w_m2         = 8.0,
    airflow_m3h_m2         = 4.0,
    infiltration_ach       = 0.50,
    sensible_w_per_person  = 75.0,
    cooling_intensity_w_m2 = 100.0,
    heating_intensity_w_m2 = 55.0,
    wall_u                 = 0.60,
    roof_u                 = 0.35,
    window_u               = 2.70,
    shgc                   = 0.35,
    glazing_ratio          = 0.30,
)

# ═══════════════════════════════════════════════════════════════════════════
# BASE CONFIG — all APPLY_* OFF, correct HX params so no div-by-zero errors
# ═══════════════════════════════════════════════════════════════════════════
BASE_CFG = HVACConfig(
    # ── Simulation ────────────────────────────────────────────────────────
    years              = YEARS,
    TIME_STEP_HOURS    = TIME_STEP,
    APO_POP            = APO_POP,
    APO_ITERS          = APO_ITERS,

    # ── HVAC system ───────────────────────────────────────────────────────
    hvac_system_type   = "Chiller_AHU",
    COP_COOL_NOM       = 4.5,
    COP_HEAT_NOM       = 3.2,
    COP_AGING_RATE     = 0.005,
    FAN_EFF            = 0.70,
    PUMP_SPECIFIC_W_M2 = 1.30,
    AUXILIARY_W_M2     = 0.55,

    # ── Setpoint range ────────────────────────────────────────────────────
    T_SET              = 23.0,
    T_SP_MIN           = 21.0,
    T_SP_MAX           = 26.0,
    AF_MIN             = 0.55,
    AF_MAX             = 1.00,

    # ── Degradation ───────────────────────────────────────────────────────
    RF_STAR            = 2e-4,
    B_FOUL             = 0.015,
    DUST_RATE          = 1.2,
    K_CLOG             = 6.0,
    DP_CLEAN           = 150.0,
    DP_MAX             = 450.0,
    DEG_TRIGGER        = 0.55,
    degradation_model  = "physics",

    # ── Cost & carbon ─────────────────────────────────────────────────────
    CO2_FACTOR         = 0.536,      # Egypt grid (IEA 2023)
    E_PRICE            = 0.12,
    COST_FILTER        = 50.0,
    COST_HX            = 300.0,
    FILTER_INTERVAL    = 90,
    HX_INTERVAL        = 180,

    # ── Objective weights ─────────────────────────────────────────────────
    W_ENERGY           = 0.35,
    W_DEGRAD           = 0.25,
    W_COMFORT          = 0.25,
    W_CARBON           = 0.15,

    # ── PLR curve (needed when APPLY_PART_LOAD_COP_TO_CORE = True) ────────
    PLR_CURVE_TYPE     = "Quadratic",
    PLR_A              = 0.85,
    PLR_B              = 0.25,
    PLR_C              = -0.10,
    PLR_D              = 0.0,
    PLR_MIN_MODIFIER   = 0.55,
    PLR_MAX_MODIFIER   = 1.15,

    # ── Latent load (needed when APPLY_LATENT_LOAD_TO_CORE = True) ────────
    INDOOR_RH_TARGET_PCT        = 50.0,
    LATENT_VENTILATION_FRACTION = 0.35,
    FLOOR_TO_FLOOR_M            = 3.2,

    # ── HX hydraulic params (needed when APPLY_HX_* = True) ───────────────
    HX_AIR_FOULING_FACTOR       = 0.75,
    HX_WATER_DP_CLEAN_KPA       = 35.0,
    HX_WATER_FLOW_NOM_M3H       = 0.0,   # 0 = auto-derived from load
    HX_WATER_FOULING_FACTOR     = 0.35,
    HX_PUMP_EFF                 = 0.65,
    HX_CHW_DT_K                 = 5.0,
    HX_HW_DT_K                  = 10.0,
    HX_UA_CLEAN_KW_K            = 0.0,
    HX_UA_LOSS_FACTOR           = 0.30,
    HX_LMTD_CORRECTION          = 0.90,

    # ── All coupled effects OFF (base case) ────────────────────────────────
    APPLY_PART_LOAD_COP_TO_CORE     = False,
    APPLY_LATENT_LOAD_TO_CORE       = False,
    APPLY_HX_AIR_PRESSURE_TO_FAN    = False,
    APPLY_HX_WATER_PRESSURE_TO_PUMP = False,
    APPLY_HX_UA_TO_CAPACITY         = False,
    APPLY_NATIVE_ZONE_LOADS         = False,
)

# ═══════════════════════════════════════════════════════════════════════════
# EFFECT VARIANTS
# Each entry switches EXACTLY the listed flags ON (rest stay OFF)
# except ALL_Effects which switches all five ON simultaneously
# ═══════════════════════════════════════════════════════════════════════════
VARIANTS: dict[str, dict[str, bool]] = {
    "BASE_AllOff": {
        "APPLY_PART_LOAD_COP_TO_CORE":     False,
        "APPLY_LATENT_LOAD_TO_CORE":        False,
        "APPLY_HX_AIR_PRESSURE_TO_FAN":    False,
        "APPLY_HX_WATER_PRESSURE_TO_PUMP": False,
        "APPLY_HX_UA_TO_CAPACITY":          False,
    },
    "01_PLR_COP": {
        "APPLY_PART_LOAD_COP_TO_CORE":     True,
        "APPLY_LATENT_LOAD_TO_CORE":        False,
        "APPLY_HX_AIR_PRESSURE_TO_FAN":    False,
        "APPLY_HX_WATER_PRESSURE_TO_PUMP": False,
        "APPLY_HX_UA_TO_CAPACITY":          False,
    },
    "02_Latent_Load": {
        "APPLY_PART_LOAD_COP_TO_CORE":     False,
        "APPLY_LATENT_LOAD_TO_CORE":        True,
        "APPLY_HX_AIR_PRESSURE_TO_FAN":    False,
        "APPLY_HX_WATER_PRESSURE_TO_PUMP": False,
        "APPLY_HX_UA_TO_CAPACITY":          False,
    },
    "03_HX_Air_Fan": {
        "APPLY_PART_LOAD_COP_TO_CORE":     False,
        "APPLY_LATENT_LOAD_TO_CORE":        False,
        "APPLY_HX_AIR_PRESSURE_TO_FAN":    True,
        "APPLY_HX_WATER_PRESSURE_TO_PUMP": False,
        "APPLY_HX_UA_TO_CAPACITY":          False,
    },
    "04_HX_Water_Pump": {
        "APPLY_PART_LOAD_COP_TO_CORE":     False,
        "APPLY_LATENT_LOAD_TO_CORE":        False,
        "APPLY_HX_AIR_PRESSURE_TO_FAN":    False,
        "APPLY_HX_WATER_PRESSURE_TO_PUMP": True,
        "APPLY_HX_UA_TO_CAPACITY":          False,
    },
    "05_HX_UA_Capacity": {
        "APPLY_PART_LOAD_COP_TO_CORE":     False,
        "APPLY_LATENT_LOAD_TO_CORE":        False,
        "APPLY_HX_AIR_PRESSURE_TO_FAN":    False,
        "APPLY_HX_WATER_PRESSURE_TO_PUMP": False,
        "APPLY_HX_UA_TO_CAPACITY":          True,
    },
    "ALL_Effects": {
        "APPLY_PART_LOAD_COP_TO_CORE":     True,
        "APPLY_LATENT_LOAD_TO_CORE":        True,
        "APPLY_HX_AIR_PRESSURE_TO_FAN":    True,
        "APPLY_HX_WATER_PRESSURE_TO_PUMP": True,
        "APPLY_HX_UA_TO_CAPACITY":          True,
    },
}

# KPI columns from summary CSV → display label
KPIS = {
    "Total Energy MWh":        "Energy (MWh)",
    "Mean Comfort Deviation C": "Comfort Dev (°C)",
    "Total CO2 tonne":          "Carbon (tonne CO2)",
    "Mean Degradation Index":   "Degradation Index",
}


def make_cfg(flags: dict) -> HVACConfig:
    """Return a fresh HVACConfig with the given APPLY_* flags overridden."""
    d = asdict(BASE_CFG)
    d.update(flags)
    return HVACConfig(**d)


def safe_float(val):
    try:
        return float(val)
    except Exception:
        return float("nan")


def run_all() -> pd.DataFrame:
    root = Path(OUTPUT_DIR)
    root.mkdir(parents=True, exist_ok=True)

    rows = []
    n = len(VARIANTS)

    print()
    print("=" * 72)
    print("  HVAC EMS Simulation — Effects Comparison")
    print(f"  {YEARS}-year horizon  |  daily time step  |  Moderate severity")
    print(f"  APO POP={APO_POP}  ITERS={APO_ITERS}  |  {n} variants × 4 strategies")
    print("=" * 72)

    for idx, (variant_name, flags) in enumerate(VARIANTS.items(), 1):
        active = [k.replace("APPLY_","").replace("_TO_CORE","")
                   .replace("_TO_FAN","").replace("_TO_PUMP","")
                  for k, v in flags.items() if v]
        label = ", ".join(active) if active else "none"
        print(f"\n[{idx}/{n}]  Variant: {variant_name}")
        print(f"         Active effects: {label or 'none (base)'}")

        cfg = make_cfg(flags)
        out_dir = root / variant_name

        try:
            result = run_scenario_model(
                output_dir          = str(out_dir),
                axis_mode           = "one_strategy",
                bldg                = BLDG,
                cfg                 = cfg,
                weather_mode        = "synthetic",
                fixed_severity      = SEVERITY,
                fixed_climate       = CLIMATE,
                degradation_model   = "physics",
                include_baseline_layer      = False,
                include_baseline_as_scenario= False,
                random_state        = RANDOM_SEED,
            )
            sumdf = pd.read_csv(result["summary_csv"])
            for _, row in sumdf.iterrows():
                entry = {
                    "variant":        variant_name,
                    "active_effects": label or "none",
                    "strategy":       row.get("strategy", "?"),
                }
                for col in KPIS:
                    entry[col] = safe_float(row.get(col, float("nan")))
                rows.append(entry)
            strats = sumdf["strategy"].tolist() if "strategy" in sumdf.columns else []
            print(f"         Done  — strategies: {strats}")

        except Exception as exc:
            import traceback
            print(f"         ERROR: {exc}")
            traceback.print_exc()

    if not rows:
        print("\nNo results collected — check errors above.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.to_csv(root / "effects_comparison_all.csv", index=False)

    # ── Delta vs BASE ──────────────────────────────────────────────────────
    base = df[df["variant"] == "BASE_AllOff"].set_index("strategy")
    delta_rows = []
    for _, row in df[df["variant"] != "BASE_AllOff"].iterrows():
        stg = row["strategy"]
        r = {"variant": row["variant"], "strategy": stg,
             "active_effects": row["active_effects"]}
        for col in KPIS:
            bval = safe_float(base.at[stg, col]) if stg in base.index else float("nan")
            val  = safe_float(row[col])
            r[f"Δ {KPIS[col]}"] = (
                f"{(val - bval) / max(abs(bval), 1e-9) * 100:+.1f}%"
                if np.isfinite(bval) and np.isfinite(val) else "n/a"
            )
            r[KPIS[col]] = val
        delta_rows.append(r)

    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(root / "effects_delta_vs_base.csv", index=False)

    # ── Console table ──────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  RESULTS — KPI VALUES BY VARIANT × STRATEGY")
    print("=" * 72)
    display_cols = ["variant", "strategy"] + list(KPIS.keys())
    available = [c for c in display_cols if c in df.columns]
    with pd.option_context("display.max_columns", 20,
                           "display.width", 160,
                           "display.float_format", "{:.3f}".format):
        print(df[available].to_string(index=False))

    print()
    print("=" * 72)
    print("  DELTA vs BASE_AllOff  (% change per effect)")
    print("=" * 72)
    delta_show = ["variant", "strategy"] + [f"Δ {v}" for v in KPIS.values()]
    delta_avail = [c for c in delta_show if c in delta_df.columns]
    print(delta_df[delta_avail].to_string(index=False))

    print()
    print(f"  Full outputs saved to:  {root.resolve()}")
    print(f"  ├── effects_comparison_all.csv   (raw KPI table)")
    print(f"  ├── effects_delta_vs_base.csv    (% delta table)")
    print(f"  └── <variant>/                   (full CSV + Excel per variant)")
    print()

    return df


if __name__ == "__main__":
    run_all()
