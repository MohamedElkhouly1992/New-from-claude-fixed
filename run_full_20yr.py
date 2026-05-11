"""
HVAC ROM-Degradation Suite
Full 20-Year PhD Simulation
============================
All five coupled physics effects ON.
All four strategies S0-S3.
Daily time step, APO_POP=12, APO_ITERS=6.

Run:   python run_full_20yr.py

Expected runtime: ~5 minutes on a modern laptop.
For a faster test run change YEARS=5, APO_POP=8, APO_ITERS=4.
"""

from __future__ import annotations
import warnings, sys
from pathlib import Path
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from hvac_v3_engine import BuildingSpec, HVACConfig, run_scenario_model
except ImportError as e:
    print(f"ERROR: Cannot import engine — {e}")
    print("Run:  pip install -r requirements.txt")
    sys.exit(1)

# ─── Settings ─────────────────────────────────────────────────────────────
YEARS       = 20        # full PhD horizon
APO_POP     = 12        # 18 for publication quality
APO_ITERS   = 6         # 10 for publication quality
TIME_STEP   = 24.0      # daily (keep this)
OUTPUT_DIR  = "full_20yr_output"

# ─── Building ─────────────────────────────────────────────────────────────
bldg = BuildingSpec(
    building_type          = "Educational / University building",
    location               = "New Mansoura, Egypt (31.4°N)",
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

# ─── HVAC Config — ALL coupled effects ON ─────────────────────────────────
cfg = HVACConfig(
    years              = YEARS,
    TIME_STEP_HOURS    = TIME_STEP,
    APO_POP            = APO_POP,
    APO_ITERS          = APO_ITERS,

    hvac_system_type   = "Chiller_AHU",
    COP_COOL_NOM       = 4.5,
    COP_HEAT_NOM       = 3.2,
    COP_AGING_RATE     = 0.005,
    FAN_EFF            = 0.70,
    PUMP_SPECIFIC_W_M2 = 1.30,
    AUXILIARY_W_M2     = 0.55,

    T_SET              = 23.0,
    T_SP_MIN           = 21.0,
    T_SP_MAX           = 26.0,
    AF_MIN             = 0.55,
    AF_MAX             = 1.00,

    RF_STAR            = 2e-4,
    B_FOUL             = 0.015,
    DUST_RATE          = 1.2,
    K_CLOG             = 6.0,
    DP_CLEAN           = 150.0,
    DP_MAX             = 450.0,
    DEG_TRIGGER        = 0.55,
    degradation_model  = "physics",

    CO2_FACTOR         = 0.536,
    E_PRICE            = 0.12,
    COST_FILTER        = 50.0,
    COST_HX            = 300.0,
    FILTER_INTERVAL    = 90,
    HX_INTERVAL        = 180,

    W_ENERGY           = 0.35,
    W_DEGRAD           = 0.25,
    W_COMFORT          = 0.25,
    W_CARBON           = 0.15,

    PLR_CURVE_TYPE     = "Quadratic",
    PLR_A              = 0.85,
    PLR_B              = 0.25,
    PLR_C              = -0.10,
    PLR_D              = 0.0,
    PLR_MIN_MODIFIER   = 0.55,
    PLR_MAX_MODIFIER   = 1.15,

    INDOOR_RH_TARGET_PCT        = 50.0,
    LATENT_VENTILATION_FRACTION = 0.35,
    FLOOR_TO_FLOOR_M            = 3.2,

    HX_AIR_FOULING_FACTOR       = 0.75,
    HX_WATER_DP_CLEAN_KPA       = 35.0,
    HX_WATER_FLOW_NOM_M3H       = 0.0,
    HX_WATER_FOULING_FACTOR     = 0.35,
    HX_PUMP_EFF                 = 0.65,
    HX_CHW_DT_K                 = 5.0,
    HX_HW_DT_K                  = 10.0,
    HX_UA_CLEAN_KW_K            = 0.0,
    HX_UA_LOSS_FACTOR           = 0.30,
    HX_LMTD_CORRECTION          = 0.90,

    # ── ALL FIVE EFFECTS ON ───────────────────────────────────────────────
    APPLY_PART_LOAD_COP_TO_CORE     = True,
    APPLY_LATENT_LOAD_TO_CORE       = True,
    APPLY_HX_AIR_PRESSURE_TO_FAN    = True,
    APPLY_HX_WATER_PRESSURE_TO_PUMP = True,
    APPLY_HX_UA_TO_CAPACITY         = True,
    APPLY_NATIVE_ZONE_LOADS         = False,
)

# ─── Run ──────────────────────────────────────────────────────────────────
print()
print("=" * 64)
print("  HVAC EMS — Full 20-Year Simulation (All Effects ON)")
print("=" * 64)
print(f"  Years       : {YEARS}")
print(f"  Time step   : {TIME_STEP} h (daily)")
print(f"  APO         : POP={APO_POP}  ITERS={APO_ITERS}")
print(f"  Strategies  : S0 S1 S2 S3")
print(f"  Severity    : Moderate")
print(f"  Climate     : C0_Baseline")
print(f"  All APPLY_* : True (5/5 effects active)")
print()

result = run_scenario_model(
    output_dir                  = OUTPUT_DIR,
    axis_mode                   = "one_strategy",
    bldg                        = bldg,
    cfg                         = cfg,
    weather_mode                = "synthetic",
    fixed_severity              = "Moderate",
    fixed_climate               = "C0_Baseline",
    degradation_model           = "physics",
    include_baseline_layer      = True,
    include_baseline_as_scenario= True,
    random_state                = 42,
)

# ─── Print headline KPIs ─────────────────────────────────────────────────
sumdf  = pd.read_csv(result["summary_csv"])
anndf  = pd.read_csv(result["annual_csv"])

KPI_COLS = [
    "strategy",
    "Total Energy MWh",
    "Mean Comfort Deviation C",
    "Total CO2 tonne",
    "Mean Degradation Index",
    "Total Cost USD",
    "Occupied Discomfort Days",
]
available = [c for c in KPI_COLS if c in sumdf.columns]

print()
print("  20-YEAR KPI SUMMARY")
print("  " + "-" * 60)
with pd.option_context("display.max_columns", 20, "display.width", 140,
                       "display.float_format", "{:.2f}".format):
    print(sumdf[available].to_string(index=False))

ANN_COLS = ["strategy", "year", "annual_energy_MWh",
            "mean_delta", "mean_comfort_dev", "annual_co2_tonne"]
avail_a = [c for c in ANN_COLS if c in anndf.columns]
print()
print("  ANNUAL BREAKDOWN (first 5 years shown)")
print("  " + "-" * 60)
with pd.option_context("display.max_columns", 20, "display.width", 140,
                       "display.float_format", "{:.3f}".format):
    print(anndf[anndf["year"] <= 5][avail_a].to_string(index=False))

print()
print(f"  Full outputs saved to: {Path(OUTPUT_DIR).resolve()}")
print(f"  ├── {Path(result['summary_csv']).name}")
print(f"  ├── {Path(result['annual_csv']).name}")
print(f"  ├── {Path(result['excel_report']).name}")
print(f"  └── {Path(result['pdf_report']).name}")
print()
