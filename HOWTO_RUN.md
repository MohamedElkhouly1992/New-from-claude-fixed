# HVAC ROM-Degradation Suite — How to Run (No Errors, No Slowness)

## 1 · Install once

```bash
pip install -r requirements.txt
```

---

## 2 · Three ways to run

| Script | What it does | Time |
|---|---|---|
| `python run_quick_compare.py` | 7 variants × 4 strategies, shows every effect on all 4 KPIs | ~3 min |
| `python run_full_20yr.py` | 20-year full run, all effects ON, all strategies | ~5 min |
| `streamlit run streamlit_app.py` | Interactive dashboard | interactive |

**Start here:**
```bash
python run_quick_compare.py
```

---

## 3 · Speed — the only three settings that matter

Open `run_quick_compare.py` (or `run_full_20yr.py`) and edit these lines at the top:

```python
YEARS     = 5      # ← 5 for quick test, 20 for PhD run
APO_POP   = 8      # ← 8 for quick test, 18 for publication
APO_ITERS = 4      # ← 4 for quick test, 10 for publication
TIME_STEP = 24.0   # ← ALWAYS keep 24.0 (daily). Hourly = 24× slower
```

| Mode | YEARS | APO_POP | APO_ITERS | Expected time |
|---|---|---|---|---|
| Quick test | 5 | 8 | 4 | ~1 min |
| Standard | 5 | 12 | 6 | ~3 min |
| PhD / Publication | 20 | 18 | 10 | ~15 min |

---

## 4 · How each effect reaches the 4 KPIs

Every `APPLY_*` flag modifies an intermediate quantity inside the main solver
(`evaluate_controls`) and the correction propagates automatically to all outputs.

```
APPLY_PART_LOAD_COP_TO_CORE
  → f(PLR) = A + B·PLR + C·PLR²          modifies COP_eff
  → P_HVAC = Q_HVAC / max(COP_eff·f(PLR), 0.8)
  → E_period = P_total · Δt_h             ← Energy KPI
  → CO2 = E_period × 0.536               ← Carbon KPI

APPLY_LATENT_LOAD_TO_CORE
  → Q_latent = ṁ_air · h_fg · Δω         adds to Q_cool
  → Q_HVAC ↑ → P_HVAC ↑
  → Energy KPI ↑  Carbon KPI ↑

APPLY_HX_AIR_PRESSURE_TO_FAN
  → ΔP_fan = ΔP_clean · α_f² · (1 + 0.75·δ)
  → P_fan ↑ → E_period ↑
  → Energy KPI ↑  Carbon KPI ↑  Degradation KPI feeds back

APPLY_HX_WATER_PRESSURE_TO_PUMP
  → ΔP_w = ΔP_w,clean · ratio² · (1 + 0.35·δ)
  → P_pump ↑ → E_period ↑
  → Energy KPI ↑  Carbon KPI ↑

APPLY_HX_UA_TO_CAPACITY
  → Q_cap = 1.2 · Q_des · max(0.4, 1 − 0.30·δ)
  → Q_HVAC = min(Q_raw, Q_cap)
  → Q_unmet → T_zone ↑ → comfort_dev ↑
  → Comfort KPI ↑  Degradation KPI ↑
```

**Key rule:** The flags must be `True` for effects to reach the KPIs.
They are `False` by default for backward compatibility.
`run_full_20yr.py` sets all five to `True`.
`run_quick_compare.py` turns them on one at a time to isolate each contribution.

---

## 5 · Output files and which column = which KPI

After any run, find these in the output folder:

| File | KPI columns |
|---|---|
| `summary_csv` | `Total Energy MWh` · `Mean Comfort Deviation C` · `Total CO2 tonne` · `Mean Degradation Index` |
| `annual_csv` | `annual_energy_MWh` · `mean_comfort_dev` · `annual_co2_tonne` · `mean_delta` |
| `dataset_csv` (daily) | `energy_kwh_period` · `comfort_dev_C` · `co2_kg_period` · `delta` |

`run_quick_compare.py` also writes:
- `effects_comparison_all.csv` — raw KPI table for all variants × strategies
- `effects_delta_vs_base.csv`  — % change relative to BASE_AllOff

---

## 6 · Streamlit app — settings to change for speed

```
Tab: ❄️ HVAC & Degradation
  Simulation horizon (years) → 5 for exploration, 20 for final
  Time step                  → Daily (fastest)   ← always keep this
  APO population             → 8–12
  APO iterations             → 4–6

Tab: ⚡ Physics Effects
  Toggle all five APPLY_* switches ON
  (click "Enable all effects" button)
```

---

## 7 · Common errors and fixes

| Error message | Cause | Fix |
|---|---|---|
| `ImportError: No module named X` | Requirements not installed | `pip install -r requirements.txt` |
| `ZeroDivisionError` in pump/fan | HX_PUMP_EFF or FAN_EFF = 0 | Both are set to 0.65/0.70 in the runners — do not set to 0 |
| `KeyError: 'excel'` in Export tab | No simulation run yet | Run a simulation first; the tab auto-finds files |
| S3 never finishes | APO_POP × APO_ITERS too large | Reduce as shown in Section 3 above |
| All KPIs identical across effects | APPLY_* flags are all False | Use `run_full_20yr.py` which sets all to True |
| `ValueError: Unsupported time-step` | TIME_STEP not in {1,3,6,12,24} | Only use those exact values |
| Very high comfort deviation | HX_UA_TO_CAPACITY ON + high δ | Reduce HX_UA_LOSS_FACTOR (default 0.30 is correct) |

---

## 8 · Minimum working config to guarantee all 4 KPIs are affected

```python
from hvac_v3_engine import BuildingSpec, HVACConfig, run_scenario_model

cfg = HVACConfig(
    years              = 5,
    TIME_STEP_HOURS    = 24.0,
    APO_POP            = 8,
    APO_ITERS          = 4,
    CO2_FACTOR         = 0.536,          # Egypt grid
    RF_STAR            = 2e-4,
    B_FOUL             = 0.015,
    DUST_RATE          = 1.2,
    K_CLOG             = 6.0,

    # HX params — required when APPLY_HX_* are True
    HX_WATER_DP_CLEAN_KPA       = 35.0,
    HX_WATER_FLOW_NOM_M3H       = 0.0,   # auto-derived
    HX_UA_LOSS_FACTOR           = 0.30,
    HX_PUMP_EFF                 = 0.65,
    HX_CHW_DT_K                 = 5.0,

    # PLR params — required when APPLY_PART_LOAD_COP is True
    PLR_CURVE_TYPE     = "Quadratic",
    PLR_A=0.85, PLR_B=0.25, PLR_C=-0.10,

    # Latent params — required when APPLY_LATENT_LOAD is True
    INDOOR_RH_TARGET_PCT        = 50.0,
    LATENT_VENTILATION_FRACTION = 0.35,

    # Switch all effects ON
    APPLY_PART_LOAD_COP_TO_CORE     = True,
    APPLY_LATENT_LOAD_TO_CORE       = True,
    APPLY_HX_AIR_PRESSURE_TO_FAN    = True,
    APPLY_HX_WATER_PRESSURE_TO_PUMP = True,
    APPLY_HX_UA_TO_CAPACITY         = True,
)

result = run_scenario_model(
    output_dir     = "my_output",
    axis_mode      = "one_strategy",
    bldg           = BuildingSpec(conditioned_area_m2=2000, floors=3, n_spaces=20),
    cfg            = cfg,
    fixed_severity = "Moderate",
    fixed_climate  = "C0_Baseline",
)
```

