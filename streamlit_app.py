"""
HVAC ROM-Degradation Suite  ·  Redesigned UI
============================================
All calculations identical to original engine.
Only the interface has been reorganised and made friendlier.

Run:  streamlit run streamlit_app.py
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st

# Silence Streamlit use_container_width deprecation warnings (Streamlit 2026)
import logging as _logging
_logging.getLogger('streamlit').setLevel(_logging.ERROR)

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False; px = go = None

from hvac_v3_engine import (
    BuildingSpec, HVACConfig,
    HVAC_PRESETS, SCENARIOS, SEVERITY_LEVELS, CLIMATE_LEVELS,
    run_scenario_model, train_surrogate_models,
    run_early_sensitivity_analysis, run_robustness_analysis,
)
from report_addons import (
    read_weather_upload, build_detailed_tables, save_detailed_outputs,
    load_validation_file, build_validation_comparison,
    create_zip_from_folder, find_result_paths,
    setup_to_json_bytes, setup_from_upload,
    build_heat_exchanger_diagnostics, build_part_load_curve_analysis,
    build_latent_load_analysis, build_native_zone_load_table,
    build_formal_validation_metrics, build_global_sensitivity_from_samples,
    build_operation_schedule_template, validate_operation_schedule,
    run_multi_objective_search, build_advanced_control_candidates,
    build_control_objective_table, build_mpc_experimental_template,
    build_rl_experimental_dataset_spec,
)

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="HVAC EMS Simulator", page_icon="❄️",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""<style>
.stApp{background:linear-gradient(160deg,#06111f 0%,#0d1d35 60%,#111a2e 100%);}
.block-container{padding-top:1rem;padding-bottom:2rem;max-width:1380px;}
h1,h2,h3,h4,h5,h6,p,label,span{color:#e8f0fb;}
[data-testid="stHeader"]{background:transparent;}
div[data-baseweb="tab-list"]{gap:.4rem;border-bottom:1px solid rgba(255,255,255,.12);padding-bottom:.2rem;}
button[data-baseweb="tab"]{background:rgba(255,255,255,.04)!important;border-radius:12px 12px 0 0!important;
  padding:.6rem .95rem!important;font-weight:700!important;font-size:.88rem!important;
  border:1px solid rgba(255,255,255,.07)!important;color:#aec4e0!important;}
button[data-baseweb="tab"][aria-selected="true"]{color:#5dddff!important;
  border-bottom:2.5px solid #5dddff!important;background:rgba(93,221,255,.08)!important;}
div[data-testid="stMetric"]{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.10);
  border-radius:14px;padding:.6rem .8rem;}
div[data-testid="stMetric"] label{color:#8ca8c8!important;font-size:.82rem!important;}
div[data-testid="stMetric"] div[data-testid="stMetricValue"]{color:#e8f0fb!important;font-size:1.45rem!important;font-weight:700!important;}
div[data-testid="stExpander"]{border:1px solid rgba(255,255,255,.09);border-radius:14px;
  background:rgba(255,255,255,.03);margin-bottom:.7rem;}
div.stButton>button{border-radius:12px!important;font-weight:700!important;border:1px solid rgba(255,255,255,.18)!important;}
div[data-testid="stDataFrame"]{border:1px solid rgba(255,255,255,.09);border-radius:12px;overflow:hidden;}
.kpi-card{background:linear-gradient(135deg,rgba(30,50,80,.7),rgba(20,35,60,.7));
  border:1px solid rgba(93,221,255,.18);border-radius:16px;padding:1rem 1.2rem;
  text-align:center;margin-bottom:.5rem;}
.kpi-val{font-size:2rem;font-weight:800;}
.kpi-lbl{font-size:.82rem;color:#8ca8c8;margin-top:.25rem;}
.kpi-unit{font-size:.75rem;color:#7090b0;}
.fx-badge{display:inline-block;padding:.2rem .6rem;border-radius:8px;font-size:.75rem;font-weight:700;margin:.15rem .1rem;}
.fx-on{background:rgba(0,200,120,.15);color:#00c878;border:1px solid rgba(0,200,120,.3);}
.fx-off{background:rgba(200,60,60,.08);color:#c06060;border:1px solid rgba(200,60,60,.2);}
.hint-box{background:rgba(93,221,255,.06);border-left:3px solid #5dddff;border-radius:0 8px 8px 0;
  padding:.5rem .8rem;font-size:.88rem;color:#b0cce0;margin-bottom:.8rem;}
</style>""", unsafe_allow_html=True)

st.markdown("""
<div style="padding:.4rem 0 1rem 0;">
  <div style="font-size:2.4rem;font-weight:850;letter-spacing:-.03em;color:#f0f6ff;">❄️ HVAC EMS Simulator</div>
  <div style="color:#7aa0c0;font-size:.95rem;margin-top:.3rem;max-width:900px;">
    Energy Management Strategy simulation · Degradation modelling ·
    Multi-scenario comparison · 4-objective optimisation (Energy · Comfort · Carbon · Degradation)
  </div>
</div>""", unsafe_allow_html=True)

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def S(k, d=None): return st.session_state.get(k, d)

def hint(txt):
    st.markdown(f'<div class="hint-box">{txt}</div>', unsafe_allow_html=True)

def kpi_card(label, value, unit, color="#5dddff"):
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-val" style="color:{color};">{value}</div>
      <div class="kpi-lbl">{label}</div>
      <div class="kpi-unit">{unit}</div>
    </div>""", unsafe_allow_html=True)

def dl_btn(path, label, key=None):
    p = Path(path)
    if p.exists() and p.is_file():
        with p.open("rb") as f:
            st.download_button(label, f.read(), file_name=p.name, key=key or f"dl_{hash(str(p))}")

TIME_STEP_MAP = {"Daily (fastest)":24.0,"12-hour":12.0,"6-hour":6.0,"3-hour":3.0,"Hourly (slow)":1.0}

PRESETS = {
    "🏫 University": dict(building_type="Educational / University building",location="New Mansoura, Egypt",
        area_m2=2000.0,floors=3,n_spaces=20,occupancy_density=0.08,lighting_w_m2=10.0,equipment_w_m2=8.0,
        sensible_w_per_person=75.0,airflow_m3h_m2=4.0,infiltration_ach=0.50,cooling_w_m2=100.0,heating_w_m2=55.0,
        wall_u=0.60,roof_u=0.35,window_u=2.70,shgc=0.35,glazing_ratio=0.30,
        hvac_system_type="Chiller_AHU",years=20,co2_factor=0.536),
    "🏢 Office": dict(building_type="Office Building",location="User-defined",
        area_m2=3000.0,floors=5,n_spaces=35,occupancy_density=0.06,lighting_w_m2=9.0,equipment_w_m2=11.0,
        sensible_w_per_person=70.0,airflow_m3h_m2=3.5,infiltration_ach=0.40,cooling_w_m2=95.0,heating_w_m2=50.0,
        wall_u=0.55,roof_u=0.30,window_u=2.50,shgc=0.32,glazing_ratio=0.35,
        hvac_system_type="VRF",years=20,co2_factor=0.400),
    "🏥 Hospital": dict(building_type="Healthcare / Hospital",location="User-defined",
        area_m2=5000.0,floors=4,n_spaces=50,occupancy_density=0.10,lighting_w_m2=14.0,equipment_w_m2=15.0,
        sensible_w_per_person=80.0,airflow_m3h_m2=6.0,infiltration_ach=0.30,cooling_w_m2=120.0,heating_w_m2=70.0,
        wall_u=0.50,roof_u=0.28,window_u=2.20,shgc=0.28,glazing_ratio=0.25,
        hvac_system_type="Chiller_AHU",years=20,co2_factor=0.400),
    "🏭 Industrial": dict(building_type="Industrial",location="User-defined",
        area_m2=1500.0,floors=1,n_spaces=8,occupancy_density=0.03,lighting_w_m2=8.0,equipment_w_m2=20.0,
        sensible_w_per_person=90.0,airflow_m3h_m2=8.0,infiltration_ach=0.80,cooling_w_m2=80.0,heating_w_m2=45.0,
        wall_u=0.80,roof_u=0.50,window_u=3.00,shgc=0.45,glazing_ratio=0.15,
        hvac_system_type="Packaged_DX",years=20,co2_factor=0.500),
}

EFFECT_INFO = {
    "APPLY_PART_LOAD_COP_TO_CORE":     {"label":"Part-load COP curve","icon":"📉","kpi":"↑ Energy · ↑ Carbon",
        "desc":"COP degrades when chiller runs below full load. Uses PLR polynomial curve (centrifugal chiller typical)."},
    "APPLY_LATENT_LOAD_TO_CORE":       {"label":"Latent moisture load","icon":"💧","kpi":"↑ Energy · ↑ Carbon",
        "desc":"Outdoor humidity adds latent cooling load for dehumidification. Important in hot-humid climates."},
    "APPLY_HX_AIR_PRESSURE_TO_FAN":    {"label":"HX air fouling → fan","icon":"🌬️","kpi":"↑ Energy · ↑ Carbon · ↑ Degradation",
        "desc":"HX fouling raises air-side pressure drop → fan works harder, directly coupling degradation to energy."},
    "APPLY_HX_WATER_PRESSURE_TO_PUMP": {"label":"HX water fouling → pump","icon":"🔧","kpi":"↑ Energy · ↑ Carbon",
        "desc":"Fouling increases hydraulic resistance in the water circuit. Pump power rises with flow squared."},
    "APPLY_HX_UA_TO_CAPACITY":         {"label":"HX UA loss → capacity","icon":"🔥","kpi":"↑ Comfort deviation · ↑ Degradation",
        "desc":"Fouling reduces overall heat transfer (UA). Limited capacity means unmet load and comfort penalty."},
    "APPLY_NATIVE_ZONE_LOADS":         {"label":"Zone-by-zone loads","icon":"🗂️","kpi":"All KPIs (fidelity increase)",
        "desc":"Computes loads zone-by-zone using individual occupancy/area schedules instead of whole-building aggregate."},
}

SCENARIO_DESC = {
    "S0":"❌ Unaware — No degradation awareness, fixed calendar maintenance",
    "S1":"⚠️  Reactive — Fix only when threshold exceeded",
    "S2":"📅 Preventive — Regular scheduled maintenance intervals",
    "S3":"🤖 Predictive (APO) — Optimizer adjusts setpoint + airflow every step",
}
CLIMATE_DESC = {
    "C0_Baseline":"Current climate (no extra warming)",
    "C1_Warm":"+1.5°C shift, mild warming trend",
    "C2_Heatwave":"Extreme summer heat-pulse events",
    "C3_FutureHot":"+4°C shift, rapid future warming",
}
SEVERITY_DESC = {
    "Mild":"Low fouling rates, easier environment",
    "Moderate":"Typical university/office conditions",
    "Severe":"Dusty/industrial environment, higher fouling",
    "High":"Extreme degradation environment",
}

def apply_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v

# ─── SESSION STATE DEFAULTS ────────────────────────────────────────────────────
if "preset_applied" not in st.session_state:
    apply_preset("🏫 University"); st.session_state["preset_applied"] = True

DEFAULTS = dict(
    years=20, time_step_label="Daily (fastest)", degradation_model="physics",
    t_set=23.0,t_sp_min=21.0,t_sp_max=26.0,af_min=0.55,af_max=1.0,
    cop_aging_rate=0.005,rf_star=2e-4,b_foul=0.015,dust_rate=1.2,
    k_clog=6.0,deg_trigger=0.55,e_price=0.12,co2_factor=0.536,
    cost_filter=50.0,cost_hx=300.0,filter_interval=90,hx_interval=180,
    cop_cool_nom=4.5,cop_heat_nom=3.2,fan_eff=0.70,
    pump_specific_w_m2=1.30,auxiliary_w_m2=0.55,
    dp_clean=150.0,dp_warn=320.0,dp_thresh=420.0,dp_max=450.0,
    use_hvac_preset=True,
    APPLY_PART_LOAD_COP_TO_CORE=True,APPLY_LATENT_LOAD_TO_CORE=True,
    APPLY_HX_AIR_PRESSURE_TO_FAN=True,APPLY_HX_WATER_PRESSURE_TO_PUMP=True,
    APPLY_HX_UA_TO_CAPACITY=True,APPLY_NATIVE_ZONE_LOADS=False,
    PLR_CURVE_TYPE="Quadratic",PLR_A=0.85,PLR_B=0.25,PLR_C=-0.10,PLR_D=0.0,
    PLR_MIN_MODIFIER=0.55,PLR_MAX_MODIFIER=1.15,
    INDOOR_RH_TARGET_PCT=50.0,LATENT_VENTILATION_FRACTION=0.35,FLOOR_TO_FLOOR_M=3.2,
    HX_AIR_FOULING_FACTOR=0.75,HX_WATER_DP_CLEAN_KPA=35.0,
    HX_WATER_FLOW_M3H=0.0,HX_WATER_FLOW_NOM_M3H=0.0,
    HX_WATER_FOULING_FACTOR=0.35,HX_PUMP_EFF=0.65,
    HX_CHW_DT_K=5.0,HX_HW_DT_K=10.0,HX_UA_CLEAN_KW_K=0.0,
    HX_UA_LOSS_FACTOR=0.30,HX_LMTD_CORRECTION=0.90,
    EMS_MODE="Disabled",EMS_OCC_CONTROL=False,EMS_NIGHT_SETBACK=False,
    EMS_DEMAND_RESPONSE=False,EMS_ECONOMIZER=False,EMS_OPTIMUM_START=False,
    EMS_CUSTOM_SCHEDULE_ENABLED=False,
    EMS_LOW_OCC_THRESHOLD=0.25,EMS_LOW_OCC_AIRFLOW_FACTOR=0.65,EMS_LOW_OCC_SETPOINT_SHIFT_C=1.0,
    EMS_NIGHT_START_HOUR=19.0,EMS_NIGHT_END_HOUR=6.0,EMS_NIGHT_SETPOINT_SHIFT_C=2.0,EMS_NIGHT_AIRFLOW_FACTOR=0.55,
    EMS_DR_START_HOUR=13.0,EMS_DR_END_HOUR=17.0,EMS_DR_SETPOINT_SHIFT_C=1.5,EMS_DR_AIRFLOW_REDUCTION=0.15,
    EMS_ECONOMIZER_TEMP_LOW_C=16.0,EMS_ECONOMIZER_TEMP_HIGH_C=22.0,EMS_ECONOMIZER_COOLING_REDUCTION=0.20,
    EMS_OPTIMUM_START_HOUR=7.0,EMS_PRECOOL_SHIFT_C=-0.8,
    linear_deg_per_day=0.00012,exp_deg_rate_per_day=0.00018,
    apo_pop=12,apo_iters=6,
    w_energy=0.35,w_degrad=0.25,w_comfort=0.25,w_carbon=0.15,
)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

# ─── BUILD ENGINE OBJECTS ──────────────────────────────────────────────────────
def build_bldg():
    return BuildingSpec(
        building_type=S("building_type","Educational / University building"),
        location=S("location","User-defined"),
        conditioned_area_m2=float(S("area_m2",2000.0)),
        floors=int(S("floors",3)), n_spaces=int(S("n_spaces",20)),
        occupancy_density_p_m2=float(S("occupancy_density",0.08)),
        lighting_w_m2=float(S("lighting_w_m2",10.0)),
        equipment_w_m2=float(S("equipment_w_m2",8.0)),
        airflow_m3h_m2=float(S("airflow_m3h_m2",4.0)),
        infiltration_ach=float(S("infiltration_ach",0.5)),
        sensible_w_per_person=float(S("sensible_w_per_person",75.0)),
        cooling_intensity_w_m2=float(S("cooling_w_m2",100.0)),
        heating_intensity_w_m2=float(S("heating_w_m2",55.0)),
        wall_u=float(S("wall_u",0.6)), roof_u=float(S("roof_u",0.35)),
        window_u=float(S("window_u",2.7)), shgc=float(S("shgc",0.35)),
        glazing_ratio=float(S("glazing_ratio",0.30)),
    )

def build_cfg(ts=24.0):
    return HVACConfig(
        years=int(S("years",20)), hvac_system_type=S("hvac_system_type","Chiller_AHU"),
        COP_COOL_NOM=float(S("cop_cool_nom",4.5)), COP_HEAT_NOM=float(S("cop_heat_nom",3.2)),
        FAN_EFF=float(S("fan_eff",0.70)), PUMP_SPECIFIC_W_M2=float(S("pump_specific_w_m2",1.30)),
        AUXILIARY_W_M2=float(S("auxiliary_w_m2",0.55)),
        T_SET=float(S("t_set",23.0)), T_SP_MIN=float(S("t_sp_min",21.0)),
        T_SP_MAX=float(S("t_sp_max",26.0)), AF_MIN=float(S("af_min",0.55)), AF_MAX=float(S("af_max",1.0)),
        DP_CLEAN=float(S("dp_clean",150.0)), DP_WARN=float(S("dp_warn",320.0)),
        DP_THRESH=float(S("dp_thresh",420.0)), DP_MAX=float(S("dp_max",450.0)),
        COP_AGING_RATE=float(S("cop_aging_rate",0.005)),
        RF_STAR=float(S("rf_star",2e-4)), B_FOUL=float(S("b_foul",0.015)),
        DUST_RATE=float(S("dust_rate",1.2)), K_CLOG=float(S("k_clog",6.0)),
        DEG_TRIGGER=float(S("deg_trigger",0.55)),
        E_PRICE=float(S("e_price",0.12)), CO2_FACTOR=float(S("co2_factor",0.536)),
        COST_FILTER=float(S("cost_filter",50.0)), COST_HX=float(S("cost_hx",300.0)),
        FILTER_INTERVAL=int(S("filter_interval",90)), HX_INTERVAL=int(S("hx_interval",180)),
        W_ENERGY=float(S("w_energy",0.35)), W_DEGRAD=float(S("w_degrad",0.25)),
        W_COMFORT=float(S("w_comfort",0.25)), W_CARBON=float(S("w_carbon",0.15)),
        degradation_model=S("degradation_model","physics"),
        LINEAR_DEG_PER_DAY=float(S("linear_deg_per_day",0.00012)),
        EXP_DEG_RATE_PER_DAY=float(S("exp_deg_rate_per_day",0.00018)),
        TIME_STEP_HOURS=ts, USE_HVAC_PRESET=bool(S("use_hvac_preset",True)),
        APO_POP=int(S("apo_pop",12)), APO_ITERS=int(S("apo_iters",6)),
        APPLY_PART_LOAD_COP_TO_CORE=bool(S("APPLY_PART_LOAD_COP_TO_CORE",True)),
        APPLY_LATENT_LOAD_TO_CORE=bool(S("APPLY_LATENT_LOAD_TO_CORE",True)),
        APPLY_HX_AIR_PRESSURE_TO_FAN=bool(S("APPLY_HX_AIR_PRESSURE_TO_FAN",True)),
        APPLY_HX_WATER_PRESSURE_TO_PUMP=bool(S("APPLY_HX_WATER_PRESSURE_TO_PUMP",True)),
        APPLY_HX_UA_TO_CAPACITY=bool(S("APPLY_HX_UA_TO_CAPACITY",True)),
        APPLY_NATIVE_ZONE_LOADS=bool(S("APPLY_NATIVE_ZONE_LOADS",False)),
        PLR_CURVE_TYPE=S("PLR_CURVE_TYPE","Quadratic"),
        PLR_A=float(S("PLR_A",0.85)), PLR_B=float(S("PLR_B",0.25)),
        PLR_C=float(S("PLR_C",-0.10)), PLR_D=float(S("PLR_D",0.0)),
        PLR_MIN_MODIFIER=float(S("PLR_MIN_MODIFIER",0.55)),
        PLR_MAX_MODIFIER=float(S("PLR_MAX_MODIFIER",1.15)),
        INDOOR_RH_TARGET_PCT=float(S("INDOOR_RH_TARGET_PCT",50.0)),
        LATENT_VENTILATION_FRACTION=float(S("LATENT_VENTILATION_FRACTION",0.35)),
        FLOOR_TO_FLOOR_M=float(S("FLOOR_TO_FLOOR_M",3.2)),
        HX_AIR_FOULING_FACTOR=float(S("HX_AIR_FOULING_FACTOR",0.75)),
        HX_WATER_DP_CLEAN_KPA=float(S("HX_WATER_DP_CLEAN_KPA",35.0)),
        HX_WATER_FLOW_M3H=float(S("HX_WATER_FLOW_M3H",0.0)),
        HX_WATER_FLOW_NOM_M3H=float(S("HX_WATER_FLOW_NOM_M3H",0.0)),
        HX_WATER_FOULING_FACTOR=float(S("HX_WATER_FOULING_FACTOR",0.35)),
        HX_PUMP_EFF=float(S("HX_PUMP_EFF",0.65)),
        HX_CHW_DT_K=float(S("HX_CHW_DT_K",5.0)), HX_HW_DT_K=float(S("HX_HW_DT_K",10.0)),
        HX_UA_CLEAN_KW_K=float(S("HX_UA_CLEAN_KW_K",0.0)),
        HX_UA_LOSS_FACTOR=float(S("HX_UA_LOSS_FACTOR",0.30)),
        HX_LMTD_CORRECTION=float(S("HX_LMTD_CORRECTION",0.90)),
        EMS_MODE=S("EMS_MODE","Disabled"),
        EMS_OCC_CONTROL=bool(S("EMS_OCC_CONTROL",False)),
        EMS_NIGHT_SETBACK=bool(S("EMS_NIGHT_SETBACK",False)),
        EMS_DEMAND_RESPONSE=bool(S("EMS_DEMAND_RESPONSE",False)),
        EMS_ECONOMIZER=bool(S("EMS_ECONOMIZER",False)),
        EMS_OPTIMUM_START=bool(S("EMS_OPTIMUM_START",False)),
        EMS_CUSTOM_SCHEDULE_ENABLED=bool(S("EMS_CUSTOM_SCHEDULE_ENABLED",False)),
        EMS_LOW_OCC_THRESHOLD=float(S("EMS_LOW_OCC_THRESHOLD",0.25)),
        EMS_LOW_OCC_AIRFLOW_FACTOR=float(S("EMS_LOW_OCC_AIRFLOW_FACTOR",0.65)),
        EMS_LOW_OCC_SETPOINT_SHIFT_C=float(S("EMS_LOW_OCC_SETPOINT_SHIFT_C",1.0)),
        EMS_NIGHT_START_HOUR=float(S("EMS_NIGHT_START_HOUR",19.0)),
        EMS_NIGHT_END_HOUR=float(S("EMS_NIGHT_END_HOUR",6.0)),
        EMS_NIGHT_SETPOINT_SHIFT_C=float(S("EMS_NIGHT_SETPOINT_SHIFT_C",2.0)),
        EMS_NIGHT_AIRFLOW_FACTOR=float(S("EMS_NIGHT_AIRFLOW_FACTOR",0.55)),
        EMS_DR_START_HOUR=float(S("EMS_DR_START_HOUR",13.0)),
        EMS_DR_END_HOUR=float(S("EMS_DR_END_HOUR",17.0)),
        EMS_DR_SETPOINT_SHIFT_C=float(S("EMS_DR_SETPOINT_SHIFT_C",1.5)),
        EMS_DR_AIRFLOW_REDUCTION=float(S("EMS_DR_AIRFLOW_REDUCTION",0.15)),
        EMS_ECONOMIZER_TEMP_LOW_C=float(S("EMS_ECONOMIZER_TEMP_LOW_C",16.0)),
        EMS_ECONOMIZER_TEMP_HIGH_C=float(S("EMS_ECONOMIZER_TEMP_HIGH_C",22.0)),
        EMS_ECONOMIZER_COOLING_REDUCTION=float(S("EMS_ECONOMIZER_COOLING_REDUCTION",0.20)),
        EMS_OPTIMUM_START_HOUR=float(S("EMS_OPTIMUM_START_HOUR",7.0)),
        EMS_PRECOOL_SHIFT_C=float(S("EMS_PRECOOL_SHIFT_C",-0.8)),
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "🏠 Quick Start",
    "🏢 Building",
    "❄️ HVAC & Degradation",
    "⚡ Physics Effects",
    "🎮 EMS Control",
    "▶️ Run & Results",
    "📊 Charts & Trends",
    "🔬 Advanced Analysis",
    "📥 Export & Tools",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 0  ·  QUICK START
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("### Choose a building preset to get started instantly")
    hint("Click any preset → all parameters will be pre-filled. Then go to <b>▶️ Run & Results</b> and press <b>Run Simulation</b>.")

    p_cols = st.columns(len(PRESETS))
    for col, name in zip(p_cols, PRESETS):
        with col:
            if st.button(name, width="stretch", key=f"preset_{name}"):
                apply_preset(name); st.success(f"Applied: {name}"); st.rerun()

    st.markdown("---")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("### Current configuration at a glance")
        n_fx = sum(bool(S(k)) for k in EFFECT_INFO)
        snap = {
            "🏢 Building": f"{S('building_type','?')} — {S('area_m2',0):.0f} m², {S('floors',0)} floors",
            "❄️ HVAC": f"{S('hvac_system_type','?')} — COP cool {S('cop_cool_nom',0):.1f}",
            "📅 Simulation": f"{S('years',0)} years  ·  {S('time_step_label','Daily (fastest)')}",
            "🌍 Grid CO₂": f"{S('co2_factor',0):.3f} kgCO₂/kWh",
            "⚡ Physics effects": f"{n_fx} / {len(EFFECT_INFO)} enabled",
            "🎮 EMS mode": S("EMS_MODE","Disabled"),
        }
        for k, v in snap.items():
            st.markdown(f"**{k}:** {v}")

    with c2:
        st.markdown("### 📋 Steps to run")
        st.markdown("""
1. **Choose preset** above, or customise in 🏢 Building
2. **Toggle effects** in ⚡ Physics Effects
3. **Set EMS** in 🎮 EMS Control *(optional)*
4. **Click Run** in ▶️ Run & Results
5. **View** 4 KPI cards + charts instantly
        """)
        st.markdown("---")
        st.markdown("**💾 Save / load settings**")
        def _snap():
            return {k: S(k) for k in ["building_type","location","area_m2","floors","n_spaces",
                "occupancy_density","lighting_w_m2","equipment_w_m2","sensible_w_per_person",
                "airflow_m3h_m2","infiltration_ach","cooling_w_m2","heating_w_m2",
                "wall_u","roof_u","window_u","shgc","glazing_ratio",
                "hvac_system_type","years","time_step_label","co2_factor","e_price",
                "degradation_model","APPLY_PART_LOAD_COP_TO_CORE","APPLY_LATENT_LOAD_TO_CORE",
                "APPLY_HX_AIR_PRESSURE_TO_FAN","APPLY_HX_WATER_PRESSURE_TO_PUMP",
                "APPLY_HX_UA_TO_CAPACITY","EMS_MODE"]}
        st.download_button("⬇️ Save setup JSON",
                           json.dumps(_snap(), indent=2).encode(), "hvac_setup.json", key="qs_save")
        upl = st.file_uploader("⬆️ Load setup JSON", type=["json"], key="qs_load")
        if upl:
            try:
                data = setup_from_upload(upl)
                for k, v in (data.items() if isinstance(data, dict) else {}):
                    if isinstance(v, dict):
                        for kk, vv in v.items(): st.session_state[kk] = vv
                    else: st.session_state[k] = v
                st.success("Setup loaded!"); st.rerun()
            except Exception as e: st.error(str(e))

    st.markdown("---")
    st.markdown("### 📖 What do the 4 KPIs mean?")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown("**⚡ Energy Consumption (MWh)**")
        st.caption("Total electricity used by compressor, fans, pumps and auxiliary systems over the full simulation horizon.")
    with k2:
        st.markdown("**🌡️ Comfort Deviation (°C)**")
        st.caption("Average gap between zone temperature and setpoint. High degradation → reduced capacity → more discomfort.")
    with k3:
        st.markdown("**🌿 Carbon Emissions (tonne CO₂)**")
        st.caption("Energy × grid emission factor. Egypt default = 0.536 kgCO₂/kWh.")
    with k4:
        st.markdown("**🔧 Degradation Index (−)**")
        st.caption("Composite 0→1 index combining HX fouling (Rf) and filter pressure drop (ΔP). Drives all other KPIs.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1  ·  BUILDING
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("### 🏢 Building Properties")
    hint("These values define your building's size, use and thermal envelope. All fields have sensible defaults — only change what applies.")

    c1, c2 = st.columns(2)
    st.session_state["building_type"] = c1.text_input("Building type / name",
        value=S("building_type","Educational / University building"),
        help="Label used in report headings only")
    st.session_state["location"] = c2.text_input("Location / city",
        value=S("location","User-defined"),
        help="Label used in outputs — does not affect weather unless you upload a file")

    st.markdown("#### 📐 Size & geometry")
    c1, c2, c3 = st.columns(3)
    st.session_state["area_m2"]    = c1.number_input("Total conditioned area (m²)", 100.0, 100000.0, float(S("area_m2",2000.0)), 100.0, help="Sum of all air-conditioned floor areas")
    st.session_state["floors"]     = c2.number_input("Number of floors", 1, 100, int(S("floors",3)), 1)
    st.session_state["n_spaces"]   = c3.number_input("Number of zones / spaces", 1, 500, int(S("n_spaces",20)), 1, help="Used in reporting. For zone-by-zone go to ⚡ Physics Effects")

    st.markdown("#### 🌡️ Thermal envelope")
    hint("U-values (W/m²K): lower = better insulation. SHGC = solar heat gain coefficient (0 → 1).")
    c1, c2, c3, c4, c5 = st.columns(5)
    st.session_state["wall_u"]        = c1.number_input("Wall U (W/m²K)",   0.05, 3.0,  float(S("wall_u",0.60)),   0.05, help="Good: 0.3–0.6")
    st.session_state["roof_u"]        = c2.number_input("Roof U (W/m²K)",   0.05, 3.0,  float(S("roof_u",0.35)),   0.05, help="Good: 0.15–0.35")
    st.session_state["window_u"]      = c3.number_input("Window U (W/m²K)", 0.5,  6.0,  float(S("window_u",2.70)), 0.1,  help="Double-glazed ≈ 2.7")
    st.session_state["shgc"]          = c4.number_input("SHGC",             0.05, 0.95, float(S("shgc",0.35)),     0.01, help="Low-e glass ≈ 0.25–0.35")
    st.session_state["glazing_ratio"] = c5.number_input("Glazing ratio",    0.05, 0.90, float(S("glazing_ratio",0.30)), 0.01)

    st.markdown("#### 👥 Occupancy & internal gains")
    c1, c2, c3, c4 = st.columns(4)
    st.session_state["occupancy_density"]    = c1.number_input("Occupancy (p/m²)", 0.001, 1.0, float(S("occupancy_density",0.08)),  0.005, format="%.3f", help="University hall: 0.10–0.15 · Office: 0.05–0.08")
    st.session_state["lighting_w_m2"]        = c2.number_input("Lighting (W/m²)",  0.0, 50.0,  float(S("lighting_w_m2",10.0)),       0.5,  help="LED: 8–10 · Fluorescent: 15–20")
    st.session_state["equipment_w_m2"]       = c3.number_input("Equipment (W/m²)", 0.0, 80.0,  float(S("equipment_w_m2",8.0)),        0.5,  help="Office PCs: 8–12 · Server room: 50+")
    st.session_state["sensible_w_per_person"]= c4.number_input("Heat/person (W)",  10.0,200.0, float(S("sensible_w_per_person",75.0)),5.0,  help="Seated office ≈ 75 W")

    st.markdown("#### 💨 Airflow & loads")
    c1, c2, c3, c4 = st.columns(4)
    st.session_state["airflow_m3h_m2"] = c1.number_input("Ventilation (m³/h·m²)", 0.5,  20.0, float(S("airflow_m3h_m2",4.0)), 0.5,  help="Typical office: 3–5")
    st.session_state["infiltration_ach"]= c2.number_input("Infiltration (ACH)",    0.0,  5.0,  float(S("infiltration_ach",0.5)),0.05, help="Tight: 0.2–0.4 · Leaky: 0.6+")
    st.session_state["cooling_w_m2"]   = c3.number_input("Cooling design (W/m²)", 10.0,400.0, float(S("cooling_w_m2",100.0)),  5.0,  help="Typical office: 80–120 W/m²")
    st.session_state["heating_w_m2"]   = c4.number_input("Heating design (W/m²)", 5.0, 200.0, float(S("heating_w_m2",55.0)),   5.0,  help="Mild climate: 30–60 W/m²")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2  ·  HVAC & DEGRADATION
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("### ❄️ HVAC System & Degradation Parameters")

    # System-type selector
    st.markdown("#### Choose HVAC system type")
    sys_meta = {
        "Chiller_AHU" :("🏭","Chiller + AHU",   "Best for large buildings"),
        "VRF"         :("🔁","VRF System",       "Flexible multi-zone"),
        "Packaged_DX" :("📦","Packaged DX",       "Small/medium buildings"),
        "Heat_Pump"   :("♻️","Heat Pump",         "Efficient heat + cool"),
        "Custom"      :("⚙️","Custom",            "Enter your own values"),
    }
    scols = st.columns(len(sys_meta))
    for col, (sys_key, (icon, name, desc)) in zip(scols, sys_meta.items()):
        is_sel = S("hvac_system_type") == sys_key
        with col:
            if st.button(f"{icon} {name}\n{desc}", width="stretch",
                         key=f"sys_{sys_key}", type="primary" if is_sel else "secondary"):
                st.session_state["hvac_system_type"] = sys_key
                st.session_state["use_hvac_preset"]  = (sys_key != "Custom")
                st.rerun()

    st.markdown("#### ⚡ Performance parameters")
    use_p = st.checkbox("Auto-fill from system preset (uncheck to enter custom values)",
                        value=bool(S("use_hvac_preset",True)), key="use_hvac_preset")
    pv = HVAC_PRESETS.get(S("hvac_system_type","Chiller_AHU"), HVAC_PRESETS["Chiller_AHU"])
    if use_p:
        for k in ("cop_cool_nom","cop_heat_nom","fan_eff","pump_specific_w_m2","auxiliary_w_m2"):
            st.session_state[k] = pv[k.upper()]
        mc1,mc2,mc3,mc4,mc5 = st.columns(5)
        mc1.metric("Cooling COP",    f'{S("cop_cool_nom",4.5):.2f}')
        mc2.metric("Heating COP",    f'{S("cop_heat_nom",3.2):.2f}')
        mc3.metric("Fan efficiency", f'{S("fan_eff",0.70):.2f}')
        mc4.metric("Pump W/m²",      f'{S("pump_specific_w_m2",1.30):.2f}')
        mc5.metric("Auxiliary W/m²", f'{S("auxiliary_w_m2",0.55):.2f}')
    else:
        nc1,nc2,nc3,nc4,nc5 = st.columns(5)
        st.session_state["cop_cool_nom"]       = nc1.number_input("Cooling COP",      0.8,12.0, float(S("cop_cool_nom",4.5)),   0.1)
        st.session_state["cop_heat_nom"]       = nc2.number_input("Heating COP",      0.8, 8.0, float(S("cop_heat_nom",3.2)),   0.1)
        st.session_state["fan_eff"]            = nc3.number_input("Fan efficiency",   0.1, 0.98,float(S("fan_eff",0.70)),       0.01)
        st.session_state["pump_specific_w_m2"] = nc4.number_input("Pump W/m²",       0.0,10.0, float(S("pump_specific_w_m2",1.30)),0.05)
        st.session_state["auxiliary_w_m2"]     = nc5.number_input("Auxiliary W/m²",  0.0, 5.0, float(S("auxiliary_w_m2",0.55)), 0.05)

    st.markdown("#### 🕰️ Simulation timeline")
    c1, c2, c3, c4 = st.columns(4)
    st.session_state["years"]             = c1.slider("Horizon (years)", 1, 30, int(S("years",20)), help="20 years = PhD standard. Use 5 for quick tests.")
    st.session_state["time_step_label"]   = c2.selectbox("Time step", list(TIME_STEP_MAP.keys()), index=0,
        help="Daily is recommended — 24× faster than hourly with minimal accuracy loss")
    st.session_state["degradation_model"]= c3.selectbox("Degradation model",
        ["physics","linear_ts","exponential_ts"],
        format_func=lambda x: {"physics":"Physics (Kern-Seaton + dust)",
                                "linear_ts":"Linear time-series",
                                "exponential_ts":"Exponential time-series"}[x])
    st.session_state["t_set"] = c4.number_input("Comfort setpoint (°C)", 16.0, 30.0, float(S("t_set",23.0)), 0.5)

    st.markdown("#### 🔩 Degradation physics")
    hint("Control how fast the HX fouls (Rf) and the filter clogs (ΔP). Defaults represent standard university/office conditions.")
    c1,c2,c3,c4 = st.columns(4)
    st.session_state["rf_star"]   = c1.number_input("Fouling asymptote Rf* (m²K/W)", 0.0,0.01, float(S("rf_star",2e-4)),  1e-5, format="%.5f", help="Kern-Seaton max fouling resistance. Typical HX: 1.5–3×10⁻⁴")
    st.session_state["b_foul"]    = c2.number_input("Fouling rate B (day⁻¹)",        0.001,0.1,float(S("b_foul",0.015)), 0.001,format="%.4f", help="Higher → faster approach to Rf*")
    st.session_state["dust_rate"] = c3.number_input("Dust rate (kg/day)",             0.0,20.0, float(S("dust_rate",1.2)),0.1,  help="Filter dust loading. Dusty: 2–3 kg/day")
    st.session_state["k_clog"]    = c4.number_input("Filter clog coeff K",            0.5,30.0, float(S("k_clog",6.0)),  0.5,  help="ΔP = DP_clean + K × dust_kg")

    with st.expander("🔧 Maintenance intervals & costs"):
        c1,c2,c3,c4 = st.columns(4)
        st.session_state["filter_interval"] = c1.number_input("Filter interval (days)",10,365,  int(S("filter_interval",90)), 5)
        st.session_state["hx_interval"]     = c2.number_input("HX interval (days)",   10,730,  int(S("hx_interval",180)),    10)
        st.session_state["cost_filter"]     = c3.number_input("Filter cost (USD)",     0.0,5000.,float(S("cost_filter",50.)),  5.)
        st.session_state["cost_hx"]         = c4.number_input("HX cleaning cost (USD)",0.0,10000.,float(S("cost_hx",300.)),  25.)
        c1,c2,c3 = st.columns(3)
        st.session_state["dp_thresh"]    = c1.number_input("ΔP reactive threshold (Pa)",  100.,1000.,float(S("dp_thresh",420.)),10.)
        st.session_state["dp_warn"]      = c2.number_input("ΔP warning level (Pa)",        50., 900.,float(S("dp_warn",320.)),  10.)
        st.session_state["deg_trigger"]  = c3.number_input("S3 degradation trigger (0–1)", 0.2,  0.9,float(S("deg_trigger",0.55)),0.01,
            help="S3 triggers maintenance when degradation index exceeds this")

    with st.expander("🚀 S3 Optimiser speed settings"):
        hint("Reduce these if S3 is too slow. For final PhD run use POP=18, ITER=10.")
        c1,c2 = st.columns(2)
        st.session_state["apo_pop"]   = c1.slider("APO population",  4,30, int(S("apo_pop",12)),  help="Candidates per iteration")
        st.session_state["apo_iters"] = c2.slider("APO iterations",  2,20, int(S("apo_iters",6)), help="Passes per time step")
        est = S("years",20)*365*S("apo_pop",12)*S("apo_iters",6)
        st.caption(f"Estimated S3 evaluations: **{est:,}** — {'✅ Fast' if est<500000 else '⚠️ May be slow' if est<1500000 else '🐢 Very slow — reduce POP/ITER'}")

    with st.expander("💰 Cost, carbon & setpoint range"):
        c1,c2,c3,c4,c5 = st.columns(5)
        st.session_state["e_price"]    = c1.number_input("Electricity (USD/kWh)",0.01,2.0,  float(S("e_price",0.12)),  0.01)
        st.session_state["co2_factor"] = c2.number_input("Grid CO₂ (kgCO₂/kWh)",0.05,1.5,  float(S("co2_factor",0.536)),0.001,format="%.3f")
        st.session_state["t_sp_min"]   = c3.number_input("Setpoint min (°C)",    14.0,25.0, float(S("t_sp_min",21.0)), 0.5)
        st.session_state["t_sp_max"]   = c4.number_input("Setpoint max (°C)",    20.0,32.0, float(S("t_sp_max",26.0)), 0.5)
        st.session_state["af_min"]     = c5.number_input("Airflow min fraction",  0.1, 1.0,  float(S("af_min",0.55)),  0.05)

    with st.expander("⚖️ Objective weights (S3 optimiser)"):
        hint("Weights define what S3 minimises. Should conceptually sum ≈ 1.0.")
        c1,c2,c3,c4 = st.columns(4)
        st.session_state["w_energy"]  = c1.slider("Energy",      0.0,1.0,float(S("w_energy",0.35)), 0.05)
        st.session_state["w_degrad"]  = c2.slider("Degradation", 0.0,1.0,float(S("w_degrad",0.25)), 0.05)
        st.session_state["w_comfort"] = c3.slider("Comfort",     0.0,1.0,float(S("w_comfort",0.25)),0.05)
        st.session_state["w_carbon"]  = c4.slider("Carbon",      0.0,1.0,float(S("w_carbon",0.15)), 0.05)
        tot = S("w_energy",.35)+S("w_degrad",.25)+S("w_comfort",.25)+S("w_carbon",.15)
        st.caption(f"Weight sum: **{tot:.2f}** {'✅' if abs(tot-1.0)<0.02 else '⚠️ Consider adjusting to sum ≈ 1.0'}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3  ·  PHYSICS EFFECTS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("### ⚡ Coupled Physics Effects")
    hint("""These six switches activate <b>publication-level coupled modules</b>.
    When ON, each effect modifies the main solver and the result propagates automatically to all four KPIs.
    All are ON by default — turn individual ones OFF to isolate their contribution.""")

    for key, info in EFFECT_INFO.items():
        cx, cy = st.columns([0.07, 0.93])
        default_on = (key != "APPLY_NATIVE_ZONE_LOADS")
        current = bool(S(key, default_on))
        with cx:
            new_val = st.checkbox("", value=current, key=f"fxcb_{key}", label_visibility="collapsed")
            st.session_state[key] = new_val
        with cy:
            badge = '<span class="fx-badge fx-on">ON</span>' if new_val else '<span class="fx-badge fx-off">OFF</span>'
            kpi_html = " · ".join(f"<code>{p}</code>" for p in info["kpi"].split(" · "))
            st.markdown(f"""
            <div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.09);
                 border-radius:12px;padding:.6rem 1rem;margin:.2rem 0;">
              <div style="font-size:1.05rem;font-weight:700;color:#d0e8f8;">
                {info['icon']} {info['label']} {badge}</div>
              <div style="color:#8ca8c8;font-size:.88rem;margin-top:.2rem;">{info['desc']}</div>
              <div style="margin-top:.25rem;font-size:.8rem;color:#5dddff;">KPI impact: {kpi_html}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    n_on = sum(bool(S(k)) for k in EFFECT_INFO)
    st.markdown(f"**{n_on} / {len(EFFECT_INFO)} effects active**")
    c1, c2 = st.columns(2)
    if c1.button("✅ Enable all effects", use_container_width=True):
        for k in EFFECT_INFO: st.session_state[k] = True
        st.rerun()
    if c2.button("⬜ Disable all (baseline comparison)", use_container_width=True):
        for k in EFFECT_INFO: st.session_state[k] = False
        st.rerun()

    with st.expander("⚙️ Part-load COP curve  f(PLR) = A + B·PLR + C·PLR² + D·PLR³"):
        c1,c2,c3,c4,c5 = st.columns(5)
        st.session_state["PLR_CURVE_TYPE"]    = c1.selectbox("Curve type",["Quadratic","Linear","Cubic"],key="plrt")
        st.session_state["PLR_A"]             = c2.number_input("A (intercept)", -1.0,2.0,float(S("PLR_A",0.85)),0.01)
        st.session_state["PLR_B"]             = c3.number_input("B (linear)",   -2.0,3.0,float(S("PLR_B",0.25)),0.01)
        st.session_state["PLR_C"]             = c4.number_input("C (quadratic)",-2.0,2.0,float(S("PLR_C",-0.10)),0.01)
        st.session_state["PLR_D"]             = c5.number_input("D (cubic)",    -1.0,1.0,float(S("PLR_D",0.0)),0.01)

    with st.expander("💧 Latent (moisture) load parameters"):
        c1,c2,c3 = st.columns(3)
        st.session_state["INDOOR_RH_TARGET_PCT"]       = c1.number_input("Indoor RH target (%)",   30.,70.,float(S("INDOOR_RH_TARGET_PCT",50.)),1.)
        st.session_state["LATENT_VENTILATION_FRACTION"]= c2.number_input("Ventilation fraction",    0.0, 1.0,float(S("LATENT_VENTILATION_FRACTION",0.35)),0.01)
        st.session_state["FLOOR_TO_FLOOR_M"]           = c3.number_input("Floor-to-floor (m)",     2.0, 6.0,float(S("FLOOR_TO_FLOOR_M",3.2)),0.1)

    with st.expander("🔧 HX hydraulic / heat transfer parameters"):
        c1,c2,c3,c4 = st.columns(4)
        st.session_state["HX_WATER_DP_CLEAN_KPA"]  = c1.number_input("Water ΔP clean (kPa)",  1.,200.,float(S("HX_WATER_DP_CLEAN_KPA",35.)),1.)
        st.session_state["HX_UA_LOSS_FACTOR"]       = c2.number_input("UA capacity loss",       0.,0.9, float(S("HX_UA_LOSS_FACTOR",0.30)),0.01,
            help="Max fraction of heat transfer capacity lost at full degradation")
        st.session_state["HX_PUMP_EFF"]             = c3.number_input("Pump efficiency",         0.1,0.98,float(S("HX_PUMP_EFF",0.65)),0.01)
        st.session_state["HX_CHW_DT_K"]             = c4.number_input("Chilled water ΔT (K)",   1.,20., float(S("HX_CHW_DT_K",5.)),0.5)
        c1,c2,c3,c4 = st.columns(4)
        st.session_state["HX_AIR_FOULING_FACTOR"]   = c1.number_input("Air fouling factor",      0.,2.,  float(S("HX_AIR_FOULING_FACTOR",0.75)),0.05)
        st.session_state["HX_WATER_FOULING_FACTOR"] = c2.number_input("Water fouling factor",    0.,2.,  float(S("HX_WATER_FOULING_FACTOR",0.35)),0.05)
        st.session_state["HX_HW_DT_K"]              = c3.number_input("Hot water ΔT (K)",        1.,30., float(S("HX_HW_DT_K",10.)),0.5)
        st.session_state["HX_LMTD_CORRECTION"]      = c4.number_input("LMTD correction",         0.5,1.0,float(S("HX_LMTD_CORRECTION",0.90)),0.01)



# ─────────────────────────────────────────────────────────────────────────────
# TAB 4  ·  EMS CONTROL
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("### 🎮 Energy Management System (EMS) Control")
    hint("EMS overlays modify setpoint and airflow during simulation. They work on top of the core physics — enabling them will change all 4 KPIs.")

    ems_opts = ["Disabled","Occupancy-based","Night setback","Demand response",
                "Economizer","Optimum start","Smart hybrid","Custom scheduled"]
    st.session_state["EMS_MODE"] = st.selectbox("EMS strategy", ems_opts,
        index=ems_opts.index(S("EMS_MODE","Disabled")),
        help="'Smart hybrid' combines occupancy, night setback and demand response")

    if S("EMS_MODE","Disabled") != "Disabled":
        st.success(f"✅ EMS active: {S('EMS_MODE')}")

    st.markdown("#### Quick toggles")
    c1,c2,c3,c4,c5 = st.columns(5)
    st.session_state["EMS_OCC_CONTROL"]    = c1.toggle("Occupancy reset",   value=bool(S("EMS_OCC_CONTROL",False)),   help="Raise setpoint + reduce airflow when occupancy is low")
    st.session_state["EMS_NIGHT_SETBACK"]  = c2.toggle("Night setback",     value=bool(S("EMS_NIGHT_SETBACK",False)), help="Higher setpoint + less airflow at night")
    st.session_state["EMS_DEMAND_RESPONSE"]= c3.toggle("Demand response",   value=bool(S("EMS_DEMAND_RESPONSE",False)),help="Reduce cooling during peak grid hours")
    st.session_state["EMS_ECONOMIZER"]     = c4.toggle("Economizer",        value=bool(S("EMS_ECONOMIZER",False)),    help="Use cool outdoor air for free cooling")
    st.session_state["EMS_OPTIMUM_START"]  = c5.toggle("Optimum start",     value=bool(S("EMS_OPTIMUM_START",False)), help="Pre-cool before occupancy")

    with st.expander("⚙️ Occupancy-based control"):
        c1,c2,c3 = st.columns(3)
        st.session_state["EMS_LOW_OCC_THRESHOLD"]       = c1.slider("Low occ threshold",    0.0,1.0,float(S("EMS_LOW_OCC_THRESHOLD",0.25)),0.05)
        st.session_state["EMS_LOW_OCC_AIRFLOW_FACTOR"]  = c2.slider("Airflow reduction",    0.1,1.0,float(S("EMS_LOW_OCC_AIRFLOW_FACTOR",0.65)),0.05)
        st.session_state["EMS_LOW_OCC_SETPOINT_SHIFT_C"]= c3.slider("Setpoint shift (°C)", -3.0,5.0,float(S("EMS_LOW_OCC_SETPOINT_SHIFT_C",1.0)),0.25)

    with st.expander("🌙 Night setback"):
        c1,c2,c3,c4 = st.columns(4)
        st.session_state["EMS_NIGHT_START_HOUR"]      = c1.number_input("Night start (hr)", 0.,24.,float(S("EMS_NIGHT_START_HOUR",19.)),0.5)
        st.session_state["EMS_NIGHT_END_HOUR"]        = c2.number_input("Night end (hr)",   0.,24.,float(S("EMS_NIGHT_END_HOUR",6.)),  0.5)
        st.session_state["EMS_NIGHT_SETPOINT_SHIFT_C"]= c3.slider("Setpoint shift (°C)",   0.0,8.0,float(S("EMS_NIGHT_SETPOINT_SHIFT_C",2.0)),0.25)
        st.session_state["EMS_NIGHT_AIRFLOW_FACTOR"]  = c4.slider("Airflow factor",         0.1,1.0,float(S("EMS_NIGHT_AIRFLOW_FACTOR",0.55)),0.05)

    with st.expander("⚡ Demand response"):
        c1,c2,c3,c4 = st.columns(4)
        st.session_state["EMS_DR_START_HOUR"]         = c1.number_input("DR start (hr)",    0.,24.,float(S("EMS_DR_START_HOUR",13.)),  0.5)
        st.session_state["EMS_DR_END_HOUR"]           = c2.number_input("DR end (hr)",      0.,24.,float(S("EMS_DR_END_HOUR",17.)),    0.5)
        st.session_state["EMS_DR_SETPOINT_SHIFT_C"]   = c3.slider("Setpoint shift (°C)",   0.0,6.0,float(S("EMS_DR_SETPOINT_SHIFT_C",1.5)),0.25)
        st.session_state["EMS_DR_AIRFLOW_REDUCTION"]  = c4.slider("Airflow reduction",      0.0,0.8,float(S("EMS_DR_AIRFLOW_REDUCTION",0.15)),0.05)

    with st.expander("🌿 Economizer"):
        c1,c2,c3 = st.columns(3)
        st.session_state["EMS_ECONOMIZER_TEMP_LOW_C"]      = c1.number_input("Outdoor temp low (°C)", -20.,30.,float(S("EMS_ECONOMIZER_TEMP_LOW_C",16.)),0.5)
        st.session_state["EMS_ECONOMIZER_TEMP_HIGH_C"]     = c2.number_input("Outdoor temp high (°C)",-10.,40.,float(S("EMS_ECONOMIZER_TEMP_HIGH_C",22.)),0.5)
        st.session_state["EMS_ECONOMIZER_COOLING_REDUCTION"]= c3.slider("Cooling reduction", 0.0,0.8,float(S("EMS_ECONOMIZER_COOLING_REDUCTION",0.20)),0.05)

    with st.expander("⏰ Optimum start"):
        c1,c2 = st.columns(2)
        st.session_state["EMS_OPTIMUM_START_HOUR"] = c1.number_input("Occupancy start (hr)", 0.,24.,float(S("EMS_OPTIMUM_START_HOUR",7.)),0.5)
        st.session_state["EMS_PRECOOL_SHIFT_C"]    = c2.slider("Pre-cool setpoint shift (°C)",-5.0,0.0,float(S("EMS_PRECOOL_SHIFT_C",-0.8)),0.1)

    st.markdown("---")
    st.markdown("### 📅 Custom Operation Schedule *(optional)*")
    hint("Upload a time-based schedule that overrides setpoint and airflow. Enable the toggle below to use it in simulation.")
    st.session_state["EMS_CUSTOM_SCHEDULE_ENABLED"] = st.toggle("Use custom schedule in simulation",
        value=bool(S("EMS_CUSTOM_SCHEDULE_ENABLED",False)))

    if "operation_schedule_df" not in st.session_state:
        st.session_state["operation_schedule_df"] = build_operation_schedule_template()

    c1,c2 = st.columns(2)
    if c1.button("📋 Load default educational schedule"):
        st.session_state["operation_schedule_df"] = build_operation_schedule_template()
    su = c2.file_uploader("Upload custom schedule CSV", type=["csv"], key="sched_up")
    if su:
        st.session_state["operation_schedule_df"] = validate_operation_schedule(pd.read_csv(su))

    sched_df = st.data_editor(st.session_state["operation_schedule_df"],
                              num_rows="dynamic", use_container_width=True, key="sched_ed")
    st.session_state["operation_schedule_df"] = validate_operation_schedule(sched_df)
    st.download_button("⬇️ Download schedule CSV",
                       st.session_state["operation_schedule_df"].to_csv(index=False).encode(),
                       "operation_schedule.csv", key="sch_dl")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5  ·  RUN & RESULTS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("### ▶️ Run Simulation")

    c1,c2,c3 = st.columns(3)
    axis_mode = c1.selectbox("What to compare", ["one_strategy","one_severity","two_axis","three_axis"],
        format_func=lambda x: {
            "one_strategy": "All 4 strategies (S0→S3) — recommended first run",
            "one_severity": "All 4 severity levels, fixed strategy",
            "two_axis":     "4 strategies × 4 severities matrix",
            "three_axis":   "Full 4×4×4 matrix — 64 runs (slow)",
        }[x])

    sev_opts = list(SEVERITY_LEVELS.keys())
    cli_opts = list(CLIMATE_LEVELS.keys())
    fixed_severity = c2.selectbox("Degradation severity", sev_opts, index=1,
        format_func=lambda x: f"{x}  —  {SEVERITY_DESC[x]}")
    fixed_climate  = c3.selectbox("Climate scenario", cli_opts,
        format_func=lambda x: f"{x.split('_')[0]}  —  {CLIMATE_DESC[x]}")

    fixed_strategy = "S3"
    if axis_mode == "one_severity":
        fixed_strategy = st.selectbox("Strategy (held fixed)", list(SCENARIOS.keys()),
            index=3, format_func=lambda x: SCENARIO_DESC[x])

    with st.expander("🌤️ Weather source  (default = synthetic built-in)"):
        wmode_ui = st.selectbox("Weather source",
            ["Synthetic (built-in)","Upload CSV or EPW","EPW file path","CSV file path"],
            key="run_wmode")
        weather_df_run = epw_path_run = csv_path_run = None
        if wmode_ui == "Upload CSV or EPW":
            wf = st.file_uploader("Upload weather file", type=["csv","epw","txt"], key="run_wf")
            if wf:
                try:
                    weather_df_run = read_weather_upload(wf)
                    st.session_state["_wdf"] = weather_df_run
                    st.success(f"Loaded {len(weather_df_run):,} weather records")
                except Exception as e: st.error(str(e))
            else:
                weather_df_run = st.session_state.get("_wdf")
        elif wmode_ui == "EPW file path":
            epw_path_run = st.text_input("EPW path", key="run_epw")
        elif wmode_ui == "CSV file path":
            csv_path_run = st.text_input("CSV path", key="run_csv")
        eng_wmode = {"Synthetic (built-in)":"synthetic","Upload CSV or EPW":"uploaded",
                     "EPW file path":"epw","CSV file path":"csv"}[wmode_ui]

    c1,c2 = st.columns(2)
    out_dir = c1.text_input("Output folder", "hvac_results", key="run_outdir")
    rnd     = c2.number_input("Random seed", 1, 9999, 42, 1)
    inc_bl  = st.checkbox("Also run no-degradation baseline (reference)", value=True)

    ts_hours = TIME_STEP_MAP[S("time_step_label","Daily (fastest)")]
    n_fx     = sum(bool(S(k)) for k in EFFECT_INFO)
    st.info(f"**Ready:** {S('years',20)}-year · {S('time_step_label')} · {n_fx}/6 effects ON · "
            f"EMS: {S('EMS_MODE','Disabled')} · {fixed_severity} severity · {fixed_climate}")

    use_sched = bool(S("EMS_CUSTOM_SCHEDULE_ENABLED",False))
    op_sched  = st.session_state.get("operation_schedule_df") if use_sched else None

    run_btn = st.button("🚀 Run Simulation", type="primary", width="content")

    if run_btn:
        bldg = build_bldg()
        cfg  = build_cfg(ts_hours)
        prog = st.progress(0, text="Initialising…")
        with st.spinner("⏳ Simulating… (a few minutes for 20-year S3 runs)"):
            try:
                prog.progress(10, text="Running scenarios…")
                result = run_scenario_model(
                    output_dir=out_dir, axis_mode=axis_mode,
                    bldg=bldg, cfg=cfg,
                    weather_mode=eng_wmode, epw_path=epw_path_run,
                    csv_path=csv_path_run, weather_df=weather_df_run,
                    fixed_strategy=fixed_strategy, fixed_severity=fixed_severity,
                    fixed_climate=fixed_climate,
                    degradation_model=S("degradation_model","physics"),
                    include_baseline_layer=inc_bl,
                    include_baseline_as_scenario=inc_bl,
                    random_state=int(rnd),
                    operation_schedule_df=op_sched,
                )
                prog.progress(90, text="Generating reports…")
                st.session_state["last_result"]     = result
                st.session_state["last_result_dir"] = out_dir
                prog.progress(100, text="Done!")
                st.success("✅ Simulation complete!")
            except Exception as e:
                prog.empty(); st.error(f"❌ {e}"); st.exception(e)

    # ── Results panel ─────────────────────────────────────────────────────────
    result = st.session_state.get("last_result")
    if result:
        st.markdown("---")
        st.markdown("### 📊 Results")
        spath = result.get("summary_csv","")
        KPI_DEFS = [
            ("Total Energy MWh",        "⚡ Energy",     "MWh",        "#5dddff"),
            ("Mean Comfort Deviation C", "🌡️ Comfort",   "°C deviation","#f9a825"),
            ("Total CO2 tonne",          "🌿 Carbon",    "tonne CO₂",  "#69f0ae"),
            ("Mean Degradation Index",   "🔧 Degradation","index 0–1",  "#ff686b"),
        ]
        if Path(spath).exists():
            sumdf = pd.read_csv(spath)
            strats = sumdf["strategy"].tolist() if "strategy" in sumdf.columns else []

            if strats:
                # KPI cards: one column per strategy
                st.markdown("#### 🎯 KPIs by strategy")
                cols = st.columns(len(strats))
                for col, stg in zip(cols, strats):
                    row = sumdf[sumdf["strategy"]==stg].iloc[0]
                    with col:
                        st.markdown(f"**{SCENARIO_DESC.get(stg,stg)}**")
                        for kcol, label, unit, color in KPI_DEFS:
                            val = row.get(kcol, np.nan)
                            kpi_card(label, f"{val:.1f}" if np.isfinite(val) else "—", unit, color)
            else:
                row = sumdf.iloc[0]
                kc = st.columns(4)
                for col,(kcol,label,unit,color) in zip(kc, KPI_DEFS):
                    with col:
                        val = row.get(kcol, np.nan)
                        kpi_card(label, f"{val:.2f}" if np.isfinite(val) else "—", unit, color)

            # Bar charts per KPI
            if PLOTLY_OK and "strategy" in sumdf.columns and len(sumdf) > 1:
                st.markdown("#### 📊 Strategy comparison")
                ft = st.tabs(["⚡ Energy","🌡️ Comfort","🌿 Carbon","🔧 Degradation"])
                for ftab,(kcol,label,unit,color) in zip(ft, KPI_DEFS):
                    with ftab:
                        if kcol in sumdf.columns:
                            fig = px.bar(sumdf, x="strategy", y=kcol,
                                         title=f"{label} by Strategy",
                                         labels={kcol:f"{label} ({unit})"},
                                         template="plotly_dark",
                                         color_discrete_sequence=[color]*10)
                            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                              plot_bgcolor="rgba(0,0,0,0)",
                                              font_color="#e0f0fb", showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📋 Full summary table"):
                st.dataframe(sumdf, use_container_width=True)

        # Downloads
        st.markdown("#### ⬇️ Download results")
        c1,c2,c3,c4 = st.columns(4)
        with c1: dl_btn(result.get("dataset_csv",""),  "📄 Daily dataset CSV", "dl_d")
        with c2: dl_btn(result.get("summary_csv",""),  "📊 Summary CSV",       "dl_s")
        with c3: dl_btn(result.get("excel_report",""), "📗 Excel report",      "dl_x")
        with c4: dl_btn(result.get("pdf_report",""),   "📕 PDF report",        "dl_p")

        if st.button("📦 Create & download ZIP of full output folder", key="run_zip"):
            try:
                zp = create_zip_from_folder(out_dir)
                dl_btn(zp, "⬇️ Download ZIP", "dl_zip2")
            except Exception as e: st.error(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6  ·  CHARTS & TRENDS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[6]:
    st.markdown("### 📊 Charts, Trends & Annual Analysis")
    result_dir_c = st.text_input("Result folder to analyse",
                                  S("last_result_dir","hvac_results"), key="charts_dir")
    all_csvs = ({p.name: p for p in Path(result_dir_c).glob("*.csv") if p.is_file()}
                if Path(result_dir_c).is_dir() else {})

    if not all_csvs:
        st.info("Run a simulation first (▶️ Run & Results) or enter an existing result folder above.")
    else:
        csv_sel = st.selectbox("Select dataset to plot", list(all_csvs.keys()))
        data    = pd.read_csv(all_csvs[csv_sel])
        st.caption(f"Loaded {len(data):,} rows × {len(data.columns)} columns")

        num_cols = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
        cat_cols = [c for c in data.columns if not pd.api.types.is_numeric_dtype(data[c])]

        if PLOTLY_OK and num_cols:
            c1,c2,c3,c4 = st.columns(4)
            ctype = c1.selectbox("Chart type",["Line","Bar","Scatter","Box","Area","Heatmap"],key="ct")
            default_x = "year" if "year" in data.columns else data.columns[0]
            x_col  = c2.selectbox("X axis", data.columns.tolist(),
                                   index=data.columns.tolist().index(default_x), key="ctx")
            y_cols = c3.multiselect("Y axis", num_cols,
                default=[c for c in ["annual_energy_MWh","mean_delta","mean_comfort_dev","annual_co2_tonne"] if c in num_cols][:2],
                key="cty")
            grp    = c4.selectbox("Colour by",["None"]+cat_cols, key="ctg")
            gcol   = None if grp=="None" else grp
            title  = st.text_input("Chart title", f"{ctype} — {', '.join(y_cols[:2])}", key="ctt")

            if y_cols:
                try:
                    if ctype=="Line":    fig=px.line(data, x=x_col, y=y_cols, color=gcol, title=title, template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Set2)
                    elif ctype=="Bar":   fig=px.bar(data, x=x_col, y=y_cols[0], color=gcol, title=title, template="plotly_dark")
                    elif ctype=="Scatter":fig=px.scatter(data, x=x_col, y=y_cols[0], color=gcol, size=y_cols[1] if len(y_cols)>1 else None, title=title, template="plotly_dark")
                    elif ctype=="Box":   fig=px.box(data, x=gcol or x_col, y=y_cols[0], color=gcol, title=title, template="plotly_dark")
                    elif ctype=="Area":  fig=px.area(data, x=x_col, y=y_cols, color=gcol, title=title, template="plotly_dark")
                    elif ctype=="Heatmap":
                        idx = gcol or (cat_cols[0] if cat_cols else x_col)
                        pivot=data.pivot_table(index=idx,columns=x_col,values=y_cols[0])
                        fig=px.imshow(pivot, title=title, template="plotly_dark", color_continuous_scale="Blues")
                    else: fig=px.line(data, x=x_col, y=y_cols, title=title, template="plotly_dark")

                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                      font_color="#e0f0fb", legend_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)
                    st.download_button("⬇️ Download chart HTML", fig.to_html(include_plotlyjs="cdn"),
                                       "chart.html","text/html", key="chart_dl")
                except Exception as e:
                    st.error(str(e))

        with st.expander("📋 Raw data preview (first 200 rows)"):
            st.dataframe(data.head(200), use_container_width=True)

        # Static figures
        figs_d = Path(result_dir_c) / "figures"
        if figs_d.is_dir():
            pngs = sorted(figs_d.glob("*.png"))
            if pngs:
                st.markdown("#### 🖼️ Generated figures")
                gcols = st.columns(min(3, len(pngs)))
                for col, p in zip(gcols*(len(pngs)//max(len(gcols),1)+1), pngs):
                    with col:
                        st.image(str(p), caption=p.stem)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 7  ·  ADVANCED ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[7]:
    st.markdown("### 🔬 Advanced Analysis")
    hint("All tools here use the same engine — they run extra analyses on top of the main simulation.")

    bldg_a = build_bldg()
    cfg_a  = build_cfg(TIME_STEP_MAP[S("time_step_label","Daily (fastest)")])

    adv = st.tabs(["🎯 Multi-objective","📈 Sensitivity","🎲 Robustness",
                   "🤖 Surrogate","🔬 HX Diagnostics","📉 Part-load COP",
                   "💧 Latent Load","🗂️ Zone Loads","📊 Global Sensitivity","🎮 Control Library"])

    # ── Multi-objective ──────────────────────────────────────────────────────
    with adv[0]:
        st.markdown("#### Multi-objective search for optimal EMS / control candidates")
        c1,c2,c3 = st.columns(3)
        moo_opt = c1.selectbox("Optimiser",["Weighted random search","Grid search","NSGA-II style screening","Custom optimizer label"],key="moo_opt")
        moo_n   = int(c2.number_input("Candidates",2,60,10,1,key="moo_n"))
        moo_yr  = int(c3.number_input("Analysis years",1,10,1,1,key="moo_yr"))
        c1,c2,c3 = st.columns(3)
        moo_stg = c1.selectbox("Strategy",list(SCENARIOS.keys()),index=3,key="moo_stg")
        moo_sev = c2.selectbox("Severity",list(SEVERITY_LEVELS.keys()),index=1,key="moo_sev")
        moo_cli = c3.selectbox("Climate",list(CLIMATE_LEVELS.keys()),key="moo_cli")
        c1,c2,c3,c4 = st.columns(4)
        mw_e=c1.number_input("Energy w",  0.,1.,.35,.05,key="mw_e")
        mw_d=c2.number_input("Degrad w",  0.,1.,.25,.05,key="mw_d")
        mw_c=c3.number_input("Comfort w", 0.,1.,.25,.05,key="mw_c")
        mw_co=c4.number_input("Carbon w", 0.,1.,.15,.05,key="mw_co")
        moo_out=st.text_input("Output folder","hvac_results/moo",key="moo_out")
        if st.button("🚀 Run MOO",type="primary",key="moo_run"):
            with st.spinner("Running multi-objective optimisation…"):
                try:
                    res=run_multi_objective_search(output_dir=moo_out,bldg=bldg_a,cfg=cfg_a,
                        fixed_strategy=moo_stg,fixed_severity=moo_sev,fixed_climate=moo_cli,
                        optimizer_name=moo_opt,n_candidates=moo_n,
                        weights={"energy":mw_e,"degradation":mw_d,"comfort":mw_c,"carbon":mw_co},
                        analysis_years=moo_yr,random_state=42)
                    st.success("MOO complete")
                    if "results_csv" in res and Path(res["results_csv"]).exists():
                        df=pd.read_csv(res["results_csv"]); st.dataframe(df,use_container_width=True)
                    dl_btn(res.get("results_csv",""),"⬇️ MOO results CSV","moo_dl")
                except Exception as e: st.exception(e)

    # ── Sensitivity ───────────────────────────────────────────────────────────
    with adv[1]:
        st.markdown("#### One-at-a-time sensitivity ranking")
        hint("Perturbs each input ±% and ranks parameters by their effect on all 4 KPIs.")
        c1,c2,c3 = st.columns(3)
        s_yr  = int(c1.number_input("Analysis years",1,10,1,key="s_yr"))
        s_pct = float(c2.number_input("Perturbation ±%",1.,50.,10.,key="s_pct"))/100.
        s_stg = c3.selectbox("Strategy",list(SCENARIOS.keys()),index=2,key="s_stg")
        c1,c2 = st.columns(2)
        s_sev = c1.selectbox("Severity",list(SEVERITY_LEVELS.keys()),index=1,key="s_sev")
        s_cli = c2.selectbox("Climate",list(CLIMATE_LEVELS.keys()),key="s_cli")
        s_out = st.text_input("Output folder","hvac_results/sensitivity",key="s_out")
        if st.button("🚀 Run sensitivity",type="primary",key="s_run"):
            with st.spinner("Running sensitivity analysis…"):
                try:
                    res=run_early_sensitivity_analysis(output_dir=s_out,bldg=bldg_a,cfg=cfg_a,
                        fixed_strategy=s_stg,fixed_severity=s_sev,fixed_climate=s_cli,
                        perturbation_pct=s_pct,analysis_years=s_yr,random_state=42)
                    st.success("Done")
                    df=pd.read_csv(res["ranking_csv"]); st.dataframe(df,use_container_width=True)
                    if PLOTLY_OK and not df.empty:
                        fig=px.bar(df.head(15),x="composite_importance",y="label",
                                   orientation="h",title="Parameter importance",template="plotly_dark")
                        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#e0f0fb")
                        st.plotly_chart(fig,use_container_width=True)
                    dl_btn(res["ranking_csv"],"⬇️ Ranking CSV","s_dl")
                except Exception as e: st.exception(e)

    # ── Robustness ────────────────────────────────────────────────────────────
    with adv[2]:
        st.markdown("#### Monte Carlo robustness analysis")
        hint("Randomly perturbs all inputs N times and reports KPI spread (P5–P95).")
        c1,c2,c3,c4 = st.columns(4)
        r_yr  = int(c1.number_input("Analysis years",1,10,1,key="r_yr"))
        r_pct = float(c2.number_input("Uncertainty ±%",1.,50.,10.,key="r_pct"))/100.
        r_n   = int(c3.number_input("Samples",3,200,15,key="r_n"))
        r_stg = c4.selectbox("Strategy",list(SCENARIOS.keys()),index=2,key="r_stg")
        c1,c2 = st.columns(2)
        r_sev = c1.selectbox("Severity",list(SEVERITY_LEVELS.keys()),index=1,key="r_sev")
        r_cli = c2.selectbox("Climate",list(CLIMATE_LEVELS.keys()),key="r_cli")
        r_out = st.text_input("Output folder","hvac_results/robustness",key="r_out")
        if st.button("🚀 Run robustness",type="primary",key="r_run"):
            with st.spinner(f"Running {r_n} Monte Carlo samples…"):
                try:
                    res=run_robustness_analysis(output_dir=r_out,bldg=bldg_a,cfg=cfg_a,
                        fixed_strategy=r_stg,fixed_severity=r_sev,fixed_climate=r_cli,
                        n_samples=r_n,uncertainty_pct=r_pct,analysis_years=r_yr,random_state=42)
                    st.success("Done")
                    df=pd.read_csv(res["summary_csv"]); st.dataframe(df,use_container_width=True)
                    if PLOTLY_OK and not df.empty and "kpi" in df.columns:
                        fig=px.bar(df,x="kpi",y=[c for c in ["p05","p50","p95"] if c in df.columns],
                                   title="KPI spread P5/P50/P95",barmode="group",template="plotly_dark")
                        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#e0f0fb")
                        st.plotly_chart(fig,use_container_width=True)
                    dl_btn(res["summary_csv"],"⬇️ Robustness CSV","r_dl")
                except Exception as e: st.exception(e)

    # ── Surrogate ─────────────────────────────────────────────────────────────
    with adv[3]:
        st.markdown("#### CatBoost surrogate model training")
        hint("Trains a fast ML surrogate on the simulation dataset. Requires CatBoost installed.")
        s_csv = st.text_input("Input CSV (from scenario run)","hvac_results/matrix_ml_dataset.csv",key="su_csv")
        su_out= st.text_input("Output folder","hvac_results/surrogate",key="su_out")
        c1,c2 = st.columns(2)
        su_it = int(c1.number_input("HP search iterations",2,30,6,key="su_it"))
        su_sh = int(c2.number_input("SHAP samples",100,5000,500,key="su_sh"))
        if st.button("🚀 Train surrogate",type="primary",key="su_run"):
            if not Path(s_csv).exists():
                st.error("Input CSV not found — run a scenario simulation first")
            else:
                with st.spinner("Training CatBoost + SHAP…"):
                    try:
                        res=train_surrogate_models(s_csv,su_out,su_it,su_sh)
                        st.success("Done")
                        if Path(res["metrics_csv"]).exists():
                            st.dataframe(pd.read_csv(res["metrics_csv"]),use_container_width=True)
                        dl_btn(res.get("excel_report",""),"⬇️ Excel","su_xl")
                        dl_btn(res.get("pdf_report",""),"⬇️ PDF","su_pdf")
                    except Exception as e: st.exception(e)

    # ── HX Diagnostics ────────────────────────────────────────────────────────
    with adv[4]:
        st.markdown("#### Heat Exchanger Diagnostics")
        hx_dir=st.text_input("Result folder",S("last_result_dir","hvac_results"),key="hx_dir")
        if st.button("🔍 Build HX diagnostic report",key="hx_run"):
            try:
                _hx_pt = find_result_paths(hx_dir)
            except Exception:
                _hx_pt = {}
            _hx_daily = Path(_hx_pt.get("daily","")) if _hx_pt.get("daily") else next(Path(hx_dir).glob("*dataset*.csv") if Path(hx_dir).is_dir() else iter([]), None)
            if _hx_daily and Path(_hx_daily).exists():
                tables=build_heat_exchanger_diagnostics(hx_dir,cfg=cfg_a)
                for k,v in tables.items():
                    if isinstance(v,pd.DataFrame) and not v.empty:
                        st.markdown(f"**{k}**"); st.dataframe(v,use_container_width=True)
            else: st.warning("No daily dataset found — run a simulation first")

    # ── Part-load COP ─────────────────────────────────────────────────────────
    with adv[5]:
        st.markdown("#### Part-load COP curve analysis")
        if st.button("📉 Show PLR curve",key="plr_btn"):
            tables=build_part_load_curve_analysis(cfg=cfg_a)
            if PLOTLY_OK and "plr_curve" in tables and not tables["plr_curve"].empty:
                df_p=tables["plr_curve"]
                fig=px.line(df_p,x="PLR",y="modifier",title="COP modifier vs PLR",template="plotly_dark")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#e0f0fb")
                st.plotly_chart(fig,use_container_width=True)
            for k,v in (tables or {}).items():
                if isinstance(v,pd.DataFrame) and not v.empty and k!="plr_curve":
                    st.markdown(f"**{k}**"); st.dataframe(v,use_container_width=True)

    # ── Latent Load ───────────────────────────────────────────────────────────
    with adv[6]:
        st.markdown("#### Latent moisture cooling load analysis")
        if st.button("💧 Analyse latent load",key="lat_btn"):
            tables=build_latent_load_analysis(bldg=bldg_a,cfg=cfg_a)
            for k,v in (tables or {}).items():
                if isinstance(v,pd.DataFrame) and not v.empty:
                    st.markdown(f"**{k}**"); st.dataframe(v,use_container_width=True)

    # ── Zone Loads ────────────────────────────────────────────────────────────
    with adv[7]:
        st.markdown("#### Zone-by-zone load analysis")
        zone_df_a=st.session_state.get("last_zone_df")
        if zone_df_a is None:
            st.info("No zone table loaded. Upload one in 📥 Export & Tools → Zone table.")
        else:
            tables=build_native_zone_load_table(bldg=bldg_a,cfg=cfg_a,zone_df=zone_df_a)
            for k,v in (tables or {}).items():
                if isinstance(v,pd.DataFrame) and not v.empty:
                    st.markdown(f"**{k}**"); st.dataframe(v,use_container_width=True)

    # ── Global Sensitivity ────────────────────────────────────────────────────
    with adv[8]:
        st.markdown("#### Global sensitivity from robustness samples (Spearman correlation)")
        gs_p=st.text_input("Robustness samples CSV","hvac_results/robustness/robustness_samples.csv",key="gs_p")
        if st.button("📊 Compute global sensitivity",key="gs_btn"):
            if Path(gs_p).exists():
                tables=build_global_sensitivity_from_samples(pd.read_csv(gs_p))
                for k,v in (tables or {}).items():
                    if isinstance(v,pd.DataFrame) and not v.empty:
                        st.markdown(f"**{k}**"); st.dataframe(v,use_container_width=True)
            else: st.error("Run robustness analysis first and check the file path")

    # ── Control Library ───────────────────────────────────────────────────────
    with adv[9]:
        st.markdown("#### Advanced HVAC control library & scoring")
        candidates=build_advanced_control_candidates()
        c1,c2,c3,c4,c5=st.columns(5)
        cw_e =c1.number_input("Energy w",  0.,1.,.35,.05,key="cw_e")
        cw_c =c2.number_input("Comfort w", 0.,1.,.25,.05,key="cw_c")
        cw_d =c3.number_input("Degrad w",  0.,1.,.20,.05,key="cw_d")
        cw_co=c4.number_input("Carbon w",  0.,1.,.10,.05,key="cw_co")
        cw_f =c5.number_input("Fault risk w",0.,1.,.10,.05,key="cw_f")
        scored=build_control_objective_table(candidates,
                    weights={"energy":cw_e,"comfort":cw_c,"degradation":cw_d,"carbon":cw_co,"fault_risk":cw_f})
        st.dataframe(scored,use_container_width=True)
        ids=scored["control_id"].tolist()
        sel=st.selectbox("Apply to EMS",ids,
            format_func=lambda x: scored.loc[scored["control_id"]==x,"control_name"].iloc[0])
        if st.button("⚡ Apply to EMS settings",key="ctl_apply"):
            row=scored.loc[scored["control_id"]==sel].iloc[0]
            st.session_state["EMS_MODE"]=row["ems_mode"]
            st.session_state["EMS_OCC_CONTROL"]=bool(row["use_occ_reset"])
            st.session_state["EMS_NIGHT_SETBACK"]=bool(row["use_night_setback"])
            st.session_state["EMS_DEMAND_RESPONSE"]=bool(row["use_demand_response"])
            st.session_state["EMS_ECONOMIZER"]=bool(row["use_economizer"])
            st.session_state["EMS_OPTIMUM_START"]=bool(row["use_optimum_start"])
            st.success(f"Applied: {row['control_name']} — verify settings in 🎮 EMS Control tab")
        st.download_button("⬇️ Download control library CSV",
                           scored.to_csv(index=False).encode(),"control_library.csv",key="ctl_dl")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 8  ·  EXPORT & TOOLS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[8]:
    st.markdown("### 📥 Export, Validation & Extra Tools")

    rd = st.text_input("Result folder", S("last_result_dir","hvac_results"), key="et_dir")

    # ── Safe helper: find_result_paths may return Path objects or plain strings,
    #    and not every key is guaranteed.  Normalise everything to Path here.
    try:
        _pt_raw = find_result_paths(rd)
    except Exception:
        _pt_raw = {}

    def _p(key: str) -> Path:
        """Return a Path for the given key, falling back to non-existent sentinel."""
        val = _pt_raw.get(key, "")
        return Path(val) if val else Path(rd) / "__not_found__"

    # Also search the folder directly for common filenames when keys are absent
    _rd = Path(rd)
    def _find_first(*globs):
        for g in globs:
            hits = sorted(_rd.glob(g)) if _rd.is_dir() else []
            if hits: return hits[0]
        return _rd / "__not_found__"

    pt_summary  = _p("summary")  if _p("summary").exists()  else _find_first("*summary*.csv","*Summary*.csv")
    pt_annual   = _p("annual")   if _p("annual").exists()   else _find_first("*annual*.csv","*Annual*.csv")
    pt_excel    = _p("excel")    if _p("excel").exists()    else _find_first("*.xlsx","*.xls")
    pt_pdf      = _p("pdf")      if _p("pdf").exists()      else _find_first("*.pdf")
    pt_daily    = _p("daily")    if _p("daily").exists()    else _find_first("*dataset*.csv","*daily*.csv","*output*.csv")
    pt_baseline = _p("baseline_summary") if _p("baseline_summary").exists() else _find_first("*baseline*summary*.csv")

    st.markdown("#### ⬇️ Direct downloads")
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        dl_btn(pt_summary,  "📊 Summary CSV",     "et_sum")
        dl_btn(pt_annual,   "📅 Annual CSV",       "et_ann")
    with c2:
        dl_btn(pt_excel,    "📗 Excel report",     "et_xl")
        dl_btn(pt_pdf,      "📕 PDF report",       "et_pdf")
    with c3:
        dl_btn(pt_daily,    "📄 Daily dataset",    "et_day")
        dl_btn(pt_baseline, "📊 Baseline summary", "et_bls")
    with c4:
        if st.button("📦 Create ZIP of folder", key="et_zip"):
            try:
                zp=create_zip_from_folder(rd); dl_btn(zp,"⬇️ Download ZIP","et_zip2")
            except Exception as e: st.error(str(e))

    # Validation
    st.markdown("---")
    st.markdown("#### ✅ Model Validation")
    hint("Upload reference data (DesignBuilder, EnergyPlus, measured) to compare against your simulation.")
    vf = st.file_uploader("Upload validation CSV", type=["csv"], key="et_vf")
    if vf and pt_summary.exists():
        try:
            vdf=load_validation_file(vf)
            sdf=pd.read_csv(pt_summary)
            cmp=build_validation_comparison(sdf,vdf,source_name=Path(vf.name).stem)
            st.dataframe(cmp,use_container_width=True)
            met=build_formal_validation_metrics(sdf,vdf)
            if isinstance(met,pd.DataFrame) and not met.empty:
                st.markdown("**Formal validation metrics (RMSE · MAE · R²)**")
                st.dataframe(met,use_container_width=True)
            out_val=Path(rd)/"validation_comparison.csv"
            cmp.to_csv(out_val,index=False)
            dl_btn(out_val,"⬇️ Validation CSV","et_val")
        except Exception as e: st.exception(e)

    # Zone table
    st.markdown("---")
    st.markdown("#### 🗂️ Zone occupancy table *(optional, for zone-by-zone mode)*")
    hint("Columns needed: zone_name · zone_type · area_m2 · occ_density · term_factor · break_factor · summer_factor")
    zu = st.file_uploader("Upload zone CSV", type=["csv"], key="et_zone")
    if zu:
        zdf=pd.read_csv(zu)
        st.session_state["last_zone_df"]=zdf
        tot_a=zdf["area_m2"].sum() if "area_m2" in zdf.columns else 0
        st.success(f"Zone table loaded: {len(zdf)} zones · total area {tot_a:.0f} m²")
        st.dataframe(zdf,use_container_width=True)

    # Detailed tables
    st.markdown("---")
    st.markdown("#### 📋 Detailed benchmark tables")
    if st.button("🔍 Build detailed tables",key="et_bench"):
        if pt_summary.exists():
            tables=build_detailed_tables(rd,bldg=build_bldg(),cfg=build_cfg(),
                                         zone_df=st.session_state.get("last_zone_df"))
            save_detailed_outputs(rd,tables)
            for k,v in tables.items():
                if isinstance(v,pd.DataFrame) and not v.empty:
                    st.markdown(f"**{k}**"); st.dataframe(v,use_container_width=True)
        else:
            st.warning("No simulation results found — run a simulation first")

    # MPC / RL
    st.markdown("---")
    with st.expander("🤖 Experimental: MPC template & RL dataset spec"):
        c1,c2=st.columns(2)
        mpc_h=int(c1.number_input("MPC horizon (hrs)",2,168,24,key="mpc_h"))
        mpc_s=int(c2.number_input("MPC step (hrs)",1,24,1,key="mpc_s"))
        if st.button("Generate MPC template",key="mpc_btn"):
            mdf=build_mpc_experimental_template(mpc_h,mpc_s)
            st.dataframe(mdf.head(48),use_container_width=True)
            st.download_button("⬇️ MPC template CSV",mdf.to_csv(index=False).encode(),"mpc_template.csv",key="mpc_dl")
        if st.button("Generate RL dataset spec",key="rl_btn"):
            rdf=build_rl_experimental_dataset_spec()
            st.dataframe(rdf,use_container_width=True)
            st.download_button("⬇️ RL spec CSV",rdf.to_csv(index=False).encode(),"rl_dataset_spec.csv",key="rl_dl")