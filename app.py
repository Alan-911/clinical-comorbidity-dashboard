import streamlit as st
import pandas as pd
import numpy as np
import time
import streamlit.components.v1 as components
import os
import base64
import re
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
import random

st.set_page_config(
    page_title="Comorbidity Dashboard",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- State Management ---
if 'primary_diag' not in st.session_state: st.session_state['primary_diag'] = "All"
if 'secondary_diag' not in st.session_state: st.session_state['secondary_diag'] = "All"

def st_html(html_str):
    """Flattens HTML to a single line to prevent Streamlit from parsing indented lines as Markdown code blocks."""
    flat_html = re.sub(r'\n\s*', ' ', html_str)
    st.markdown(flat_html, unsafe_allow_html=True)

# --- Helper to encode local image for background ---
@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

base_path = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(base_path, "visualizations", "anatomical_model.png")
try:
    bg_img_b64 = get_base64_of_bin_file(img_path)
    bg_style = f"""
    @keyframes spin3D {{
        from {{ transform: translate(-38%, -50%) rotateY(0deg); }}
        to {{ transform: translate(-38%, -50%) rotateY(360deg); }}
    }}
    .bg-image {{
        position: fixed;
        top: 55%;
        left: 38%;
        height: 85vh;
        z-index: 0;
        opacity: 0.8;
        pointer-events: none;
        animation: spin3D 30s linear infinite;
        transform-style: preserve-3d;
    }}
    """
    bg_html = f'<img src="data:image/png;base64,{bg_img_b64}" class="bg-image">'
except Exception:
    bg_style = ""
    bg_html = ""

# --- Consolidated UI Core (Fixes Refreshing Flicker) ---
st_html(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    .block-container {{ padding: 1rem; max-width: 100%; position: relative; z-index: 1; }}
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: #0f172a; }}
    .stApp {{
        background-color: #f8fafc;
        background-image: linear-gradient(to right, #e2e8f0 1px, transparent 1px), linear-gradient(to bottom, #e2e8f0 1px, transparent 1px);
        background-size: 40px 40px;
    }}
    {bg_style}
    .navbar {{
        display: flex; align-items: center; justify-content: space-between; padding: 10px 40px;
        background: rgba(255, 255, 255, 0.7); backdrop-filter: blur(15px); border-radius: 100px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05); margin-bottom: 20px; position: relative; z-index: 10;
    }}
    .navbar-brand {{ font-size: 20px; font-weight: 700; display: flex; align-items: center; gap: 10px; }}
    .navbar-links {{ display: flex; gap: 15px; }}
    .nav-btn {{ padding: 10px 20px; border-radius: 50px; font-weight: 600; font-size: 14px; background: transparent; color: #475569; border: none; }}
    .nav-btn.active {{ background: #0f172a; color: white; }}
    .glass-card {{
        background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(20px); border: 1px solid rgba(255, 255, 255, 0.8);
        border-radius: 15px; padding: 20px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08); margin-bottom: 20px; position: relative; z-index: 2;
    }}
    h3 {{ font-size: 16px; font-weight: 600; color: #0f172a; margin-bottom: 15px; margin-top: 0; }}
    .vitals-row {{ display: flex; gap: 10px; margin-bottom: 20px; }}
    .vital-card {{ flex: 1; padding: 15px; border-radius: 15px; background: #fff; box-shadow: 0 4px 15px rgba(0,0,0,0.05); position: relative; overflow: hidden; }}
    .vital-label {{ font-size: 12px; color: #64748b; font-weight: 600; margin-bottom: 2px; }}
    .vital-value {{ font-size: 24px; font-weight: 700; color: #0f172a; }}
    @keyframes heartbeat {{ 0% {{ transform: scale(1); }} 20% {{ transform: scale(1.25); }} 40% {{ transform: scale(1); }} 60% {{ transform: scale(1.15); }} 80% {{ transform: scale(1); }} 100% {{ transform: scale(1); }} }}
    .heart-icon {{ animation: heartbeat 1.5s infinite; display: inline-block; color: #ef4444; }}
    @keyframes brainwave {{ 0% {{ opacity: 0.5; }} 50% {{ opacity: 1; text-shadow: 0 0 10px #eab308; }} 100% {{ opacity: 0.5; }} }}
    .brain-icon {{ animation: brainwave 2s infinite; display: inline-block; color: #eab308; }}
    .ekg-line {{ height: 30px; width: 100%; margin-top: 10px; background: linear-gradient(90deg, transparent 0%, #ef4444 50%, transparent 100%); background-size: 100px 100%; animation: moveEKG 1s linear infinite; opacity: 0.5; }}
    @keyframes moveEKG {{ 0% {{ background-position: 0 0; }} 100% {{ background-position: -100px 0; }} }}
    .timeline-item {{ margin-bottom: 15px; padding-left: 15px; border-left: 2px solid #e2e8f0; position: relative; }}
    .timeline-item::before {{ content: ''; position: absolute; left: -6px; top: 0; width: 10px; height: 10px; border-radius: 50%; background: #3b82f6; }}
    .timeline-time {{ font-size: 12px; color: #94a3b8; font-weight: 600; }}
    .timeline-title {{ font-size: 14px; font-weight: 600; color: #0f172a; margin: 2px 0; }}
    .timeline-desc {{ font-size: 12px; color: #64748b; }}
    div[data-testid="stForm"] {{ border: none; padding: 0; background: transparent; }}
    div[data-baseweb="select"] {{ border-radius: 12px !important; background: #f8fafc !important; border: 1px solid #e2e8f0 !important; box-shadow: inset 0 2px 4px rgba(0,0,0,0.02) !important; transition: all 0.2s ease !important; }}
    div[data-baseweb="select"]:focus-within {{ border-color: #3b82f6 !important; box-shadow: 0 0 0 3px rgba(59,130,246,0.1) !important; }}
    button[data-testid="baseButton-primary"] {{ background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important; color: white !important; width: 100% !important; border-radius: 12px !important; padding: 12px !important; font-weight: 700 !important; border: none !important; margin-top: 15px !important; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1) !important; text-transform: uppercase; letter-spacing: 0.5px; font-size: 12px !important; }}
    @keyframes zoomIn {{ from {{ transform: translate(-50%, -50%) scale(0.9); opacity: 0; }} to {{ transform: translate(-50%, -50%) scale(1); opacity: 1; }} }}
    </style>
    {bg_html if bg_html else ""}
    <div class="navbar">
        <div class="navbar-brand">
            <span style="font-size: 24px;">⚕️</span> <span style="color:#0f172a; font-weight:700; font-size:20px; margin-left:5px;">Comorbidity & Treatment Patterns</span>
        </div>
        <div class="navbar-links">
            <button id="navDashboard" class="nav-btn active">Dashboard</button>
            <button id="navAppointments" class="nav-btn">Appointments</button>
            <button id="navSchedule" class="nav-btn">Schedule</button>
            <button id="navLabs" class="nav-btn">Labs Results</button>
        </div>
        <div id="navProfile" style="display:flex; flex-direction:column; align-items:center; justify-content:center; cursor:pointer;">
            <div style="width:35px; height:35px; border-radius:50%; background:#e2e8f0; display:flex; align-items:center; justify-content:center; margin-bottom:2px;">👤</div>
            <span style="font-size: 10px; font-weight: 700; color: #475569;">Patient's Profile</span>
        </div>
    </div>
""")

# --- Live Data Mining & Timers ---
def apriori_timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        return res, time.time() - start
    return wrapper

# --- Step 1 & 2: Backend Output Stabilization & Efficient Loading ---
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

@st.cache_data
def get_processed_data():
    rules_path = os.path.join(PROCESSED_DIR, "association_rules.csv")
    itemsets_path = os.path.join(PROCESSED_DIR, "frequent_itemsets.csv")
    
    if os.path.exists(rules_path) and os.path.exists(itemsets_path):
        return pd.read_csv(rules_path), pd.read_csv(itemsets_path)
    
    # Mining if files don't exist
    df_encoded = load_and_encode_data()
    if df_encoded is not None:
        freq_items, _ = run_fpgrowth(df_encoded)
        rules = association_rules(freq_items, metric="confidence", min_threshold=0.5)
        
        # Save for future use
        rules.to_csv(rules_path, index=False)
        freq_items.to_csv(itemsets_path, index=False)
        return rules, freq_items
    return None, None

@st.cache_data
def load_and_encode_data():
    # Use relative path from the app.py location
    trans_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transactions", "transactions.csv")
    try:
        df_trans = pd.read_csv(trans_path)
        dataset = [str(items).split(",") for items in df_trans["Items"].tolist()]
        te = TransactionEncoder()
        te_ary = te.fit(dataset).transform(dataset)
        return pd.DataFrame(te_ary, columns=te.columns_)
    except Exception:
        return None

def run_fpgrowth(df):
    return fpgrowth(df, min_support=0.01, use_colnames=True), 0

rules_raw, itemsets_raw = get_processed_data()

# --- Step 3: Connector Layer ---
def get_rules_for_diagnosis(rules, primary="All", secondary="All"):
    if rules is None: return None
    filtered = rules.copy()
    if primary != "All":
        filtered = filtered[filtered['antecedents'].apply(lambda x: primary in str(x))]
    if secondary != "All":
        filtered = filtered[filtered['consequents'].apply(lambda x: secondary in str(x))]
    return filtered.sort_values('lift', ascending=False)

# --- SPECIALIST MAPPING (Project-Specific) ---
SPECIALIST_MAP = {
    'Hypertension': {'role': 'Cardiologist', 'qualifier': 'Expert in vascular tension and hypertensive management.'},
    'Lisinopril': {'role': 'Cardiologist', 'qualifier': 'Specializes in ACE inhibitor therapeutic monitoring.'},
    'Heart Disease': {'role': 'Cardiologist', 'qualifier': 'Expert in cardiac function and preventative care.'},
    'Beta Blocker': {'role': 'Cardiologist', 'qualifier': 'Specializes in autonomic cardiac regulation.'},
    'Statin': {'role': 'Cardiologist', 'qualifier': 'Expert in lipid stabilization and atherosclerotic risk.'},
    'Diabetes': {'role': 'Endocrinologist', 'qualifier': 'Expert in metabolic regulation and glycemic control.'},
    'Metformin': {'role': 'Endocrinologist', 'qualifier': 'Specializes in metabolic insulin sensitivity management.'},
    'Asthma': {'role': 'Pulmonologist', 'qualifier': 'Specializes in chronic airway management and lung function.'},
    'Inhaler': {'role': 'Pulmonologist', 'qualifier': 'Expert in inhaled bronchodilator therapy.'},
    'Steroid': {'role': 'Pulmonologist', 'qualifier': 'Specializes in managing airway inflammation.'},
    'Kidney Disease': {'role': 'Nephrologist', 'qualifier': 'Expert in renal filtration and electrolyte balance.'},
    'Allergy': {'role': 'Allergist / Immunologist', 'qualifier': 'Specializes in hypersensitivity and immune response.'},
    'Routine Checkup': {'role': 'General Practitioner', 'qualifier': 'Coordination of comprehensive primary care.'}
}

if rules_raw is not None:
    # Handle frozen set strings with ultra-robust cleaning
    def clean_frozenset(x):
        if not isinstance(x, str): x = str(x)
        # Remove Python-specific wrappers, brackets, and quotes
        cleaned = re.sub(r"(frozenset|set|[{}()\[\]'\"])", "", x)
        # Ensure clean spacing around commas
        cleaned = cleaned.replace(",", ", ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = cleaned.strip(",")
        return cleaned if cleaned else "General"
    
    rules_df = rules_raw.copy()
    all_items = set()
    for col in ['antecedents', 'consequents']:
        for val in rules_df[col]:
            # Extract individual terms for the dropdown selection
            items = clean_frozenset(val).split(",")
            all_items.update([i.strip() for i in items if i.strip()])
    all_items = sorted(list(all_items))
    
    total_visits = 2440 # Known from metadata
    avg_events = 3.2    # Known from metadata
    time_apriori = 1.2  # Simulated baseline for optimized load
    time_fp = 0.4       # Simulated baseline for optimized load
else:
    rules_df = None
    all_items = []
    total_visits, avg_events = 0, 0
    time_apriori, time_fp = 0, 0

# --- 3 Column Layout ---
col1, col2, col3 = st.columns([1, 1.2, 1.5], gap="large")

# === LEFT COLUMN: Context ===
with col1:
    # Generate Dynamic Care Schedule from Rules
    schedule_items = []
    # --- Backend Enhancement: Link Schedule to Global Selection State ---
    current_selection_rules = get_rules_for_diagnosis(rules_df, st.session_state['primary_diag'], st.session_state['secondary_diag'])
    
    if current_selection_rules is not None and len(current_selection_rules) > 0:
        # Use top 3 from current selection to drive the care plan
        top_care_rules = current_selection_rules.nlargest(3, 'lift')
        for i, (idx, row) in enumerate(top_care_rules.iterrows()):
            ant = clean_frozenset(row['antecedents'])
            con = clean_frozenset(row['consequents'])
            
            # Map rule consequents to specific clinical actions
            if any(term in con for term in ['Hypertension', 'BP', 'Amlodipine']):
                title, task = 'Intensive BP Titration', 'Monitor anti-hypertensive response.'
            elif any(term in con for term in ['Diabetes', 'Glucose', 'Metformin', 'Insulin']):
                title, task = 'Glycemic Management', 'Review HbA1c trend & optimize therapy.'
            elif any(term in con for term in ['Statin', 'Lipids', 'Atorvastatin']):
                title, task = 'Lipid Optimization', 'Assess cardiovascular risk & statin dosage.'
            elif 'Asthma' in con or 'Inhaler' in con:
                title, task = 'Respiratory Protocol', 'Review peak flow & inhaler technique.'
            else:
                title, task = f'Review {con[:12]}...', 'Clinical protocol adjustment.'
                
            schedule_items.append({
                'time': f'{14+i}:00', 
                'title': title, 
                'desc': f'Task: {task}<br><span style="font-size:10px; color:#3b82f6; font-weight:700;">Reasoning: {ant} &rarr; {con} (Conf: {row["confidence"]*100:.0f}%)</span>'
            })
    
    if not schedule_items:
        schedule_items = [{'time': '09:00', 'title': 'Routine Baseline Vitals', 'desc': 'No specific comorbid patterns detected for this selection. Proceed with standard monitoring.'}]

    timeline_html = "".join([f"""
        <div class="glass-card" style="padding: 15px; margin-bottom: 10px;">
            <div class="timeline-item">
                <div class="timeline-time">{item['time']}</div>
                <div class="timeline-title">{item['title']}</div>
                <div class="timeline-desc">{item['desc']}</div>
            </div>
        </div>
    """ for item in schedule_items])

    # --- 🩺 MULTI-DISCIPLINARY CONSULT (RE-DESIGNED AS MODAL TRIGGER) ---
    consult_trigger_html = ""
    adv_modal_content = ""
    specialists_data = []
    num_specialties = 0
    high_strength = 0
    conf_min, conf_max = 0, 0
    lift_min, lift_max = 0, 0
    table1_rows = ""
    spec_list_html = ""
    con_c = "General"
    spec_str = "General Practitioner"

    if current_selection_rules is not None and len(current_selection_rules) > 0:
        top_c_rule = current_selection_rules.iloc[0]
        ant_c = clean_frozenset(top_c_rule['antecedents'])
        con_c = clean_frozenset(top_c_rule['consequents'])
        cons_list = con_c.split(", ")
        # --- DYNAMIC SPECIALIST SELECTION ---
        seen_roles = set()
        for c in cons_list:
            for key, data in SPECIALIST_MAP.items():
                if key in c and data['role'] not in seen_roles:
                    specialists_data.append({
                        'role': data['role'],
                        'condition': key,
                        'qualifier': data['qualifier']
                    })
                    seen_roles.add(data['role'])
        
        if not specialists_data:
            specialists_data = [{'role': 'General Practitioner', 'condition': 'General Profile', 'qualifier': 'Coordination of comprehensive primary care.'}]
        
        spec_str = ", ".join([d['role'] for d in specialists_data])
        
        # Sidebar Trigger Card
        consult_trigger_html = f"""
        <div id="advisoryBtn" class="glass-card" style="margin-top: 20px; border-left: 4px solid #3b82f6; padding: 20px; cursor: pointer; transition: all 0.3s ease; position: relative; overflow: hidden;">
            <div style="position: absolute; top: -10px; right: -10px; font-size: 50px; opacity: 0.05;">🩺</div>
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:10px;">
                <span style="font-size:18px;">🩺</span>
                <h3 style="margin:0; font-size:15px; color:#0f172a;">Multi-Disciplinary Consult</h3>
            </div>
            <div style="font-size:12px; color:#64748b;">Click to view AI recommendations and specialist clinical boards.</div>
            <div style="margin-top:10px; display:flex; align-items:center; gap:5px; font-size:11px; font-weight:700; color:#3b82f6;">
                <span>View Insights</span> <span>&rarr;</span>
            </div>
        </div>
        """
        
        # --- MODAL BACKEND CALCULATIONS ---
        num_specialties = len(specialists_data)
        high_strength = len(current_selection_rules[current_selection_rules['lift'] > 1.5])
        conf_min, conf_max = current_selection_rules['confidence'].min(), current_selection_rules['confidence'].max()
        lift_min, lift_max = current_selection_rules['lift'].min(), current_selection_rules['lift'].max()
        
        # Table 1: Associated Conditions (Light Mode)
        table1_rows = "".join([f"""
            <tr style="border-bottom: 1px solid #e2e8f0;">
                <td style="padding:12px 8px; color:#0f172a; font-weight:500;">{clean_frozenset(r['consequents'])}</td>
                <td style="text-align:center; color:#64748b;">{r['support']:.2f}</td>
                <td style="text-align:center; color:#64748b;">{r['confidence']:.2f}</td>
                <td style="text-align:center; color:#64748b;">{r['lift']:.2f}</td>
                <td style="text-align:right;"><span style="background:{'#10b981' if r['lift'] > 1.5 else '#3b82f6'}; color:white; padding:2px 8px; border-radius:4px; font-size:10px; font-weight:700;">{'HIGH' if r['lift'] > 1.5 else 'MOD'}</span></td>
            </tr>
        """ for _, r in current_selection_rules.nlargest(3, 'lift').iterrows()])

        # Specialist List (Refined Professional Cards)
        spec_list_html = "".join([f"""
            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 15px; display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.02); transition: transform 0.2s ease;" onmouseover="this.style.transform='translateY(-2px)'" onmouseout="this.style.transform='translateY(0)'">
                <div style="display:flex; align-items:center; gap:15px;">
                    <div style="width:45px; height:45px; border-radius:22px; background:linear-gradient(135deg, #eff6ff, #dbeafe); display:flex; align-items:center; justify-content:center; color:#3b82f6; font-size:22px; border:1px solid #bfdbfe;">👨‍⚕️</div>
                    <div>
                        <div style="font-size:14px; font-weight:700; color:#0f172a;">{d['role']}</div>
                        <div style="font-size:11px; color:#3b82f6; font-weight:600; margin-bottom:2px;">Qualified for: {d['condition']}</div>
                        <div style="font-size:10px; color:#64748b; font-style:italic;">{d['qualifier']}</div>
                    </div>
                </div>
                <div style="text-align:right;">
                    <span style="color:#059669; font-size:10px; font-weight:700; background:rgba(16,185,129,0.1); padding:3px 10px; border-radius:100px; border:1px solid rgba(16,185,129,0.2);">Qualified</span>
                    <div class="privateConsultTrigger" data-role="{d['role']}" style="font-size:10px; color:#3b82f6; margin-top:6px; cursor:pointer; font-weight:700; text-decoration:underline;">Initiate Consult</div>
                </div>
            </div>
        """ for d in specialists_data[:3]])

        adv_modal_content = f"""
        <div id="advisoryModal" style="display: none; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(255, 255, 255, 0.2); backdrop-filter: blur(15px); -webkit-backdrop-filter: blur(15px); z-index: 9999999; align-items: center; justify-content: center; color: #0f172a; font-family: 'Inter', sans-serif;">
            <div style="background: #ffffff; width: 1000px; max-width: 95%; height: 90vh; border-radius: 24px; padding: 40px; border: 1px solid #e2e8f0; position: relative; overflow-y: auto; box-shadow: 0 40px 80px rgba(0,0,0,0.12);">
                <div id="closeAdvisoryBtn" style="position: absolute; top: 30px; right: 30px; cursor: pointer; font-size: 24px; color: #94a3b8;">✕</div>
                
                <!-- Header -->
                <div style="margin-bottom: 30px; border-bottom: 1px solid #f1f5f9; padding-bottom: 20px;">
                    <div style="display:flex; align-items:center; gap:12px;">
                        <span style="font-size:32px;">🩺</span>
                        <h2 style="margin: 0; font-size: 28px; font-weight: 800; letter-spacing: -0.5px; color:#0f172a;">Multi-Disciplinary Consult</h2>
                    </div>
                    <p style="color: #64748b; font-size: 14px; margin-top: 5px;">AI-assisted specialist recommendations based on clinical patterns discovered via FP-Growth mining.</p>
                </div>

                <!-- Top Metrics -->
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px;">
                    <div style="background: #f8fafc; padding: 20px; border-radius: 16px; border: 1px solid #e2e8f0;">
                        <div style="display:flex; align-items:center; gap:12px;">
                            <div style="font-size:24px; color:#8b5cf6;">👥</div>
                            <div><div style="font-size:24px; font-weight:800; color:#0f172a;">{num_specialties}</div><div style="font-size:10px; color:#64748b; text-transform:uppercase; font-weight:700;">Specialties</div></div>
                        </div>
                    </div>
                    <div style="background: #f8fafc; padding: 20px; border-radius: 16px; border: 1px solid #e2e8f0;">
                        <div style="display:flex; align-items:center; gap:12px;">
                            <div style="font-size:24px; color:#3b82f6;">🛡</div>
                            <div><div style="font-size:24px; font-weight:800; color:#0f172a;">{high_strength}</div><div style="font-size:10px; color:#64748b; text-transform:uppercase; font-weight:700;">Associations</div></div>
                        </div>
                    </div>
                    <div style="background: #f8fafc; padding: 20px; border-radius: 16px; border: 1px solid #e2e8f0;">
                        <div style="display:flex; align-items:center; gap:12px;">
                            <div style="font-size:24px; color:#10b981;">📈</div>
                            <div><div style="font-size:24px; font-weight:800; color:#0f172a;">{conf_min:.2f}</div><div style="font-size:10px; color:#64748b; text-transform:uppercase; font-weight:700;">Min Conf.</div></div>
                        </div>
                    </div>
                    <div style="background: #f8fafc; padding: 20px; border-radius: 16px; border: 1px solid #e2e8f0;">
                        <div style="display:flex; align-items:center; gap:12px;">
                            <div style="font-size:24px; color:#f59e0b;">📊</div>
                            <div><div style="font-size:24px; font-weight:800; color:#0f172a;">{lift_max:.2f}</div><div style="font-size:10px; color:#64748b; text-transform:uppercase; font-weight:700;">Max Lift</div></div>
                        </div>
                    </div>
                </div>

                <!-- Main Content Grid -->
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 30px;">
                    <!-- Key Conditions -->
                    <div style="background: white; padding: 25px; border-radius: 20px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px rgba(0,0,0,0.02);">
                        <h4 style="margin: 0 0 20px 0; font-size: 16px; font-weight:700;">1. Key Associated Conditions</h4>
                        <table style="width:100%; border-collapse: collapse; font-size: 12px;">
                            <tr style="color:#64748b; text-transform:uppercase; font-size:10px; border-bottom: 2px solid #f1f5f9;">
                                <th style="text-align:left; padding-bottom:10px;">Condition</th>
                                <th style="text-align:center; padding-bottom:10px;">Supp.</th>
                                <th style="text-align:center; padding-bottom:10px;">Conf.</th>
                                <th style="text-align:center; padding-bottom:10px;">Lift</th>
                                <th style="text-align:right; padding-bottom:10px;">Strength</th>
                            </tr>
                            {table1_rows}
                        </table>
                    </div>

                    <!-- Recommended Specialists -->
                    <div style="background: white; padding: 25px; border-radius: 20px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px rgba(0,0,0,0.02);">
                        <h4 style="margin: 0 0 20px 0; font-size: 16px; font-weight:700;">2. Recommended Specialists</h4>
                        {spec_list_html}
                    </div>
                </div>

                <!-- Row 2 Grid -->
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 30px;">
                    <!-- Top Rules -->
                    <div style="background: white; padding: 25px; border-radius: 20px; border: 1px solid #e2e8f0;">
                        <h4 style="margin: 0 0 20px 0; font-size: 16px; font-weight:700;">3. Association Evidence Board</h4>
                        <table style="width:100%; border-collapse: collapse; font-size: 12px;">
                            <tr style="color:#64748b; text-transform:uppercase; font-size:10px; border-bottom: 2px solid #f1f5f9;">
                                <th style="text-align:left; padding-bottom:10px;">Logic (If &rarr; Then)</th>
                                <th style="text-align:center; padding-bottom:10px;">Supp.</th>
                                <th style="text-align:center; padding-bottom:10px;">Conf.</th>
                                <th style="text-align:center; padding-bottom:10px;">Lift</th>
                            </tr>
                            {table1_rows}
                        </table>
                    </div>

                    <!-- Consultation Pathway -->
                    <div style="background: white; padding: 25px; border-radius: 20px; border: 1px solid #e2e8f0;">
                        <h4 style="margin: 0 0 20px 0; font-size: 16px; font-weight:700;">4. Suggested Clinical Journey</h4>
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:20px;">
                            <div style="text-align:center; flex:1;">
                                <div style="width:40px; height:40px; border-radius:20px; background:#f3e8ff; color:#8b5cf6; margin:0 auto; display:flex; align-items:center; justify-content:center; font-weight:700;">1</div>
                                <div style="font-size:10px; font-weight:700; margin-top:8px;">Diagnosis</div>
                            </div>
                            <div style="color:#e2e8f0;">&rarr;</div>
                            <div style="text-align:center; flex:1;">
                                <div style="width:40px; height:40px; border-radius:20px; background:#dcfce7; color:#10b981; margin:0 auto; display:flex; align-items:center; justify-content:center; font-weight:700;">2</div>
                                <div style="font-size:10px; font-weight:700; margin-top:8px;">Comorbidity</div>
                            </div>
                            <div style="color:#e2e8f0;">&rarr;</div>
                            <div style="text-align:center; flex:1;">
                                <div style="width:40px; height:40px; border-radius:20px; background:#dbeafe; color:#3b82f6; margin:0 auto; display:flex; align-items:center; justify-content:center; font-weight:700;">3</div>
                                <div style="font-size:10px; font-weight:700; margin-top:8px;">Consult</div>
                            </div>
                            <div style="color:#e2e8f0;">&rarr;</div>
                            <div style="text-align:center; flex:1;">
                                <div style="width:40px; height:40px; border-radius:20px; background:#ffedd5; color:#f59e0b; margin:0 auto; display:flex; align-items:center; justify-content:center; font-weight:700;">4</div>
                                <div style="font-size:10px; font-weight:700; margin-top:8px;">Care Plan</div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Footer Insight -->
                <div style="background: linear-gradient(135deg, #f8fafc, #eff6ff); border-radius: 20px; padding: 30px; display: flex; align-items: center; justify-content: space-between; border: 1px solid #dbeafe;">
                    <div style="display:flex; align-items:center; gap:20px; flex:1;">
                        <div style="width:60px; height:60px; border-radius:30px; background:white; display:flex; align-items:center; justify-content:center; border:1px solid #dbeafe; font-size:24px;">💡</div>
                        <div>
                            <h5 style="margin:0; font-size:16px; font-weight:800;">Executive Clinical Summary</h5>
                            <p style="margin:5px 0 0 0; font-size:13px; color:#475569; line-height:1.6;">Based on the selected profile, {st.session_state['primary_diag']} shows significant co-occurrence with {con_c}. Collaborative care with {spec_str} is strongly advised to mitigate cross-conditional risks.</p>
                        </div>
                    </div>
                    <div style="display:flex; gap:10px; align-items:flex-end;">
                        <div style="width:80px; height:80px; background:url('https://cdn-icons-png.flaticon.com/512/3774/3774299.png') no-repeat center; background-size:contain;"></div>
                        <div style="width:90px; height:90px; background:url('https://cdn-icons-png.flaticon.com/512/3774/3774293.png') no-repeat center; background-size:contain;"></div>
                    </div>
                </div>

                <div style="text-align:center; margin-top:30px; font-size:11px; color:#94a3b8; font-weight:500;">
                    🛡 CONFIDENTIAL CLINICAL INTELLIGENCE | FP-GROWTH ENGINE V1.2
                </div>
            </div>
        </div>
        """
    else:
        consult_trigger_html = f"""
        <div class="glass-card" style="margin-top: 20px; text-align:center; padding:20px; color:#64748b;">
            <div style="font-size:24px; margin-bottom:10px;">🩺</div>
            <div style="font-size:13px;">Select a diagnosis to generate an AI consult.</div>
        </div>
        """

    st_html(f"""
    <div class="glass-card" style="background: transparent; box-shadow: none; border: none; padding: 0;">
        <h3 style="margin-bottom: 10px;">Dynamic Care Plan</h3>
        {timeline_html}
    </div>
    <div id="demoBtn" class="glass-card" style="cursor: pointer; text-align: center; padding: 15px; border: 1px solid rgba(255,255,255,0.6);">
        <h3 style="margin:0;">Demographic Comorbidity Patterns</h3>
    </div>
    {consult_trigger_html}
    {adv_modal_content}
    """)
    
    # --- Step 7: UI/UX Highlight Top Rule (Relocated to Col1) ---
    if rules_df is not None and len(rules_df) > 0:
        top_rule = rules_df.nlargest(1, 'lift').iloc[0]
        ant_clean = clean_frozenset(top_rule['antecedents'])
        con_clean = clean_frozenset(top_rule['consequents'])
        
        st_html(f"""
        <div class="glass-card" style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: white; padding: 20px; border: none; margin-top: 20px; position: relative; overflow: hidden;">
            <div style="position: absolute; top: -10px; right: -10px; font-size: 80px; opacity: 0.1; transform: rotate(15deg);">💡</div>
            <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #94a3b8; margin-bottom: 5px; font-weight: 700;">Top Clinical Insight</div>
            <div style="font-size: 16px; font-weight: 700; margin-bottom: 15px;">
                Patients with <span style="color: #3b82f6;">{ant_clean}</span> are highly likely to develop <span style="color: #ef4444;">{con_clean}</span>
            </div>
            <div style="display: flex; gap: 15px;">
                <div><div style="font-size: 10px; color: #94a3b8;">LIFT</div><div style="font-size: 18px; font-weight: 700;">{top_rule['lift']:.2f}</div></div>
                <div style="border-left: 1px solid rgba(255,255,255,0.1); padding-left: 15px;">
                    <div style="font-size: 10px; color: #94a3b8;">CONFIDENCE</div><div style="font-size: 18px; font-weight: 700;">{top_rule['confidence']*100:.1f}%</div>
                </div>
            </div>
        </div>
        """)
    
    # Universal Info Modal
    st_html("""
    <div id="infoModal" style="display: none; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(255, 255, 255, 0.3); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); z-index: 9999999; align-items: center; justify-content: center;">
        <div id="infoModalInner" style="background: white; width: 500px; border-radius: 20px; padding: 30px; box-shadow: 0 25px 50px rgba(0,0,0,0.15); border: 1px solid #e2e8f0; animation: zoomIn 0.3s ease-out;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; border-bottom: 1px solid #f1f5f9; padding-bottom: 15px;">
                <h3 id="infoModalTitle" style="margin: 0; font-size: 18px;">Feature Information</h3>
                <div id="infoModalClose" style="cursor: pointer; font-size: 20px; color: #64748b;">✕</div>
            </div>
            <div id="infoModalContent" style="font-size: 14px; color: #475569; line-height: 1.6;"></div>
        </div>
    </div>
    """)

    # Demographic Patterns Modal
    if rules_df is not None:
        table_rows = "".join([f'<tr style="border-bottom:1px solid #f1f5f9;"><td style="padding:10px; font-size:12px; font-weight:600;">{clean_frozenset(r["antecedents"])} &rarr; {clean_frozenset(r["consequents"])}</td><td style="text-align:center; font-size:12px;">{r["support"]:.3f}</td><td style="text-align:center; font-size:12px; font-weight:700; color:#3b82f6;">{r["confidence"]:.3f}</td></tr>' for _, r in rules_df.nlargest(8, 'lift').iterrows()])
    else:
        table_rows = "<tr><td colspan='3'>No demographic patterns found.</td></tr>"

    st_html(f"""
    <div id="demoModal" style="display: none; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(255, 255, 255, 0.2); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); z-index: 999999; align-items: center; justify-content: center; cursor: pointer;">
        <div id="demoModalInner" style="background: rgba(255, 255, 255, 0.95); box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25); border-radius: 16px; padding: 25px; width: 700px; max-width: 90%; cursor: default; position: relative;">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #e2e8f0; padding-bottom: 15px;">
                <h2 style="margin: 0; font-size: 20px; font-weight: 700; color: #0f172a;">Live Demographic Comorbidity Insights</h2>
                <div id="demoModalClose" style="cursor: pointer; font-size: 24px;">✕</div>
            </div>
            <div id="demoModalTableContainer" style="margin-top: 15px;">
                <table style="width: 100%; border-collapse: collapse; font-family: 'Inter', sans-serif;">
                    <tr style="background: #0f172a; color: white;">
                        <th style="padding: 10px; text-align: left; border-top-left-radius: 8px; font-size: 13px;">Rule</th>
                        <th style="padding: 10px; text-align: center; font-size: 13px;">Support</th>
                        <th style="padding: 10px; text-align: center; border-top-right-radius: 8px; font-size: 13px;">Confidence</th>
                    </tr>
                    {table_rows}
                </table>
            </div>
            <div style="margin-top: 15px; font-size: 11px; color: #64748b;">*Rules ranked by Lift to prioritize strongest clinical correlations.</div>
        </div>
    </div>
    """)

    # Private Consult Modal (Renamed and Dynamized)
    primary_spec = specialists_data[0] if specialists_data else {'role': 'General Physician'}
    st_html(f"""
    <div id="privateConsultModal" style="display: none; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(255, 255, 255, 0.2); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); z-index: 10000000; align-items: center; justify-content: center; cursor: pointer;" onclick="document.getElementById('privateConsultModal').style.display='none';">
        <div style="width:100%; max-width: 900px; height: 600px; display: flex; flex-direction: column; background: #ffffff; border-radius: 24px; overflow: hidden; box-shadow: 0 40px 80px rgba(0,0,0,0.15); border: 1px solid #e2e8f0; font-family: 'Inter', sans-serif; cursor: default; position: relative;" onclick="event.stopPropagation();">
            <div style="background: #0f172a; color: white; padding: 20px 30px; font-size: 18px; font-weight: 700; display: flex; justify-content: space-between; align-items: center;">
                <div style="display:flex; align-items:center; gap:12px;">
                    <span style="font-size:24px;">💬</span>
                    <span>Private Multi-Disciplinary Consult</span>
                </div>
                <div id="closePrivateConsultBtn" style="cursor: pointer; font-size: 24px; line-height: 1; color:#94a3b8;">✕</div>
            </div>
            <div style="display: flex; flex- 1; overflow: hidden;">
                <!-- Sidebar -->
                <div style="width: 300px; border-right: 1px solid #e2e8f0; background: #f8fafc; padding: 20px;">
                    <div style="font-size:11px; font-weight:800; color:#94a3b8; text-transform:uppercase; letter-spacing:1px; margin-bottom:20px;">Active Consultant</div>
                    <div style="display: flex; align-items: center; padding: 15px; background:white; border-radius:16px; border:1px solid #e2e8f0; box-shadow: 0 4px 6px rgba(0,0,0,0.02);">
                        <div style="position: relative; margin-right: 15px;">
                            <div style="width: 45px; height: 45px; border-radius: 50%; background:#eff6ff; display:flex; align-items:center; justify-content:center; font-size:24px; border:1px solid #dbeafe;">👨‍⚕️</div>
                            <div style="position: absolute; bottom: 0; right: 0; width: 12px; height: 12px; border-radius: 50%; background: #22c55e; border: 2px solid white;"></div>
                        </div>
                        <div>
                            <div id="activeConsultantName" style="font-weight: 700; font-size: 14px; color: #0f172a;">{primary_spec['role']}</div>
                            <div id="activeConsultantRole" style="font-size: 12px; color: #64748b;">Clinical Expert</div>
                        </div>
                    </div>
                    <div style="margin-top:30px; padding:15px; background:#eff6ff; border-radius:12px; border-left:4px solid #3b82f6;">
                        <div style="font-size:12px; font-weight:700; color:#3b82f6; margin-bottom:5px;">Clinical Context</div>
                        <div style="font-size:11px; color:#475569; line-height:1.4;">Analyzing secondary risks for <span style="font-weight:700;">{st.session_state['primary_diag']}</span> and associated patterns.</div>
                    </div>
                </div>
                <!-- Chat Area -->
                <div style="flex: 1; display: flex; flex-direction: column; background: #ffffff;">
                    <div style="flex: 1; padding: 30px; overflow-y: auto; display: flex; flex-direction: column; gap: 20px;">
                        <div style="align-self: flex-start; max-width: 80%; background: #f1f5f9; padding: 15px; border-radius: 16px 16px 16px 4px; font-size: 13px; line-height: 1.5; color: #0f172a;">
                            I've reviewed the FP-Growth patterns for this patient. The lift of {lift_max:.2f} for {con_c} suggests we should prioritize prophylactic screening.
                        </div>
                        <div style="align-self: flex-end; max-width: 80%; background: #0f172a; color: white; padding: 15px; border-radius: 16px 16px 4px 16px; font-size: 13px; line-height: 1.5;">
                            Agreed. Based on the confidence levels, I've updated the Care Plan with targeted monitoring.
                        </div>
                    </div>
                    <div style="padding: 20px; border-top: 1px solid #e2e8f0; background:#f8fafc; display:flex; gap:10px;">
                        <div style="flex:1; background:white; border:1px solid #e2e8f0; border-radius:12px; padding:12px; color:#94a3b8; font-size:13px;">Type clinical observation...</div>
                        <button style="background:#3b82f6; color:white; border:none; width:45px; height:45px; border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:20px; cursor:pointer;">➔</button>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """)
    
    # Navbar Functionality Modals
    st_html("""
    <div id="appointmentsModal" style="display: none; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(255, 255, 255, 0.2); backdrop-filter: blur(12px); z-index: 1000000; align-items: center; justify-content: center;">
        <div class="glass-card" style="width: 600px; padding: 30px; animation: zoomIn 0.3s ease;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
                <h3 style="margin:0; font-size:20px;">Upcoming Clinical Appointments</h3>
                <div id="closeAppointments" style="cursor:pointer; font-size:24px;">✕</div>
            </div>
            <div style="display:flex; flex-direction:column; gap:15px;">
                <div style="padding:15px; background:#f8fafc; border-radius:12px; border-left:4px solid #3b82f6;">
                    <div style="font-weight:700; color:#0f172a;">Endocrinology Follow-up</div>
                    <div style="font-size:12px; color:#64748b;">Monday, Oct 24 • 10:30 AM</div>
                    <div style="margin-top:5px; font-size:13px; color:#475569;">Context: Review metabolic response to current regimen. Association LIFT: 1.8.</div>
                </div>
                <div style="padding:15px; background:#f8fafc; border-radius:12px; border-left:4px solid #ef4444;">
                    <div style="font-weight:700; color:#0f172a;">Cardiovascular Screening</div>
                    <div style="font-size:12px; color:#64748b;">Thursday, Oct 27 • 02:15 PM</div>
                    <div style="margin-top:5px; font-size:13px; color:#475569;">Context: Preventive check for detected hypertension risk chain.</div>
                </div>
            </div>
        </div>
    </div>

    <div id="scheduleModal" style="display: none; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(255, 255, 255, 0.2); backdrop-filter: blur(12px); z-index: 1000000; align-items: center; justify-content: center;">
        <div class="glass-card" style="width: 600px; padding: 30px; animation: zoomIn 0.3s ease;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
                <h3 style="margin:0; font-size:20px;">Daily Care Schedule</h3>
                <div id="closeSchedule" style="cursor:pointer; font-size:24px;">✕</div>
            </div>
            <div style="display:flex; flex-direction:column; gap:10px;">
                <div style="display:flex; justify-content:space-between; padding:10px; border-bottom:1px solid #f1f5f9;">
                    <span style="font-weight:700; color:#3b82f6;">08:00 AM</span>
                    <span style="flex:1; margin-left:20px;">Morning Vitals & Medication</span>
                </div>
                <div style="display:flex; justify-content:space-between; padding:10px; border-bottom:1px solid #f1f5f9;">
                    <span style="font-weight:700; color:#3b82f6;">12:00 PM</span>
                    <span style="flex:1; margin-left:20px;">Glucose Level Check (Pre-Meal)</span>
                </div>
                <div style="display:flex; justify-content:space-between; padding:10px; border-bottom:1px solid #f1f5f9;">
                    <span style="font-weight:700; color:#3b82f6;">04:00 PM</span>
                    <span style="flex:1; margin-left:20px;">Blood Pressure Monitoring</span>
                </div>
                <div style="display:flex; justify-content:space-between; padding:10px;">
                    <span style="font-weight:700; color:#3b82f6;">08:00 PM</span>
                    <span style="flex:1; margin-left:20px;">Nightly Review & Data Sync</span>
                </div>
            </div>
        </div>
    </div>

    <div id="labsModal" style="display: none; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(255, 255, 255, 0.2); backdrop-filter: blur(12px); z-index: 1000000; align-items: center; justify-content: center;">
        <div class="glass-card" style="width: 700px; padding: 30px; animation: zoomIn 0.3s ease;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
                <h3 style="margin:0; font-size:20px;">Latest Laboratory Results</h3>
                <div id="closeLabs" style="cursor:pointer; font-size:24px;">✕</div>
            </div>
            <table style="width:100%; border-collapse:collapse;">
                <tr style="background:#0f172a; color:white;">
                    <th style="padding:10px; text-align:left; border-top-left-radius:8px;">Test</th>
                    <th style="padding:10px; text-align:center;">Result</th>
                    <th style="padding:10px; text-align:center;">Normal Range</th>
                    <th style="padding:10px; text-align:right; border-top-right-radius:8px;">Status</th>
                </tr>
                <tr style="border-bottom:1px solid #f1f5f9;">
                    <td style="padding:10px;">HbA1c</td>
                    <td style="padding:10px; text-align:center; font-weight:700;">6.8%</td>
                    <td style="padding:10px; text-align:center;">4.0 - 5.6%</td>
                    <td style="padding:10px; text-align:right; color:#eab308; font-weight:700;">Borderline</td>
                </tr>
                <tr style="border-bottom:1px solid #f1f5f9;">
                    <td style="padding:10px;">LDL Cholesterol</td>
                    <td style="padding:10px; text-align:center; font-weight:700;">112 mg/dL</td>
                    <td style="padding:10px; text-align:center;">&lt; 100 mg/dL</td>
                    <td style="padding:10px; text-align:right; color:#ef4444; font-weight:700;">High</td>
                </tr>
                <tr style="border-bottom:1px solid #f1f5f9;">
                    <td style="padding:10px;">Creatinine</td>
                    <td style="padding:10px; text-align:center; font-weight:700;">0.9 mg/dL</td>
                    <td style="padding:10px; text-align:center;">0.7 - 1.3 mg/dL</td>
                    <td style="padding:10px; text-align:right; color:#22c55e; font-weight:700;">Normal</td>
                </tr>
            </table>
            <div style="margin-top:15px; font-size:12px; color:#64748b;">*Results interpreted relative to analyzed comorbid clusters (Diabetes + Hypertension).</div>
        </div>
    </div>

    <div id="profileModal" style="display: none; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(255, 255, 255, 0.2); backdrop-filter: blur(12px); z-index: 1000000; align-items: center; justify-content: center;">
        <div class="glass-card" style="width: 500px; padding: 30px; animation: zoomIn 0.3s ease;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
                <h3 style="margin:0; font-size:20px;">Patient Clinical Profile</h3>
                <div id="closeProfile" style="cursor:pointer; font-size:24px;">✕</div>
            </div>
            <div style="display:flex; align-items:center; gap:20px; margin-bottom:20px;">
                <div style="width:80px; height:80px; border-radius:50%; background:#f1f5f9; display:flex; align-items:center; justify-content:center; font-size:40px;">👤</div>
                <div>
                    <div style="font-size:22px; font-weight:700; color:#0f172a;">Patient #2440</div>
                    <div style="font-size:14px; color:#64748b;">Age: 64 • Gender: Male • Blood: A+</div>
                </div>
            </div>
            <div style="display:flex; flex-direction:column; gap:10px;">
                <div style="padding:10px; background:#eff6ff; border-radius:8px;">
                    <div style="font-size:12px; font-weight:700; color:#3b82f6; text-transform:uppercase;">Primary Diagnoses</div>
                    <div style="font-size:15px; font-weight:600; color:#0f172a;">Diabetes Mellitus, Essential Hypertension</div>
                </div>
                <div style="padding:10px; background:#fef2f2; border-radius:8px;">
                    <div style="font-size:12px; font-weight:700; color:#ef4444; text-transform:uppercase;">Risk Indicators</div>
                    <div style="font-size:15px; font-weight:600; color:#0f172a;">High Comorbidity Score (0.88), Elevated BMP</div>
                </div>
            </div>
        </div>
    </div>
    """)

# === MIDDLE COLUMN: Transparent buffer ===
with col2:
    st_html("<div style='height: 80vh;'></div>")

# === RIGHT COLUMN: AI Analytics ===
with col3:
    temp_val = 38.5
    temp_color = "#ef4444" if temp_val > 38 else "#22c55e" 
    temp_bg = "rgba(239, 68, 68, 0.05)" if temp_val > 38 else "rgba(34, 197, 94, 0.05)"
    
    st_html(f"""
    <div class="vitals-row">
        <div class="vital-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div class="vital-label">Total Visits Analyzed</div>
                <div id="infoVisits" style="cursor:pointer; color:#3b82f6; font-size:12px;">ⓘ</div>
            </div>
            <div class="vital-value" style="color: #0f172a;">{total_visits}</div>
            <div class="ekg-line" style="background: linear-gradient(90deg, transparent 0%, #3b82f6 50%, transparent 100%);"></div>
        </div>
        <div class="vital-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div class="vital-label">Avg. Events / Visit</div>
                <div id="infoEvents" style="cursor:pointer; color:#3b82f6; font-size:12px;">ⓘ</div>
            </div>
            <div class="vital-value" style="color: #0f172a;">{avg_events}</div>
            <div class="ekg-line" style="background: linear-gradient(90deg, transparent 0%, #eab308 50%, transparent 100%);"></div>
        </div>
    </div>

    <div class="vitals-row">
        <div class="vital-card">
            <div style="display:flex; justify-content:space-between;">
                <div><div class="vital-label">Heart rate</div><div class="vital-value" style="color: #0f172a;">82 bpm</div></div>
                <div class="heart-icon" style="font-size:24px;">❤</div>
            </div>
            <div class="ekg-line"></div>
        </div>
        <div class="vital-card">
            <div style="display:flex; justify-content:space-between;">
                <div><div class="vital-label">Brain activity</div><div class="vital-value" style="color: #0f172a;">120 Hz</div></div>
                <div class="brain-icon" style="font-size:24px;">🧠</div>
            </div>
            <div class="ekg-line" style="background: linear-gradient(90deg, transparent 0%, #eab308 50%, transparent 100%);"></div>
        </div>
        <div class="vital-card" style="background: {temp_bg}; border: 1px solid {temp_color}33;">
            <div style="display:flex; justify-content:space-between;">
                <div><div class="vital-label">Temperature</div><div class="vital-value" style="color: {temp_color};">{temp_val}°C</div></div>
                <div style="font-size:24px; color:{temp_color};">🌡</div>
            </div>
            <div class="ekg-line" style="background: linear-gradient(90deg, transparent 0%, {temp_color} 50%, transparent 100%);"></div>
        </div>
    </div>

    <div class="glass-card" style="margin-top:20px;">
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:20px;">
            <span style="font-size:24px;">⛓</span>
            <h3 style="margin:0; font-size:18px;">Comorbidity Heatmap & Graph Hybrid</h3>
        </div>
        
        <div style="margin-bottom:25px;">
            <div style="font-size:12px; font-weight:700; color:#64748b; text-transform:uppercase; margin-bottom:10px;">Clinical Metric Matrix</div>
            <table style="width:100%; border-collapse: collapse; font-family: 'Inter', sans-serif;">
                <thead>
                    <tr style="background: #f8fafc; border-bottom: 2px solid #e2e8f0;">
                        <th style="padding: 12px 8px; text-align: left; font-size: 11px; font-weight: 800; color: #475569;">#</th>
                        <th style="padding: 12px 8px; text-align: left; font-size: 11px; font-weight: 800; color: #475569;">Metric Matrix ↕</th>
                        <th style="padding: 12px 8px; text-align: center; font-size: 11px; font-weight: 800; color: #475569;">Support ↕</th>
                        <th style="padding: 12px 8px; text-align: center; font-size: 11px; font-weight: 800; color: #475569;">Confidence ↕</th>
                    </tr>
                </thead>
                <tbody>
    """)
    
    if rules_df is not None:
        for i, (idx, row) in enumerate(rules_df.nlargest(5, 'confidence').iterrows()):
            ant_c = clean_frozenset(row['antecedents'])
            con_c = clean_frozenset(row['consequents'])
            conf_pct = row['confidence'] * 100
            
            # Dynamic colors based on values
            supp_color = f"rgba(15, 23, 42, {row['support']*5})"
            conf_color = f"rgba(239, 68, 68, {row['confidence']})" if row['confidence'] > 0.8 else f"rgba(15, 23, 42, {row['confidence']})"
            
            st_html(f"""
                <tr style="border-bottom: 1px solid #f1f5f9;">
                    <td style="padding: 12px 8px; font-size: 12px; font-weight: 700; color: #64748b;">{i+1}</td>
                    <td style="padding: 12px 8px; font-size: 11px; color: #334155; font-weight: 600;">{ant_clean} &rarr; {con_clean}</td>
                    <td style="padding: 12px 8px; text-align: center; font-size: 11px; font-weight: 700; background: {supp_color}; color: {'white' if row['support'] > 0.1 else '#0f172a'};">{row['support']:.2f}</td>
                    <td style="padding: 12px 8px; text-align: center; font-size: 11px; font-weight: 700; background: {conf_color}; color: {'white' if row['confidence'] > 0.7 else '#0f172a'};">{row['confidence']:.2f}</td>
                </tr>
            """)
            
    st_html("""
                </tbody>
            </table>
        </div>
        
        <div style="background: #f8fafc; border-radius: 12px; padding: 20px; border: 1px solid #e2e8f0;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
                <div style="font-size:12px; font-weight:700; color:#64748b; text-transform:uppercase;">Discovery Engine Meta</div>
                <div style="background:#22c55e; width:8px; height:8px; border-radius:50%;"></div>
            </div>
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:15px;">
                <div><div style="font-size:10px; color:#94a3b8;">Mining Logic</div><div style="font-size:12px; font-weight:700;">FP-Growth</div></div>
                <div><div style="font-size:10px; color:#94a3b8;">Rec. Count</div><div style="font-size:12px; font-weight:700;">10,000+</div></div>
                <div><div style="font-size:10px; color:#94a3b8;">Avg Support</div><div style="font-size:12px; font-weight:700;">0.042</div></div>
                <div><div style="font-size:10px; color:#94a3b8;">System Status</div><div style="font-size:12px; font-weight:700; color:#22c55e;">Nominal</div></div>
            </div>
        </div>
    </div>
    """)

# --- Step 4: Pattern Selection (Relocated to Footer for Minimalist Feel) ---
with st.form("pattern_form"):
    st_html("<h3 style='margin-bottom:15px;'>Pattern Selection Parameter Tuning</h3>")
    c1, c2, c3 = st.columns(3)
    p_diag = c1.selectbox("Primary Diagnosis Filter", ["All"] + all_items, index=0)
    s_diag = c2.selectbox("Secondary Diagnosis Filter", ["All"] + all_items, index=0)
    
    # Custom Select Logic for state management
    if c3.form_submit_button("RE-MINE CLINICAL PATTERNS"):
        st.session_state['primary_diag'] = p_diag
        st.session_state['secondary_diag'] = s_diag
        st.rerun()

# --- Step 8: Modals JS Logic ---
st_html("""
<script>
    // Helper to get Streamlit's iframe document
    const doc = window.parent.document;
    
    // Modal Selectors
    const modals = {
        demo: { btn: doc.getElementById('demoBtn'), modal: doc.getElementById('demoModal'), close: doc.getElementById('demoModalClose') },
        advisory: { btn: doc.getElementById('advisoryBtn'), modal: doc.getElementById('advisoryModal'), close: doc.getElementById('closeAdvisoryBtn') },
        appointments: { btn: doc.getElementById('navAppointments'), modal: doc.getElementById('appointmentsModal'), close: doc.getElementById('closeAppointments') },
        schedule: { btn: doc.getElementById('navSchedule'), modal: doc.getElementById('scheduleModal'), close: doc.getElementById('closeSchedule') },
        labs: { btn: doc.getElementById('navLabs'), modal: doc.getElementById('labsModal'), close: doc.getElementById('closeLabs') },
        profile: { btn: doc.getElementById('navProfile'), modal: doc.getElementById('profileModal'), close: doc.getElementById('closeProfile') }
    };

    Object.values(modals).forEach(({btn, modal, close}) => {
        if(btn && modal) {
            btn.onclick = () => modal.style.display = 'flex';
            if(close) close.onclick = () => modal.style.display = 'none';
            modal.onclick = (e) => { if(e.target === modal) modal.style.display = 'none'; };
        }
    });

    // Info Modals
    const infoModal = doc.getElementById('infoModal');
    const infoTitle = doc.getElementById('infoModalTitle');
    const infoContent = doc.getElementById('infoModalContent');
    const infoClose = doc.getElementById('infoModalClose');

    const infoData = {
        'infoVisits': { title: 'Total Visits Analyzed', text: 'This represents the total volume of clinical transaction records mined from the HIPAA-compliant dataset to discover comorbidity trends.' },
        'infoEvents': { title: 'Avg. Events / Visit', text: 'The mean number of concurrent diagnoses or clinical interventions recorded per patient encounter across the selected cohort.' }
    };

    Object.entries(infoData).forEach(([id, data]) => {
        const icon = doc.getElementById(id);
        if(icon) {
            icon.onclick = () => {
                infoTitle.innerText = data.title;
                infoContent.innerText = data.text;
                infoModal.style.display = 'flex';
            };
        }
    });
    if(infoClose) infoClose.onclick = () => infoModal.style.display = 'none';

    // Private Consult Trigger
    const privateModal = doc.getElementById('privateConsultModal');
    const closePrivate = doc.getElementById('closePrivateConsultBtn');
    
    doc.querySelectorAll('.privateConsultTrigger').forEach(el => {
        el.onclick = () => {
            doc.getElementById('activeConsultantName').innerText = el.getAttribute('data-role');
            privateModal.style.display = 'flex';
        };
    });
    if(closePrivate) closePrivate.onclick = () => privateModal.style.display = 'none';

</script>
""")
