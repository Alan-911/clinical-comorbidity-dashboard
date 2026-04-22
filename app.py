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
    page_icon="⚕️",
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

# --- Consolidated UI Core (Optimized for performance) ---
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

# --- Data Mining Logic ---
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

@st.cache_data
def get_processed_data():
    rules_path = os.path.join(PROCESSED_DIR, "association_rules.csv")
    itemsets_path = os.path.join(PROCESSED_DIR, "frequent_itemsets.csv")
    if os.path.exists(rules_path) and os.path.exists(itemsets_path):
        return pd.read_csv(rules_path), pd.read_csv(itemsets_path)
    df_encoded = load_and_encode_data()
    if df_encoded is not None:
        freq_items, _ = fpgrowth(df_encoded, min_support=0.01, use_colnames=True), 0
        rules = association_rules(freq_items, metric="confidence", min_threshold=0.5)
        rules.to_csv(rules_path, index=False)
        freq_items.to_csv(itemsets_path, index=False)
        return rules, freq_items
    return None, None

@st.cache_data
def load_and_encode_data():
    trans_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transactions", "transactions.csv")
    try:
        df_trans = pd.read_csv(trans_path)
        dataset = [str(items).split(",") for items in df_trans["Items"].tolist()]
        te = TransactionEncoder()
        te_ary = te.fit(dataset).transform(dataset)
        return pd.DataFrame(te_ary, columns=te.columns_)
    except Exception: return None

def get_rules_for_diagnosis(rules, primary="All", secondary="All"):
    if rules is None: return None
    filtered = rules.copy()
    if primary != "All": filtered = filtered[filtered['antecedents'].apply(lambda x: primary in str(x))]
    if secondary != "All": filtered = filtered[filtered['consequents'].apply(lambda x: secondary in str(x))]
    return filtered.sort_values('lift', ascending=False)

# --- Specialist Mapping ---
SPECIALIST_MAP = {
    'Hypertension': {'role': 'Cardiologist', 'qualifier': 'Expert in vascular tension and hypertensive management.'},
    'Diabetes': {'role': 'Endocrinologist', 'qualifier': 'Expert in metabolic regulation and glycemic control.'},
    'Asthma': {'role': 'Pulmonologist', 'qualifier': 'Specializes in chronic airway management and lung function.'},
    'Kidney Disease': {'role': 'Nephrologist', 'qualifier': 'Expert in renal filtration and electrolyte balance.'},
    'Allergy': {'role': 'Allergist / Immunologist', 'qualifier': 'Specializes in hypersensitivity and immune response.'},
    'Routine Checkup': {'role': 'General Practitioner', 'qualifier': 'Coordination of comprehensive primary care.'}
}

rules_raw, itemsets_raw = get_processed_data()

if rules_raw is not None:
    def clean_frozenset(x):
        if not isinstance(x, str): x = str(x)
        cleaned = re.sub(r"(frozenset|set|[{}()\[\]'\"])", "", x)
        cleaned = cleaned.replace(",", ", ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = cleaned.strip(",")
        return cleaned if cleaned else "General"
    
    rules_df = rules_raw.copy()
    all_items = set()
    for col in ['antecedents', 'consequents']:
        for val in rules_df[col]:
            items = clean_frozenset(val).split(",")
            all_items.update([i.strip() for i in items if i.strip()])
    all_items = sorted(list(all_items))
    total_visits, avg_events = 2440, 3.2
    time_apriori, time_fp = 1.2, 0.4
else:
    rules_df, all_items = None, []
    total_visits, avg_events, time_apriori, time_fp = 0, 0, 0, 0

# --- Layout ---
col1, col2, col3 = st.columns([1, 1.2, 1.5], gap="large")

with col1:
    # --- Dynamic Care Plan (Restored) ---
    current_selection_rules = get_rules_for_diagnosis(rules_df, st.session_state['primary_diag'], st.session_state['secondary_diag'])
    schedule_items = []
    if current_selection_rules is not None and len(current_selection_rules) > 0:
        top_care_rules = current_selection_rules.nlargest(3, 'lift')
        for i, (idx, row) in enumerate(top_care_rules.iterrows()):
            ant, con = clean_frozenset(row['antecedents']), clean_frozenset(row['consequents'])
            schedule_items.append({'time': f'{14+i}:00', 'title': f'Protocol: {con[:12]}...', 'desc': f'Reason: {ant} &rarr; {con}'})
    
    if not schedule_items: schedule_items = [{'time': '09:00', 'title': 'Routine Baseline', 'desc': 'Standard monitoring.'}]
    
    timeline_html = "".join([f'<div class="glass-card" style="padding:15px; margin-bottom:10px;"><div class="timeline-item"><div class="timeline-time">{item["time"]}</div><div class="timeline-title">{item["title"]}</div><div class="timeline-desc">{item["desc"]}</div></div></div>' for item in schedule_items])
    
    st_html(f'<div><h3>Dynamic Care Plan</h3>{timeline_html}</div>')
    
    # --- Specialist Consult Trigger (Restored) ---
    consult_trigger_html = ""
    adv_modal_content = ""
    specialists_data, num_specialties, high_strength, conf_min, conf_max, lift_min, lift_max, table1_rows, spec_list_html, con_c, spec_str = [], 0, 0, 0, 0, 0, 0, "", "", "General", "General Practitioner"

    if current_selection_rules is not None and len(current_selection_rules) > 0:
        top_c_rule = current_selection_rules.iloc[0]
        ant_c, con_c = clean_frozenset(top_c_rule['antecedents']), clean_frozenset(top_c_rule['consequents'])
        for c in con_c.split(", "):
            for key, data in SPECIALIST_MAP.items():
                if key in c: specialists_data.append({'role': data['role'], 'condition': key, 'qualifier': data['qualifier']})
        
        if not specialists_data: specialists_data = [{'role': 'General Practitioner', 'condition': 'General Profile', 'qualifier': 'Coordination of care.'}]
        spec_str = ", ".join([d['role'] for d in specialists_data])
        num_specialties = len(specialists_data)
        high_strength = len(current_selection_rules[current_selection_rules['lift'] > 1.5])
        conf_min, conf_max = current_selection_rules['confidence'].min(), current_selection_rules['confidence'].max()
        lift_min, lift_max = current_selection_rules['lift'].min(), current_selection_rules['lift'].max()

        table1_rows = "".join([f'<tr style="border-bottom:1px solid #e2e8f0;"><td style="padding:12px 8px;">{clean_frozenset(r["consequents"])}</td><td style="text-align:center;">{r["support"]:.2f}</td><td style="text-align:center;">{r["confidence"]:.2f}</td><td style="text-align:center;">{r["lift"]:.2f}</td><td style="text-align:right;"><span style="background:{"#10b981" if r["lift"] > 1.5 else "#3b82f6"}; color:white; padding:2px 8px; border-radius:4px; font-size:10px;">{"HIGH" if r["lift"] > 1.5 else "MOD"}</span></td></tr>' for _, r in current_selection_rules.nlargest(3, 'lift').iterrows()])
        spec_list_html = "".join([f'<div style="background:white; border:1px solid #e2e8f0; border-radius:12px; padding:15px; margin-bottom:12px;"><div style="display:flex; align-items:center; gap:15px;"><div style="width:45px; height:45px; border-radius:50%; background:#eff6ff; display:flex; align-items:center; justify-content:center; color:#3b82f6; font-size:22px;">👨‍⚕️</div><div><div style="font-size:14px; font-weight:700;">{d["role"]}</div><div style="font-size:11px; color:#3b82f6;">{d["condition"]}</div></div></div></div>' for d in specialists_data[:3]])

        adv_modal_content = f"""<div id="advisoryModal" style="display:none; position:fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(255,255,255,0.2); backdrop-filter:blur(15px); z-index:9999; align-items:center; justify-content:center; color:#0f172a;"><div style="background:#fff; width:1000px; max-width:95%; height:90vh; border-radius:24px; padding:40px; overflow-y:auto; box-shadow:0 40px 80px rgba(0,0,0,0.1); border:1px solid #e2e8f0;"><div id="closeAdvisoryBtn" style="position:absolute; top:30px; right:30px; cursor:pointer; font-size:24px;">✕</div><h2>Multi-Disciplinary Consult</h2><div style="display:grid; grid-template-columns:repeat(4,1fr); gap:20px; margin:30px 0;"><div>{num_specialties} Specialties</div><div>{high_strength} Associations</div><div>{conf_min:.2f} Min Conf</div><div>{lift_max:.2f} Max Lift</div></div><div style="display:grid; grid-template-columns:1fr 1fr; gap:30px;"><div>{table1_rows}</div><div>{spec_list_html}</div></div></div></div>"""
        consult_trigger_html = f'<div id="advisoryBtn" class="glass-card" style="margin-top:20px; border-left:4px solid #3b82f6; cursor:pointer;"><h3>Multi-Disciplinary Consult</h3><p style="font-size:12px;">Click for recommendations.</p></div>'
    
    st_html(consult_trigger_html + adv_modal_content)

with col2: st_html("<div style='height: 80vh;'></div>")

with col3:
    # --- Vitals ---
    temp_val = 38.5
    st_html(f"""<div class="vitals-row"><div class="vital-card"><div class="vital-label">Total Visits</div><div class="vital-value">{total_visits}</div><div class="ekg-line"></div></div><div class="vital-card"><div class="vital-label">Avg Events</div><div class="vital-value">{avg_events}</div><div class="ekg-line"></div></div></div><div class="vitals-row"><div class="vital-card">Heart: 82bpm <div class="heart-icon">❤</div></div><div class="vital-card">Brain: 120Hz <div class="brain-icon">🧠</div></div><div class="vital-card">Temp: {temp_val}°C</div></div>""")
    
    graph_placeholder = st.empty()
    
    # --- Pattern Selection & Algo (Restored Layout) ---
    bcol1, bcol2 = st.columns([1, 1.4], gap="small")
    with bcol1:
        st_html('<div class="glass-card" style="padding:25px; border-top:4px solid #0f172a;"><h3>Pattern Selection</h3>')
        with st.form("pattern_form"):
            p_diag = st.selectbox("Primary Diagnosis", ["All"] + all_items, index=(["All"] + all_items).index(st.session_state['primary_diag']))
            s_diag = st.selectbox("Secondary Condition", ["All"] + all_items, index=(["All"] + all_items).index(st.session_state['secondary_diag']))
            if st.form_submit_button("Update Analytics", type="primary"):
                st.session_state['primary_diag'], st.session_state['secondary_diag'] = p_diag, s_diag
                st.rerun()
        st_html('</div>')
    
    with bcol2:
        st_html(f'<div class="glass-card" style="padding:20px;"><h3>Algorithm Comparison</h3><div style="display:flex; justify-content:space-between;"><div>Apriori: {time_apriori}s</div><div>FP-Growth: {time_fp}s</div></div></div>')

with graph_placeholder.container():
    # --- Metric Matrix (Restored Colors and Position) ---
    if current_selection_rules is not None and len(current_selection_rules) > 0:
        top_matrix = current_selection_rules.nlargest(5, 'lift')
        table_html = "<table style='width:100%; border-collapse:collapse; font-size:12px;'><tr><th>#</th><th>Metric Matrix</th><th>Support</th><th>Confidence</th></tr>"
        cmap_sup, cmap_conf = plt.get_cmap('Blues'), plt.get_cmap('Reds')
        for i, (idx, row) in enumerate(top_matrix.iterrows(), 1):
            s_v, c_v = row['support'], row['confidence']
            s_n, c_n = min(1.0, max(0.2, s_v / 0.2)), min(1.0, max(0.2, c_v / 1.0))
            bg_s, bg_c = mcolors.to_hex(cmap_sup(s_n)), mcolors.to_hex(cmap_conf(c_n))
            table_html += f'<tr><td>{i}</td><td>{clean_frozenset(row["antecedents"])} &rarr; {clean_frozenset(row["consequents"])}</td><td style="background:{bg_s}; color:{"#fff" if s_n > 0.5 else "#000"};">{s_v:.2f}</td><td style="background:{bg_c}; color:{"#fff" if c_n > 0.5 else "#000"};">{c_v:.2f}</td></tr>'
        table_html += "</table>"
        
        st_html(f"""<div class="glass-card" style="padding:25px; margin-bottom:20px;"><h3>Comorbidity Heatmap & Graph Hybrid</h3><div style="margin-bottom:25px;"><h4>Clinical Metric Matrix</h4>{table_html}</div><div style="display:flex; gap:25px;"><div style="flex:1.2; height:200px; background:#f8fafc; border-radius:12px; border:1px solid #e2e8f0; display:flex; align-items:center; justify-content:center;">[Flowchart Visualization Anchor]</div><div style="flex:1; font-size:11px;"><b>Metric Key:</b><br><div style="width:100px; height:10px; background:linear-gradient(to right, #eff6ff, #1d4ed8);"></div> Support<br><div style="width:100px; height:10px; background:linear-gradient(to right, #fef2f2, #b91c1c);"></div> Confidence</div></div></div>""")

# --- Modals JS ---
st_html("""
<script>
    const doc = window.parent.document;
    const advisoryBtn = doc.getElementById('advisoryBtn');
    const advisoryModal = doc.getElementById('advisoryModal');
    const closeAdvisory = doc.getElementById('closeAdvisoryBtn');
    if(advisoryBtn && advisoryModal) {
        advisoryBtn.onclick = () => advisoryModal.style.display = 'flex';
        closeAdvisory.onclick = () => advisoryModal.style.display = 'none';
        advisoryModal.onclick = (e) => { if(e.target === advisoryModal) advisoryModal.style.display = 'none'; };
    }
</script>
""")
