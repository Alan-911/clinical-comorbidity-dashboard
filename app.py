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
    bg_style, bg_html = "", ""

# --- Consolidated CSS & Nav ---
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
    {bg_html}
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

# --- Data Mining Engine ---
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
        freq_items = fpgrowth(df_encoded, min_support=0.01, use_colnames=True)
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

def clean_frozenset(x):
    if not isinstance(x, str): x = str(x)
    cleaned = re.sub(r"(frozenset|set|[{}()\[\]'\"])", "", x)
    cleaned = cleaned.replace(",", ", ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.strip(",") if cleaned else "General"

rules_raw, itemsets_raw = get_processed_data()
if rules_raw is not None:
    rules_df = rules_raw.copy()
    all_items = set()
    for col in ['antecedents', 'consequents']:
        for val in rules_df[col]:
            items = clean_frozenset(val).split(",")
            all_items.update([i.strip() for i in items if i.strip()])
    all_items = sorted(list(all_items))
    total_visits, avg_events, time_apriori, time_fp = 2440, 3.2, 1.2, 0.4
else:
    rules_df, all_items = None, []
    total_visits, avg_events, time_apriori, time_fp = 0, 0, 0, 0

# --- Layout ---
col1, col2, col3 = st.columns([1, 1.2, 1.5], gap="large")

with col1:
    # --- Dynamic Care Plan ---
    current_selection_rules = rules_df.copy() if rules_df is not None else None
    if st.session_state['primary_diag'] != "All":
        current_selection_rules = current_selection_rules[current_selection_rules['antecedents'].apply(lambda x: st.session_state['primary_diag'] in str(x))]
    if st.session_state['secondary_diag'] != "All":
        current_selection_rules = current_selection_rules[current_selection_rules['consequents'].apply(lambda x: st.session_state['secondary_diag'] in str(x))]
    
    schedule_items = []
    if current_selection_rules is not None and len(current_selection_rules) > 0:
        for i, (idx, row) in enumerate(current_selection_rules.nlargest(3, 'lift').iterrows()):
            ant, con = clean_frozenset(row['antecedents']), clean_frozenset(row['consequents'])
            schedule_items.append({'time': f'{14+i}:00', 'title': f'Risk: {con[:12]}...', 'desc': f'Link: {ant} &rarr; {con}'})
    
    if not schedule_items: schedule_items = [{'time': '09:00', 'title': 'Routine Baseline', 'desc': 'Standard monitoring.'}]
    
    timeline_html = "".join([f'<div class="glass-card" style="padding:15px; margin-bottom:10px;"><div class="timeline-item"><div class="timeline-time">{item["time"]}</div><div class="timeline-title">{item["title"]}</div><div class="timeline-desc">{item["desc"]}</div></div></div>' for item in schedule_items])
    st_html(f'<div><h3>Dynamic Care Plan</h3>{timeline_html}</div>')
    
    # --- Specialist Board Trigger ---
    st_html(f"""
        <div id="advisoryBtn" class="glass-card" style="margin-top:20px; border-left:4px solid #3b82f6; cursor:pointer;">
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="font-size:18px;">🩺</span><h3 style="margin:0; font-size:15px;">Multi-Disciplinary Consult</h3>
            </div>
            <p style="font-size:12px; color:#64748b; margin-top:5px;">AI-assisted specialist recommendations board.</p>
        </div>
    """)

with col2: st_html("<div style='height: 80vh;'></div>")

with col3:
    # --- Vitals ---
    st_html(f"""
        <div class="vitals-row">
            <div class="vital-card"><div class="vital-label">Total Visits</div><div class="vital-value">{total_visits}</div><div class="ekg-line" style="background:linear-gradient(90deg, transparent, #3b82f6, transparent);"></div></div>
            <div class="vital-card"><div class="vital-label">Avg Events</div><div class="vital-value">{avg_events}</div><div class="ekg-line" style="background:linear-gradient(90deg, transparent, #eab308, transparent);"></div></div>
        </div>
        <div class="vitals-row">
            <div class="vital-card">Heart: <span id="liveHR">82</span>bpm <div class="heart-icon">❤</div></div>
            <div class="vital-card">Brain: <span id="liveBrain">120</span>Hz <div class="brain-icon">🧠</div></div>
            <div class="vital-card">Temp: <span id="liveTemp">38.5</span>°C <span style="color:#ef4444;">🌡</span></div>
        </div>
    """)
    
    # --- Main Heatmap & Graph Placeholder ---
    graph_container = st.empty()
    
    # --- Pattern Selection & Algorithm (Restored Footer Layout) ---
    bcol1, bcol2 = st.columns([1, 1.4], gap="small")
    with bcol1:
        st_html('<div class="glass-card" style="padding:20px; border-top:4px solid #0f172a;"><h3>Pattern Selection</h3>')
        with st.form("pattern_form"):
            p_diag = st.selectbox("Primary Diagnosis", ["All"] + all_items, index=(["All"] + all_items).index(st.session_state['primary_diag']))
            s_diag = st.selectbox("Secondary Condition", ["All"] + all_items, index=(["All"] + all_items).index(st.session_state['secondary_diag']))
            if st.form_submit_button("Update Analytics Board", type="primary"):
                st.session_state['primary_diag'], st.session_state['secondary_diag'] = p_diag, s_diag
                st.rerun()
        st_html('</div>')
    
    with bcol2:
        st_html(f"""
            <div class="glass-card" style="padding:20px;">
                <h3>Algorithm Comparison</h3>
                <div style="display:flex; justify-content:space-between;">
                    <div><div style="font-size:12px;">Apriori</div><div style="font-size:24px; font-weight:700;">{time_apriori}s</div></div>
                    <div style="border-left:1px solid #e2e8f0; padding-left:15px;"><div style="font-size:12px;">FP-Growth</div><div style="font-size:24px; font-weight:700;">{time_fp}s</div></div>
                </div>
            </div>
        """)

with graph_container.container():
    # --- Metric Matrix (Restored Visuals) ---
    if current_selection_rules is not None and len(current_selection_rules) > 0:
        top_matrix = current_selection_rules.nlargest(5, 'lift')
        table_html = "<table style='width:100%; border-collapse:collapse; font-size:12px;'>"
        table_html += "<tr><th style='text-align:left; border-bottom:2px solid #e2e8f0;'>#</th><th style='text-align:left; border-bottom:2px solid #e2e8f0;'>Metric Matrix</th><th style='text-align:center; border-bottom:2px solid #e2e8f0;'>Support</th><th style='text-align:center; border-bottom:2px solid #e2e8f0;'>Confidence</th></tr>"
        
        cmap_sup, cmap_conf = plt.get_cmap('Blues'), plt.get_cmap('Reds')
        for i, (idx, row) in enumerate(top_matrix.iterrows(), 1):
            s_v, c_v = row['support'], row['confidence']
            s_n, c_n = min(1.0, max(0.2, s_v / 0.2)), min(1.0, max(0.2, c_v / 1.0))
            bg_s, bg_c = mcolors.to_hex(cmap_sup(s_n)), mcolors.to_hex(cmap_conf(c_n))
            tc_s, tc_c = ("#fff" if s_n > 0.5 else "#000"), ("#fff" if c_n > 0.5 else "#000")
            table_html += f'<tr><td style="font-weight:700;">{i}</td><td>{clean_frozenset(row["antecedents"])} &rarr; {clean_frozenset(row["consequents"])}</td><td style="background:{bg_s}; color:{tc_s}; text-align:center; padding:5px;">{s_v:.2f}</td><td style="background:{bg_c}; color:{tc_c}; text-align:center; padding:5px;">{c_v:.2f}</td></tr>'
        table_html += "</table>"
        
        st_html(f"""
            <div class="glass-card" style="padding:25px;">
                <h3>Comorbidity Heatmap & Graph Hybrid</h3>
                <div style="margin-bottom:25px;"><h4>Clinical Metric Matrix</h4>{table_html}</div>
                <div style="display:flex; gap:25px; align-items:flex-start;">
                    <div style="flex:1.2; height:220px; background:#f8fafc; border-radius:12px; border:1px solid #e2e8f0; display:flex; align-items:center; justify-content:center;">
                        <svg width="100%" height="100%" viewBox="0 0 400 200">
                            <defs><marker id="arr" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><polygon points="0 0, 6 3, 0 6" fill="#94a3b8" /></marker></defs>
                            <line x1="80" y1="90" x2="220" y2="40" stroke="#94a3b8" stroke-width="2" marker-end="url(#arr)" />
                            <line x1="80" y1="90" x2="220" y2="90" stroke="#94a3b8" stroke-width="2" marker-end="url(#arr)" />
                            <rect x="20" y="75" width="80" height="30" rx="15" fill="#3b82f6" /><text x="60" y="94" fill="white" font-size="10" text-anchor="middle">Profile</text>
                            <rect x="220" y="25" width="80" height="30" rx="15" fill="#ef4444" /><text x="260" y="44" fill="white" font-size="10" text-anchor="middle">Comorbid</text>
                            <rect x="220" y="75" width="80" height="30" rx="15" fill="#f59e0b" /><text x="260" y="94" fill="white" font-size="10" text-anchor="middle">Progression</text>
                        </svg>
                    </div>
                    <div style="flex:1;">
                        <div style="font-size:12px; font-weight:700; margin-bottom:10px;">Metric Heatmap Key:</div>
                        <div style="display:flex; flex-direction:column; gap:8px;">
                            <div style="display:flex; align-items:center; gap:8px;"><div style="width:40px; height:8px; background:linear-gradient(to right, #eff6ff, #1d4ed8); border-radius:4px;"></div> <span style="font-size:10px;">Support</span></div>
                            <div style="display:flex; align-items:center; gap:8px;"><div style="width:40px; height:8px; background:linear-gradient(to right, #fef2f2, #b91c1c); border-radius:4px;"></div> <span style="font-size:10px;">Confidence</span></div>
                        </div>
                    </div>
                </div>
            </div>
        """)

# --- All Modals (Restored) ---
SPECIALIST_MAP = {
    'Hypertension': {'role': 'Cardiologist', 'qualifier': 'Expert in vascular tension.'},
    'Diabetes': {'role': 'Endocrinologist', 'qualifier': 'Expert in metabolic regulation.'},
    'Asthma': {'role': 'Pulmonologist', 'qualifier': 'Specializes in airway management.'},
}

st_html(f"""
    <div id="appointmentsModal" style="display:none; position:fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(255,255,255,0.2); backdrop-filter:blur(15px); z-index:10000; align-items:center; justify-content:center;">
        <div class="glass-card" style="width:500px;"><h3>Upcoming Appointments</h3><p>Oct 24 - Endocrinology</p><p>Oct 27 - Cardiology</p><button id="closeApps" style="margin-top:10px;">Close</button></div>
    </div>
    <div id="scheduleModal" style="display:none; position:fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(255,255,255,0.2); backdrop-filter:blur(15px); z-index:10000; align-items:center; justify-content:center;">
        <div class="glass-card" style="width:500px;"><h3>Daily Schedule</h3><p>08:00 - Morning Vitals</p><p>12:00 - Glucose Check</p><button id="closeSched" style="margin-top:10px;">Close</button></div>
    </div>
    <div id="labsModal" style="display:none; position:fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(255,255,255,0.2); backdrop-filter:blur(15px); z-index:10000; align-items:center; justify-content:center;">
        <div class="glass-card" style="width:600px;"><h3>Labs Results</h3><p>HbA1c: 6.8% (High)</p><p>LDL: 112 mg/dL (High)</p><button id="closeLabs" style="margin-top:10px;">Close</button></div>
    </div>
""")

# --- JS Logic ---
st_html("""
<script>
    const p = window.parent.document;
    const bindModal = (btnId, modalId, closeId) => {
        const btn = p.getElementById(btnId);
        const modal = p.getElementById(modalId);
        const close = p.getElementById(closeId);
        if(btn && modal) {
            btn.onclick = () => modal.style.display = 'flex';
            if(close) close.onclick = () => modal.style.display = 'none';
        }
    };
    bindModal('advisoryBtn', 'advisoryModal', 'closeAdvisoryBtn');
    bindModal('navAppointments', 'appointmentsModal', 'closeApps');
    bindModal('navSchedule', 'scheduleModal', 'closeSched');
    bindModal('navLabs', 'labsModal', 'closeLabs');
    
    // Live Vitals
    setInterval(() => {
        const hr = p.getElementById('liveHR');
        const br = p.getElementById('liveBrain');
        const te = p.getElementById('liveTemp');
        if(hr) hr.innerText = 80 + Math.floor(Math.random() * 10);
        if(br) br.innerText = 110 + Math.floor(Math.random() * 40);
        if(te) te.innerText = (38.2 + Math.random() * 0.6).toFixed(1);
    }, 2000);
</script>
""")
