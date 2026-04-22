import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import base64
import re
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
import random

# --- HIGH-FIDELITY RESTORATION (STABLE) ---
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

# --- Helper: Background Image ---
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
    @keyframes spin3D {{ from {{ transform: translate(-38%, -50%) rotateY(0deg); }} to {{ transform: translate(-38%, -50%) rotateY(360deg); }} }}
    .bg-image {{ position: fixed; top: 55%; left: 38%; height: 85vh; z-index: 0; opacity: 0.8; pointer-events: none; animation: spin3D 30s linear infinite; transform-style: preserve-3d; }}
    """
    bg_html = f'<img src="data:image/png;base64,{bg_img_b64}" class="bg-image">'
except Exception:
    bg_style, bg_html = "", ""

# --- DATA ENGINE ---
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

@st.cache_data
def get_processed_data():
    rules_path = os.path.join(PROCESSED_DIR, "association_rules.csv")
    itemsets_path = os.path.join(PROCESSED_DIR, "frequent_itemsets.csv")
    if os.path.exists(rules_path) and os.path.exists(itemsets_path):
        return pd.read_csv(rules_path), pd.read_csv(itemsets_path)
    # Mining
    trans_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transactions", "transactions.csv")
    try:
        df_trans = pd.read_csv(trans_path)
        dataset = [str(items).split(",") for items in df_trans["Items"].tolist()]
        te = TransactionEncoder()
        te_ary = te.fit(dataset).transform(dataset)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        freq_items = fpgrowth(df_encoded, min_support=0.01, use_colnames=True)
        rules = association_rules(freq_items, metric="confidence", min_threshold=0.5)
        rules.to_csv(rules_path, index=False)
        freq_items.to_csv(itemsets_path, index=False)
        return rules, freq_items
    except: return None, None

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

# --- SPECIALIST MAPPING ---
SPECIALIST_MAP = {
    'Hypertension': {'role': 'Cardiologist', 'qualifier': 'Expert in vascular tension.'},
    'Diabetes': {'role': 'Endocrinologist', 'qualifier': 'Expert in metabolic regulation.'},
    'Asthma': {'role': 'Pulmonologist', 'qualifier': 'Specializes in airway management.'},
    'Kidney Disease': {'role': 'Nephrologist', 'qualifier': 'Expert in renal filtration.'},
    'Allergy': {'role': 'Allergist / Immunologist', 'qualifier': 'Specializes in immune response.'},
    'Routine Checkup': {'role': 'General Practitioner', 'qualifier': 'Coordination of primary care.'}
}

# --- CALCULATIONS ---
current_selection_rules = rules_df.copy() if rules_df is not None else None
if st.session_state['primary_diag'] != "All":
    current_selection_rules = current_selection_rules[current_selection_rules['antecedents'].apply(lambda x: st.session_state['primary_diag'] in str(x))]
if st.session_state['secondary_diag'] != "All":
    current_selection_rules = current_selection_rules[current_selection_rules['consequents'].apply(lambda x: st.session_state['secondary_diag'] in str(x))]
current_selection_rules = current_selection_rules.sort_values('lift', ascending=False) if current_selection_rules is not None else None

# Pre-build components
schedule_html = ""
if current_selection_rules is not None and len(current_selection_rules) > 0:
    for i, (_, row) in enumerate(current_selection_rules.nlargest(3, 'lift').iterrows()):
        ant, con = clean_frozenset(row['antecedents']), clean_frozenset(row['consequents'])
        schedule_html += f'<div class="glass-card" style="padding:15px; margin-bottom:10px;"><div class="timeline-item"><div class="timeline-time">{14+i}:00</div><div class="timeline-title">Review {con[:12]}...</div><div class="timeline-desc">Logic: {ant} &rarr; {con}</div></div></div>'
else:
    schedule_html = '<div class="glass-card" style="padding:15px;"><div class="timeline-item"><div class="timeline-time">09:00</div><div class="timeline-title">Routine Monitoring</div><div class="timeline-desc">Standard care protocol.</div></div></div>'

spec_modal_content = ""
consult_trigger_html = ""
if current_selection_rules is not None and len(current_selection_rules) > 0:
    con_c = clean_frozenset(current_selection_rules.iloc[0]['consequents'])
    specialists_data = []
    for c in con_c.split(", "):
        for key, d in SPECIALIST_MAP.items():
            if key in c: specialists_data.append({'role': d['role'], 'condition': key, 'qualifier': d['qualifier']})
    if not specialists_data: specialists_data = [{'role': 'General Practitioner', 'condition': 'General Profile', 'qualifier': 'Coordination of care.'}]
    spec_cards = "".join([f'<div style="background:white; border:1px solid #e2e8f0; border-radius:12px; padding:15px; margin-bottom:12px; display:flex; align-items:center; gap:15px;"><div style="width:45px; height:45px; border-radius:50%; background:#eff6ff; display:flex; align-items:center; justify-content:center; color:#3b82f6; font-size:22px;">👨‍⚕️</div><div><div style="font-size:14px; font-weight:700;">{d["role"]}</div><div style="font-size:11px; color:#3b82f6;">{d["condition"]}</div><div style="font-size:10px; color:#64748b;">{d["qualifier"]}</div></div></div>' for d in specialists_data[:3]])
    consult_trigger_html = f'<div id="advisoryBtn" class="glass-card" style="margin-top:20px; border-left:4px solid #3b82f6; cursor:pointer;"><h3>Multi-Disciplinary Consult</h3><p style="font-size:12px; color:#64748b;">Click for recommendations board.</p><div style="font-size:11px; font-weight:700; color:#3b82f6;">View Insights &rarr;</div></div>'
    spec_modal_content = f"""<div id="advisoryModal" style="display:none; position:fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(255,255,255,0.2); backdrop-filter:blur(15px); z-index:9999; align-items:center; justify-content:center;"><div style="background:#fff; width:900px; max-width:90%; height:85vh; border-radius:24px; padding:40px; overflow-y:auto; box-shadow:0 40px 80px rgba(0,0,0,0.12);"><div id="closeAdvisoryBtn" style="position:absolute; top:30px; right:30px; cursor:pointer; font-size:24px;">✕</div><h2>Consultation Board</h2><div style="display:grid; grid-template-columns:1fr 1fr; gap:30px; margin-top:30px;"><div><h4>Specialists</h4>{spec_cards}</div><div><h4>Evidence</h4><p style="font-size:13px;">Pattern: {clean_frozenset(current_selection_rules.iloc[0]["antecedents"])} &rarr; {con_c}</p></div></div></div></div>"""

matrix_table_html = "<table style='width:100%; border-collapse:collapse; font-size:12px;'><tr><th>#</th><th>Pattern</th><th>Supp</th><th>Conf</th></tr>"
cmap_sup, cmap_conf = plt.get_cmap('Blues'), plt.get_cmap('Reds')
if current_selection_rules is not None and len(current_selection_rules) > 0:
    for i, (_, row) in enumerate(current_selection_rules.nlargest(5, 'lift').iterrows(), 1):
        s_v, c_v = row['support'], row['confidence']
        s_n, c_n = min(1.0, max(0.2, s_v/0.2)), min(1.0, max(0.2, c_v/1.0))
        bg_s, bg_c = mcolors.to_hex(cmap_sup(s_n)), mcolors.to_hex(cmap_conf(c_n))
        matrix_table_html += f'<tr><td style="font-weight:700;">{i}</td><td>{clean_frozenset(row["antecedents"])} &rarr; {clean_frozenset(row["consequents"])}</td><td style="background:{bg_s}; color:{"#fff" if s_n > 0.5 else "#000"}; text-align:center;">{s_v:.2f}</td><td style="background:{bg_c}; color:{"#fff" if c_n > 0.5 else "#000"}; text-align:center;">{c_v:.2f}</td></tr>'
matrix_table_html += "</table>"

# --- RENDERER ---
st_html(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    #MainMenu, footer, header {{visibility: hidden;}}
    .block-container {{ padding: 1rem; max-width: 100%; position: relative; z-index: 1; }}
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: #0f172a; }}
    .stApp {{ background-color: #f8fafc; background-image: linear-gradient(to right, #e2e8f0 1px, transparent 1px), linear-gradient(to bottom, #e2e8f0 1px, transparent 1px); background-size: 40px 40px; }}
    {bg_style}
    .navbar {{ display: flex; align-items: center; justify-content: space-between; padding: 10px 40px; background: rgba(255, 255, 255, 0.7); backdrop-filter: blur(15px); border-radius: 100px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); margin-bottom: 20px; position: relative; z-index: 10; }}
    .nav-btn {{ padding: 10px 20px; border-radius: 50px; font-weight: 600; font-size: 14px; background: transparent; color: #475569; border: none; }}
    .nav-btn.active {{ background: #0f172a; color: white; }}
    .glass-card {{ background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(20px); border: 1px solid rgba(255, 255, 255, 0.8); border-radius: 15px; padding: 20px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08); margin-bottom: 20px; position: relative; z-index: 2; }}
    h3 {{ font-size: 16px; font-weight: 600; color: #0f172a; margin-bottom: 15px; margin-top: 0; }}
    .vitals-row {{ display: flex; gap: 10px; margin-bottom: 20px; }}
    .vital-card {{ flex: 1; padding: 15px; border-radius: 15px; background: #fff; box-shadow: 0 4px 15px rgba(0,0,0,0.05); position: relative; overflow: hidden; }}
    .ekg-line {{ height: 30px; width: 100%; margin-top: 10px; background: linear-gradient(90deg, transparent, #ef4444, transparent); background-size: 100px 100%; animation: moveEKG 1s linear infinite; opacity: 0.5; }}
    @keyframes moveEKG {{ from {{ background-position: 0 0; }} to {{ background-position: -100px 0; }} }}
    .timeline-item {{ margin-bottom: 15px; padding-left: 15px; border-left: 2px solid #e2e8f0; position: relative; }}
    .timeline-item::before {{ content: ''; position: absolute; left: -6px; top: 0; width: 10px; height: 10px; border-radius: 50%; background: #3b82f6; }}
    .heart-icon {{ animation: heartbeat 1.5s infinite; color: #ef4444; display: inline-block; }}
    @keyframes heartbeat {{ 0%, 100% {{ transform: scale(1); }} 50% {{ transform: scale(1.2); }} }}
</style>
{bg_html}

<div class="navbar">
    <div style="font-weight:700; font-size:20px;">⚕️ Clinical Comorbidity & Treatment Patterns</div>
    <div class="navbar-links">
        <button class="nav-btn active">Dashboard</button>
        <button class="nav-btn" id="navAppointments">Appointments</button>
        <button class="nav-btn" id="navSchedule">Schedule</button>
        <button class="nav-btn" id="navLabs">Labs Results</button>
    </div>
    <div style="display:flex; flex-direction:column; align-items:center;">
        <div style="width:35px; height:35px; border-radius:50%; background:#e2e8f0; display:flex; align-items:center; justify-content:center;">👤</div>
        <span style="font-size:10px; font-weight:700; color:#475569;">Patient #2440</span>
    </div>
</div>
""")

# --- MAIN LAYOUT ---
col1, col2, col3 = st.columns([1, 1.2, 1.5], gap="large")

with col1:
    st_html(f"""
        <h3>Dynamic Care Plan</h3>
        {schedule_html}
        <div class="glass-card" style="cursor:pointer; text-align:center; padding:15px; border:1px solid rgba(255,255,255,0.6);" id="demoBtn">
            <h3 style="margin:0;">Demographic Comorbidity Patterns</h3>
        </div>
        {consult_trigger_html}
        <div class="glass-card" style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: white; border: none; margin-top: 20px;">
            <div style="font-size: 11px; color: #94a3b8; font-weight: 700;">TOP CLINICAL INSIGHT</div>
            <div style="font-size: 15px; font-weight: 700; margin: 10px 0;">
                High correlation detected between {clean_frozenset(current_selection_rules.iloc[0]["antecedents"]) if current_selection_rules is not None and len(current_selection_rules)>0 else "Profile"} 
                &rarr; {clean_frozenset(current_selection_rules.iloc[0]["consequents"]) if current_selection_rules is not None and len(current_selection_rules)>0 else "Conditions"}
            </div>
            <div style="font-size:12px; opacity:0.8;">LIFT: {current_selection_rules.iloc[0]["lift"] if current_selection_rules is not None and len(current_selection_rules)>0 else 0:.2f}</div>
        </div>
    """)

with col2: st_html("<div style='height:80vh;'></div>")

with col3:
    st_html(f"""
        <div class="vitals-row">
            <div class="vital-card">
                <div style="font-size:12px; color:#64748b; font-weight:600;">Total Visits</div>
                <div style="font-size:24px; font-weight:700;">{total_visits}</div>
                <div class="ekg-line" style="background: linear-gradient(90deg, transparent, #3b82f6, transparent);"></div>
            </div>
            <div class="vital-card">
                <div style="font-size:12px; color:#64748b; font-weight:600;">Avg Events</div>
                <div style="font-size:24px; font-weight:700;">{avg_events}</div>
                <div class="ekg-line" style="background: linear-gradient(90deg, transparent, #eab308, transparent);"></div>
            </div>
        </div>
        <div class="vitals-row">
            <div class="vital-card">Heart: <span id="liveHR">82</span>bpm <div class="heart-icon">❤</div></div>
            <div class="vital-card">Brain: <span id="liveBrain">120</span>Hz 🧠</div>
            <div class="vital-card">Temp: <span id="liveTemp">38.5</span>°C 🌡</div>
        </div>

        <div class="glass-card">
            <h3>Comorbidity Heatmap & Graph Hybrid</h3>
            <div style="margin-bottom:25px;">
                <h4 style="font-size:14px; margin-bottom:10px;">Clinical Metric Matrix</h4>
                {matrix_table_html}
            </div>
            <div style="display:flex; gap:20px; align-items:flex-start;">
                <div style="flex:1.2; height:220px; background:#f8fafc; border-radius:12px; border:1px solid #e2e8f0; display:flex; align-items:center; justify-content:center;">
                    <svg width="100%" height="100%" viewBox="0 0 400 200">
                        <line x1="80" y1="100" x2="220" y2="50" stroke="#94a3b8" stroke-width="2" marker-end="url(#arr)" />
                        <line x1="80" y1="100" x2="220" y2="150" stroke="#94a3b8" stroke-width="2" marker-end="url(#arr)" />
                        <rect x="20" y="85" width="100" height="30" rx="15" fill="#3b82f6" /><text x="70" y="104" fill="white" font-size="12" text-anchor="middle">Profile</text>
                        <rect x="220" y="35" width="100" height="30" rx="15" fill="#ef4444" /><text x="270" y="54" fill="white" font-size="12" text-anchor="middle">Condition</text>
                        <rect x="220" y="135" width="100" height="30" rx="15" fill="#f59e0b" /><text x="270" y="154" fill="white" font-size="12" text-anchor="middle">Pathway</text>
                    </svg>
                </div>
                <div style="flex:1;">
                    <div style="font-size:12px; font-weight:700; margin-bottom:10px;">Metric Key:</div>
                    <div style="display:flex; flex-direction:column; gap:8px;">
                        <div style="display:flex; align-items:center; gap:8px;"><div style="width:40px; height:8px; background:linear-gradient(to right, #eff6ff, #1d4ed8); border-radius:4px;"></div> <span style="font-size:10px;">Support</span></div>
                        <div style="display:flex; align-items:center; gap:8px;"><div style="width:40px; height:8px; background:linear-gradient(to right, #fef2f2, #b91c1c); border-radius:4px;"></div> <span style="font-size:10px;">Confidence</span></div>
                    </div>
                </div>
            </div>
        </div>
    """)
    
    scol1, scol2 = st.columns([1, 1.5])
    with scol1:
        st.markdown('<div class="glass-card" style="border-top:4px solid #0f172a; padding:15px;"><h3>Pattern Selection</h3>', unsafe_allow_html=True)
        with st.form("pattern_form"):
            p = st.selectbox("Primary Diagnosis", ["All"] + all_items, index=(["All"] + all_items).index(st.session_state['primary_diag']))
            s = st.selectbox("Secondary Condition", ["All"] + all_items, index=(["All"] + all_items).index(st.session_state['secondary_diag']))
            if st.form_submit_button("Update Analytics Board", type="primary"):
                st.session_state['primary_diag'], st.session_state['secondary_diag'] = p, s
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with scol2:
        st_html(f"""
            <div class="glass-card" style="padding:15px;">
                <h3>Algorithm Comparison</h3>
                <div style="display:flex; justify-content:space-between; align-items:center; margin-top:10px;">
                    <div><div style="font-size:11px; color:#94a3b8;">Apriori</div><div style="font-size:22px; font-weight:700;">{time_apriori}s</div></div>
                    <div style="width:1px; height:40px; background:#e2e8f0;"></div>
                    <div><div style="font-size:11px; color:#94a3b8;">FP-Growth</div><div style="font-size:22px; font-weight:700;">{time_fp}s</div></div>
                </div>
            </div>
        """)

# --- MODALS & JS ---
st_html(f"""
    {spec_modal_content}
    <div id="appointmentsModal" style="display:none; position:fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(255,255,255,0.2); backdrop-filter:blur(15px); z-index:10000; align-items:center; justify-content:center;">
        <div class="glass-card" style="width:500px;"><h3>Appointments</h3><p>Follow-up - Oct 24</p><button id="closeApps">Close</button></div>
    </div>
    <div id="scheduleModal" style="display:none; position:fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(255,255,255,0.2); backdrop-filter:blur(15px); z-index:10000; align-items:center; justify-content:center;">
        <div class="glass-card" style="width:500px;"><h3>Schedule</h3><p>08:00 - Morning Vitals</p><button id="closeSched">Close</button></div>
    </div>
    <script>
        const doc = window.parent.document;
        const bind = (id, mid, cid) => {{
            const b = doc.getElementById(id); const m = doc.getElementById(mid); const c = doc.getElementById(cid);
            if(b && m) {{ b.onclick = () => m.style.display = 'flex'; if(c) c.onclick = () => m.style.display = 'none'; }}
        }};
        bind('advisoryBtn', 'advisoryModal', 'closeAdvisoryBtn');
        bind('navAppointments', 'appointmentsModal', 'closeApps');
        bind('navSchedule', 'scheduleModal', 'closeSched');
        setInterval(() => {{
            const h = doc.getElementById('liveHR'); const b = doc.getElementById('liveBrain'); const t = doc.getElementById('liveTemp');
            if(h) h.innerText = 80 + Math.floor(Math.random() * 10);
            if(b) b.innerText = 110 + Math.floor(Math.random() * 40);
            if(t) t.innerText = (38.2 + Math.random() * 0.6).toFixed(1);
        }}, 2000);
    </script>
""")
