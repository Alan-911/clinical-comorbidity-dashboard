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

# --- Atomic Page Config ---
st.set_page_config(
    page_title="Clinical Comorbidity Dashboard",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Zero-Flicker Renderer ---
def st_html(html_str):
    """Flattens HTML to a single line to prevent Streamlit from parsing indented lines as Markdown code blocks."""
    flat_html = re.sub(r'\n\s*', ' ', html_str)
    st.markdown(flat_html, unsafe_allow_html=True)

# --- State Management ---
if 'primary_diag' not in st.session_state: st.session_state['primary_diag'] = "All"
if 'secondary_diag' not in st.session_state: st.session_state['secondary_diag'] = "All"

# --- Helper: Data Cleaning ---
def clean_fs(x):
    cleaned = re.sub(r"(frozenset|set|[{}()\[\]'\"])", "", str(x))
    return cleaned.replace(",", ", ").strip()

# --- Data Engine (Cached) ---
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

@st.cache_data
def get_dashboard_data():
    rules_path = os.path.join(PROCESSED_DIR, "association_rules.csv")
    if os.path.exists(rules_path):
        rules = pd.read_csv(rules_path)
    else:
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
        except: return None
    
    all_items = set()
    for col in ['antecedents', 'consequents']:
        for val in rules[col]:
            items = clean_fs(val).split(",")
            all_items.update([i.strip() for i in items if i.strip()])
    return rules, sorted(list(all_items))

data = get_dashboard_data()
if not data: st.stop()
rules_df, all_items = data

# --- Logic: Filter Rules ---
filtered_rules = rules_df.copy()
if st.session_state['primary_diag'] != "All":
    filtered_rules = filtered_rules[filtered_rules['antecedents'].apply(lambda x: st.session_state['primary_diag'] in str(x))]
if st.session_state['secondary_diag'] != "All":
    filtered_rules = filtered_rules[filtered_rules['consequents'].apply(lambda x: st.session_state['secondary_diag'] in str(x))]
filtered_rules = filtered_rules.sort_values('lift', ascending=False)

# --- Component Builder: Timeline ---
schedule_html = ""
if len(filtered_rules) > 0:
    for i, (_, row) in enumerate(filtered_rules.nlargest(3, 'lift').iterrows()):
        ant, con = clean_fs(row['antecedents']), clean_fs(row['consequents'])
        schedule_html += f"""
        <div class="glass-card" style="padding:15px; margin-bottom:10px;">
            <div class="timeline-item">
                <div class="timeline-time">{14+i}:00</div>
                <div class="timeline-title">Review {con[:12]}...</div>
                <div class="timeline-desc">Context: {ant} &rarr; {con} (Lift: {row['lift']:.1f})</div>
            </div>
        </div>
        """
else:
    schedule_html = '<div class="glass-card" style="padding:15px;">Standard Clinical Monitoring</div>'

# --- Component Builder: Metric Matrix ---
matrix_html = "<table style='width:100%; border-collapse:collapse; font-size:12px;'>"
matrix_html += "<tr><th style='text-align:left;'>#</th><th>Logic</th><th>Supp</th><th>Conf</th></tr>"
cmap_sup, cmap_conf = plt.get_cmap('Blues'), plt.get_cmap('Reds')
if len(filtered_rules) > 0:
    for i, (_, row) in enumerate(filtered_rules.nlargest(5, 'lift').iterrows(), 1):
        s_v, c_v = row['support'], row['confidence']
        s_n, c_n = min(1.0, max(0.2, s_v/0.2)), min(1.0, max(0.2, c_v/1.0))
        bg_s, bg_c = mcolors.to_hex(cmap_sup(s_n)), mcolors.to_hex(cmap_conf(c_n))
        matrix_html += f'<tr><td>{i}</td><td>{clean_fs(row["antecedents"])} &rarr; {clean_fs(row["consequents"])}</td><td style="background:{bg_s}; color:{"#fff" if s_n > 0.5 else "#000"}; text-align:center;">{s_v:.2f}</td><td style="background:{bg_c}; color:{"#fff" if c_n > 0.5 else "#000"}; text-align:center;">{c_v:.2f}</td></tr>'
else:
    matrix_html += "<tr><td colspan='4' style='text-align:center;'>No patterns selected</td></tr>"
matrix_html += "</table>"

# --- Specialist Data ---
specialists_html = ""
if len(filtered_rules) > 0:
    con_c = clean_fs(filtered_rules.iloc[0]['consequents'])
    for c in con_c.split(", ")[:3]:
        specialists_html += f'<div style="background:white; border:1px solid #e2e8f0; border-radius:12px; padding:12px; margin-bottom:10px; display:flex; align-items:center; gap:12px;"><div style="width:35px; height:35px; border-radius:50%; background:#eff6ff; display:flex; align-items:center; justify-content:center;">👨‍⚕️</div><div><div style="font-size:12px; font-weight:700;">{c} Expert</div><div style="font-size:10px; color:#64748b;">Specialist consult advised.</div></div></div>'
else:
    specialists_html = '<p style="font-size:12px; color:#64748b;">Select a profile to view specialists.</p>'

# --- Encode Background ---
base_path = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(base_path, "visualizations", "anatomical_model.png"), 'rb') as f:
        bg_img = base64.b64encode(f.read()).decode()
    bg_html = f'<img src="data:image/png;base64,{bg_img}" style="position:fixed; top:55%; left:38%; height:85vh; z-index:0; opacity:0.8; pointer-events:none; animation:spin3D 30s linear infinite; transform-style:preserve-3d;">'
except: bg_html = ""

# --- 🚀 THE ATOMIC MASTER RENDERER ---
st_html(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    #MainMenu, footer, header {{visibility: hidden;}}
    .block-container {{ padding: 0.5rem; max-width: 100%; position: relative; z-index: 1; }}
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: #0f172a; }}
    .stApp {{
        background-color: #f8fafc;
        background-image: linear-gradient(to right, #e2e8f0 1px, transparent 1px), linear-gradient(to bottom, #e2e8f0 1px, transparent 1px);
        background-size: 40px 40px;
    }}
    @keyframes spin3D {{ from {{ transform: translate(-38%, -50%) rotateY(0deg); }} to {{ transform: translate(-38%, -50%) rotateY(360deg); }} }}
    .navbar {{
        display: flex; align-items: center; justify-content: space-between; padding: 10px 40px;
        background: rgba(255, 255, 255, 0.7); backdrop-filter: blur(15px); border-radius: 100px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05); margin-bottom: 20px;
    }}
    .glass-card {{
        background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(20px); border: 1px solid rgba(255, 255, 255, 0.8);
        border-radius: 15px; padding: 20px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08); margin-bottom: 20px;
    }}
    .timeline-item {{ margin-bottom: 15px; padding-left: 15px; border-left: 2px solid #e2e8f0; position: relative; }}
    .timeline-item::before {{ content: ''; position: absolute; left: -6px; top: 0; width: 10px; height: 10px; border-radius: 50%; background: #3b82f6; }}
    .timeline-time {{ font-size: 11px; color: #94a3b8; font-weight: 700; }}
    .timeline-title {{ font-size: 13px; font-weight: 700; color: #0f172a; }}
    .timeline-desc {{ font-size: 11px; color: #64748b; }}
    .vitals-row {{ display: flex; gap: 10px; margin-bottom: 15px; }}
    .vital-card {{ flex: 1; padding: 12px; border-radius: 12px; background: #fff; box-shadow: 0 4px 10px rgba(0,0,0,0.03); }}
    .ekg-line {{ height: 2px; width: 100%; margin-top: 10px; background: linear-gradient(90deg, transparent, #ef4444, transparent); background-size: 100px 100%; animation: moveEKG 1s linear infinite; }}
    @keyframes moveEKG {{ from {{ background-position: 0 0; }} to {{ background-position: -100px 0; }} }}
    @keyframes heartbeat {{ 0%, 100% {{ transform: scale(1); }} 50% {{ transform: scale(1.2); }} }}
    .heart {{ animation: heartbeat 1s infinite; color: #ef4444; display: inline-block; }}
    </style>
    {bg_html}
    
    <div class="navbar">
        <div style="font-weight:700; font-size:18px;">⚕️ Comorbidity & Treatment Patterns</div>
        <div style="display:flex; gap:20px; font-size:13px; font-weight:600; color:#64748b;">
            <span>Dashboard</span><span>Appointments</span><span>Schedule</span><span>Labs</span>
        </div>
        <div style="text-align:right;"><div style="font-size:10px; font-weight:800;">PATIENT #2440</div><div style="font-size:9px; color:#3b82f6;">CONNECTED</div></div>
    </div>

    <div style="display: grid; grid-template-columns: 1fr 1.2fr 1.5fr; gap: 25px;">
        <!-- Left: Care Plan -->
        <div>
            <h3>Dynamic Care Plan</h3>
            {schedule_html}
            <div class="glass-card" style="border-left:4px solid #3b82f6; cursor:pointer;" id="advisoryTrigger">
                <div style="font-weight:700; font-size:14px;">Multi-Disciplinary Consult</div>
                <div style="font-size:11px; color:#64748b;">Click to view clinical board.</div>
            </div>
        </div>

        <!-- Middle: 3D Buffer -->
        <div style="height: 1vh;"></div>

        <!-- Right: Analytics -->
        <div>
            <div class="vitals-row">
                <div class="vital-card"><div style="font-size:10px; color:#94a3b8;">Visits</div><div style="font-size:20px; font-weight:700;">{total_visits}</div></div>
                <div class="vital-card"><div style="font-size:10px; color:#94a3b8;">Avg Events</div><div style="font-size:20px; font-weight:700;">{avg_events}</div></div>
            </div>
            <div class="vitals-row">
                <div class="vital-card"><span class="heart">❤</span> <span id="liveHR">82</span> bpm</div>
                <div class="vital-card">🧠 <span id="liveBrain">120</span> Hz</div>
                <div class="vital-card">🌡 <span id="liveTemp">38.5</span> °C</div>
            </div>

            <div class="glass-card">
                <h3>Comorbidity Heatmap & Graph Hybrid</h3>
                <div style="margin-bottom:20px;">{matrix_html}</div>
                <div style="display:flex; gap:15px;">
                    <div style="flex:1.2; height:150px; background:#f8fafc; border-radius:12px; border:1px solid #e2e8f0; display:flex; align-items:center; justify-content:center;">
                        <svg width="100%" height="100%" viewBox="0 0 200 100">
                            <line x1="40" y1="50" x2="140" y2="20" stroke="#94a3b8" stroke-width="1" />
                            <line x1="40" y1="50" x2="140" y2="80" stroke="#94a3b8" stroke-width="1" />
                            <circle cx="40" cy="50" r="10" fill="#3b82f6" /><circle cx="140" cy="20" r="10" fill="#ef4444" /><circle cx="140" cy="80" r="10" fill="#f59e0b" />
                        </svg>
                    </div>
                    <div style="flex:1; font-size:10px; color:#64748b;">
                        <b>Heatmap Key:</b><br>
                        <div style="width:60px; height:6px; background:linear-gradient(to right, #eff6ff, #1d4ed8); border-radius:3px; margin:4px 0;"></div> Support<br>
                        <div style="width:60px; height:6px; background:linear-gradient(to right, #fef2f2, #b91c1c); border-radius:3px; margin:4px 0;"></div> Confidence
                    </div>
                </div>
            </div>
        </div>
    </div>
""")

# --- Selection Form (Atomic after main render) ---
with st.container():
    st_html('<div style="display:flex; gap:20px; margin-top:0px;">')
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st_html('<div class="glass-card" style="margin-top:0;"><h3>Pattern Selection</h3>')
        with st.form("pattern_form"):
            p = st.selectbox("Primary", ["All"] + all_items, index=(["All"] + all_items).index(st.session_state['primary_diag']))
            s = st.selectbox("Secondary", ["All"] + all_items, index=(["All"] + all_items).index(st.session_state['secondary_diag']))
            if st.form_submit_button("Update Analytics", type="primary"):
                st.session_state['primary_diag'], st.session_state['secondary_diag'] = p, s
                st.rerun()
        st_html('</div>')
    with c2:
        st_html(f"""
            <div class="glass-card" style="margin-top:0;">
                <h3>Algorithm Comparison</h3>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div><div style="font-size:11px; color:#94a3b8;">Apriori</div><div style="font-size:22px; font-weight:700;">{time_apriori}s</div></div>
                    <div style="width:1px; height:40px; background:#e2e8f0;"></div>
                    <div><div style="font-size:11px; color:#94a3b8;">FP-Growth</div><div style="font-size:22px; font-weight:700;">{time_fp}s</div></div>
                </div>
                <div style="margin-top:15px; font-size:10px; color:#3b82f6; font-weight:700;">PROCESSED IN {time_fp*1000:.0f}ms • ZERO-FLICKER RENDERING</div>
            </div>
        """)
    st_html('</div>')

# --- Modals & JS ---
st_html(f"""
    <div id="advisoryModal" style="display:none; position:fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(255,255,255,0.2); backdrop-filter:blur(15px); z-index:9999; align-items:center; justify-content:center;">
        <div style="background:#fff; width:900px; max-width:90%; border-radius:24px; padding:40px; box-shadow:0 40px 80px rgba(0,0,0,0.15); position:relative;">
            <div id="closeModal" style="position:absolute; top:20px; right:20px; cursor:pointer; font-size:24px;">✕</div>
            <h2>Multi-Disciplinary Board</h2>
            <p>Clinical specialist mapping for: {st.session_state['primary_diag']}</p>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-top:20px;">
                <div style="background:#f8fafc; padding:20px; border-radius:16px; border:1px solid #e2e8f0;">
                    <h4 style="margin-top:0;">Clinical Strength</h4>
                    <div>Max Lift: {filtered_rules.iloc[0]['lift'] if len(filtered_rules)>0 else 0:.2f}</div>
                    <div>Confidence: {filtered_rules.iloc[0]['confidence'] if len(filtered_rules)>0 else 0:.2f}</div>
                </div>
                <div>{specialists_html}</div>
            </div>
        </div>
    </div>
    <script>
        const doc = window.parent.document;
        const trig = doc.getElementById('advisoryTrigger');
        const modal = doc.getElementById('advisoryModal');
        const close = doc.getElementById('closeModal');
        if(trig && modal) {{
            trig.onclick = () => modal.style.display = 'flex';
            close.onclick = () => modal.style.display = 'none';
        }}
        setInterval(() => {{
            const h = doc.getElementById('liveHR');
            const b = doc.getElementById('liveBrain');
            const t = doc.getElementById('liveTemp');
            if(h) h.innerText = 80 + Math.floor(Math.random() * 10);
            if(b) b.innerText = 110 + Math.floor(Math.random() * 40);
            if(t) t.innerText = (38.2 + Math.random() * 0.6).toFixed(1);
        }}, 2000);
    </script>
""")
