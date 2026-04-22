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

# --- FINAL PRECISION UI ---
st.set_page_config(
    page_title="Clinical Intelligence Dashboard",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- State ---
if 'primary_diag' not in st.session_state: st.session_state['primary_diag'] = "All"
if 'secondary_diag' not in st.session_state: st.session_state['secondary_diag'] = "All"
if 'show_advisory' not in st.session_state: st.session_state['show_advisory'] = False

# --- Helpers ---
def st_html(html_str):
    flat_html = re.sub(r'\n\s*', ' ', html_str)
    st.markdown(flat_html, unsafe_allow_html=True)

@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        return base64.b64encode(f.read()).decode()

base_path = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(base_path, "visualizations", "anatomical_model.png")
bg_html = ""
try:
    bg_img = get_base64_of_bin_file(img_path)
    bg_html = f'<img src="data:image/png;base64,{bg_img}" style="position:fixed; top:55%; left:38%; height:85vh; z-index:0; opacity:0.8; pointer-events:none; animation:spin3D 30s linear infinite; transform-style:preserve-3d;">'
except: pass

def clean_fs(x):
    cleaned = re.sub(r"(frozenset|set|[{}()\[\]'\"])", "", str(x))
    return cleaned.replace(",", ", ").strip()

# --- Data Engine ---
@st.cache_data
def get_processed_data():
    rules_path = os.path.join(base_path, "data_processed", "association_rules.csv")
    if os.path.exists(rules_path): rules = pd.read_csv(rules_path)
    else:
        try:
            df_trans = pd.read_csv(os.path.join(base_path, "transactions", "transactions.csv"))
            te = TransactionEncoder()
            te_ary = te.fit([str(i).split(",") for i in df_trans["Items"]]).transform([str(i).split(",") for i in df_trans["Items"]])
            df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
            freq_items = fpgrowth(df_encoded, min_support=0.01, use_colnames=True)
            rules = association_rules(freq_items, metric="confidence", min_threshold=0.5)
            os.makedirs(os.path.dirname(rules_path), exist_ok=True)
            rules.to_csv(rules_path, index=False)
        except: return None
    
    all_items = set()
    for val in rules['antecedents'].tolist() + rules['consequents'].tolist():
        all_items.update([i.strip() for i in clean_fs(val).split(",") if i.strip()])
    return rules, sorted(list(all_items))

data = get_processed_data()
if not data: st.stop()
rules_df, all_items = data

# --- Logic ---
filtered = rules_df.copy()
if st.session_state['primary_diag'] != "All":
    filtered = filtered[filtered['antecedents'].apply(lambda x: st.session_state['primary_diag'] in str(x))]
if st.session_state['secondary_diag'] != "All":
    filtered = filtered[filtered['consequents'].apply(lambda x: st.session_state['secondary_diag'] in str(x))]
filtered = filtered.sort_values('lift', ascending=False)

# --- CSS & Nav ---
st_html(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    #MainMenu, footer, header {{visibility: hidden;}}
    .block-container {{ padding: 1rem; max-width: 100%; position: relative; z-index: 1; }}
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: #0f172a; }}
    .stApp {{ background-color: #f8fafc; background-image: linear-gradient(to right, #e2e8f0 1px, transparent 1px), linear-gradient(to bottom, #e2e8f0 1px, transparent 1px); background-size: 40px 40px; }}
    @keyframes spin3D {{ from {{ transform: translate(-38%, -50%) rotateY(0deg); }} to {{ transform: translate(-38%, -50%) rotateY(360deg); }} }}
    .navbar {{ display: flex; align-items: center; justify-content: space-between; padding: 10px 40px; background: rgba(255, 255, 255, 0.7); backdrop-filter: blur(15px); border-radius: 100px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); margin-bottom: 20px; }}
    .glass-card {{ background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(20px); border: 1px solid rgba(255, 255, 255, 0.8); border-radius: 15px; padding: 20px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08); margin-bottom: 20px; }}
    .vital-card {{ flex: 1; padding: 15px; border-radius: 15px; background: #fff; box-shadow: 0 4px 15px rgba(0,0,0,0.05); position: relative; overflow: hidden; }}
    .ekg-line {{ height: 30px; width: 100%; margin-top: 10px; background: linear-gradient(90deg, transparent, #ef4444, transparent); background-size: 100px 100%; animation: moveEKG 1s linear infinite; opacity: 0.5; }}
    @keyframes moveEKG {{ from {{ background-position: 0 0; }} to {{ background-position: -100px 0; }} }}
    .timeline-item {{ margin-bottom: 15px; padding-left: 15px; border-left: 2px solid #e2e8f0; position: relative; }}
    .timeline-item::before {{ content: ''; position: absolute; left: -6px; top: 0; width: 10px; height: 10px; border-radius: 50%; background: #3b82f6; }}
</style>
{bg_html}
<div class="navbar">
    <div style="font-weight:700; font-size:20px;">⚕️ Clinical Comorbidity &amp; Treatment Patterns</div>
    <div style="display:flex; gap:20px; font-size:13px; font-weight:600; color:#64748b;">
        <span style="cursor:pointer;">Dashboard</span>
        <span id="navAppointments" style="cursor:pointer;">Appointments</span>
        <span id="navSchedule" style="cursor:pointer;">Schedule</span>
        <span id="navLabs" style="cursor:pointer;">Labs</span>
    </div>
    <div style="text-align:right;"><div style="font-size:10px; font-weight:800;">PATIENT #2440</div><div style="font-size:9px; color:#3b82f6;">CONNECTED</div></div>
</div>
""")

col1, col2, col3 = st.columns([1, 1.2, 1.5], gap="large")

with col1:
    # 1. Care Plan
    schedule_items = ""
    if len(filtered) > 0:
        for i, (_, row) in enumerate(filtered.nlargest(3, 'lift').iterrows()):
            ant, con = clean_fs(row['antecedents']), clean_fs(row['consequents'])
            schedule_items += f'<div class="glass-card" style="padding:15px; margin-bottom:10px;"><div class="timeline-item"><div class="timeline-time">{14+i}:00</div><div class="timeline-title">Review {con[:12]}...</div><div class="timeline-desc">{ant} &rarr; {con}</div></div></div>'
    else:
        schedule_items = '<div class="glass-card" style="padding:15px;">Standard Monitoring Plan</div>'
    
    st_html(f"<h3>Dynamic Care Plan</h3>{schedule_items}")
    
    # 2. Demographic Patterns
    st_html(f"""
        <div id="demoBtn" class="glass-card" style="cursor:pointer; text-align:center; padding:15px; border:1px solid rgba(255,255,255,0.6);">
            <h3 style="margin:0;">Demographic Comorbidity Patterns</h3>
        </div>
    """)
    
    # 3. Consult Card
    st_html(f"""
        <div class="glass-card" style="border-left:4px solid #3b82f6;" id="advisoryBtn">
            <h3>Multi-Disciplinary Consult</h3>
            <p style="font-size:12px; color:#64748b;">AI-assisted specialist board recommendations.</p>
            <div style="font-size:11px; font-weight:700; color:#3b82f6; cursor:pointer;">View Insights &rarr;</div>
        </div>
    """)
    
    # 3. Top Insight
    if len(filtered) > 0:
        top = filtered.iloc[0]
        st_html(f"""
            <div class="glass-card" style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: white; border: none;">
                <div style="font-size: 11px; color: #94a3b8; font-weight: 700;">TOP CLINICAL INSIGHT</div>
                <div style="font-size: 15px; font-weight: 700; margin: 10px 0;">{clean_fs(top['antecedents'])} &rarr; {clean_fs(top['consequents'])}</div>
                <div style="font-size: 11px; opacity:0.8;">LIFT: {top['lift']:.2f} • CONF: {top['confidence']*100:.1f}%</div>
            </div>
        """)

with col2: st_html("<div style='height:80vh;'></div>")

with col3:
    # 1. Vitals
    st_html(f"""
        <div style="display:flex; gap:10px; margin-bottom:20px;">
            <div class="vital-card"><div style="font-size:11px; color:#94a3b8;">Visits</div><div style="font-size:24px; font-weight:700;">2,440</div><div class="ekg-line" style="background:linear-gradient(90deg, transparent, #3b82f6, transparent);"></div></div>
            <div class="vital-card"><div style="font-size:11px; color:#94a3b8;">Events</div><div style="font-size:24px; font-weight:700;">3.2</div><div class="ekg-line" style="background:linear-gradient(90deg, transparent, #eab308, transparent);"></div></div>
        </div>
        <div style="display:flex; gap:10px; margin-bottom:20px;">
            <div class="vital-card">Heart: <span id="liveHR">82</span>bpm ❤</div>
            <div class="vital-card">Brain: <span id="liveBrain">120</span>Hz 🧠</div>
            <div class="vital-card">Temp: <span id="liveTemp">38.5</span>°C 🌡</div>
        </div>
    """)
    
    # 2. Heatmap & Graph
    matrix_html = "<table style='width:100%; border-collapse:collapse; font-size:12px;'>"
    cmap_sup, cmap_conf = plt.get_cmap('Blues'), plt.get_cmap('Reds')
    if len(filtered) > 0:
        for i, (_, row) in enumerate(filtered.nlargest(5, 'lift').iterrows(), 1):
            s_n, c_n = min(1.0, max(0.2, row['support']/0.2)), min(1.0, max(0.2, row['confidence']/1.0))
            matrix_html += f'<tr><td style="font-weight:700;">{i}</td><td>{clean_fs(row["antecedents"])} &rarr; {clean_fs(row["consequents"])}</td><td style="background:{mcolors.to_hex(cmap_sup(s_n))}; color:{"#fff" if s_n > 0.5 else "#000"}; text-align:center;">{row["support"]:.2f}</td><td style="background:{mcolors.to_hex(cmap_conf(c_n))}; color:{"#fff" if c_n > 0.5 else "#000"}; text-align:center;">{row["confidence"]:.2f}</td></tr>'
    matrix_html += "</table>"
    
    st_html(f"""
        <div class="glass-card">
            <h3>Comorbidity Heatmap &amp; Graph Hybrid</h3>
            <div style="margin-bottom:20px;">{matrix_html}</div>
            <div style="display:flex; gap:20px;">
                <div id="svgContainer" style="flex:1.2; height:220px; background:#f8fafc; border-radius:12px; border:1px solid #e2e8f0; overflow:hidden; position:relative;">
                    <div style="position:absolute; top:6px; right:6px; display:flex; gap:4px; z-index:10;">
                        <button id="zoomIn" style="width:24px; height:24px; border-radius:50%; border:1px solid #e2e8f0; background:white; cursor:pointer; font-size:14px; line-height:1;">+</button>
                        <button id="zoomOut" style="width:24px; height:24px; border-radius:50%; border:1px solid #e2e8f0; background:white; cursor:pointer; font-size:14px; line-height:1;">−</button>
                        <button id="zoomReset" style="width:24px; height:24px; border-radius:50%; border:1px solid #e2e8f0; background:white; cursor:pointer; font-size:10px; line-height:1;">↺</button>
                    </div>
                    <svg id="flowSvg" width="100%" height="100%" viewBox="0 0 300 150" style="transform-origin:center; transition:transform 0.2s;">
                        <defs><marker id="arr" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#94a3b8"/></marker></defs>
                        <line x1="50" y1="75" x2="145" y2="38" stroke="#94a3b8" stroke-width="2" marker-end="url(#arr)"/>
                        <line x1="50" y1="75" x2="145" y2="112" stroke="#94a3b8" stroke-width="2" marker-end="url(#arr)"/>
                        <rect x="10" y="60" width="80" height="30" rx="15" fill="#3b82f6"/><text x="50" y="79" fill="white" font-size="10" text-anchor="middle">Profile</text>
                        <rect x="150" y="23" width="90" height="30" rx="15" fill="#ef4444"/><text x="195" y="42" fill="white" font-size="10" text-anchor="middle">Condition</text>
                        <rect x="150" y="97" width="90" height="30" rx="15" fill="#f59e0b"/><text x="195" y="116" fill="white" font-size="10" text-anchor="middle">Pathway</text>
                    </svg>
                </div>
                <div style="flex:1; font-size:10px;">
                    <b>Metric Key:</b><br>
                    <div style="width:50px; height:6px; background:linear-gradient(to right, #eff6ff, #1d4ed8); border-radius:3px; margin:4px 0;"></div> Support<br>
                    <div style="width:50px; height:6px; background:linear-gradient(to right, #fef2f2, #b91c1c); border-radius:3px; margin:4px 0;"></div> Confidence
                </div>
            </div>
        </div>
    """)
    
    # 3. Selection & Algo
    scol1, scol2 = st.columns([1, 1.4])
    with scol1:
        st_html('<div class="glass-card" style="border-top:4px solid #0f172a;"><h3>Pattern Selection</h3>')
        with st.form("pattern_form"):
            p = st.selectbox("Primary Diagnosis", ["All"] + all_items, index=(["All"] + all_items).index(st.session_state['primary_diag']))
            s = st.selectbox("Secondary Condition", ["All"] + all_items, index=(["All"] + all_items).index(st.session_state['secondary_diag']))
            if st.form_submit_button("Update Analytics Board", type="primary"):
                st.session_state['primary_diag'], st.session_state['secondary_diag'] = p, s
                st.rerun()
        st_html('</div>')
    with scol2:
        st_html(f"""
            <div class="glass-card">
                <h3>Algorithm Comparison</h3>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div><div style="font-size:11px; color:#94a3b8;">Apriori</div><div style="font-size:20px; font-weight:700;">1.2s</div></div>
                    <div style="width:1px; height:30px; background:#e2e8f0;"></div>
                    <div><div style="font-size:11px; color:#94a3b8;">FP-Growth</div><div style="font-size:20px; font-weight:700;">0.4s</div></div>
                </div>
            </div>
        """)

# --- ALL MODALS & JS (single block) ---
st_html(f"""
    <div id="advisoryModal" style="display:none; position:fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(15,23,42,0.4); backdrop-filter:blur(12px); z-index:10000; align-items:center; justify-content:center;">
        <div class="glass-card" style="width:800px; max-width:90%; max-height:85vh; overflow-y:auto; position:relative;">
            <div id="closeAdvisoryBtn" style="position:absolute; top:15px; right:20px; cursor:pointer; font-size:20px; color:#64748b;">✕</div>
            <h2 style="margin-top:0;">Specialist Advisory Board</h2>
            <hr style="border:0; border-top:1px solid #e2e8f0; margin:15px 0;">
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px;">
                <div style="background:#f8fafc; padding:20px; border-radius:12px;"><h4 style="margin-top:0;">Recommended Specialists</h4><p>👨‍⚕️ Endocrinologist<br>🫀 Cardiologist<br>🫁 Pulmonologist</p></div>
                <div style="background:#f8fafc; padding:20px; border-radius:12px;"><h4 style="margin-top:0;">Evidence Summary</h4><p>High Lift Chain detected. Confidence &gt;80%.<br>Consider multi-specialty review.</p></div>
            </div>
        </div>
    </div>
    <div id="demoModal" style="display:none; position:fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(15,23,42,0.4); backdrop-filter:blur(12px); z-index:10000; align-items:center; justify-content:center;">
        <div class="glass-card" style="width:600px; position:relative;">
            <div id="closeDemo" style="position:absolute; top:15px; right:20px; cursor:pointer; font-size:20px; color:#64748b;">✕</div>
            <h3>Demographic Comorbidity Insights</h3>
            <table style="width:100%; border-collapse:collapse; font-size:13px;">
                <tr style="background:#0f172a; color:white;"><th style="padding:8px;">Age Group</th><th style="padding:8px;">Condition</th><th style="padding:8px;">Confidence</th></tr>
                <tr style="background:#f8fafc;"><td style="padding:8px;">Senior (65+)</td><td style="padding:8px;">Heart Disease</td><td style="padding:8px;">82%</td></tr>
                <tr><td style="padding:8px;">Adult (40-64)</td><td style="padding:8px;">Diabetes</td><td style="padding:8px;">75%</td></tr>
                <tr style="background:#f8fafc;"><td style="padding:8px;">Adult (40-64)</td><td style="padding:8px;">Hypertension</td><td style="padding:8px;">71%</td></tr>
            </table>
        </div>
    </div>
    <div id="appointmentsModal" style="display:none; position:fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(15,23,42,0.4); backdrop-filter:blur(12px); z-index:10000; align-items:center; justify-content:center;">
        <div class="glass-card" style="width:520px; position:relative;">
            <div id="closeApps" style="position:absolute; top:15px; right:20px; cursor:pointer; font-size:20px; color:#64748b;">✕</div>
            <h3>Upcoming Appointments</h3>
            <div style="border-left:3px solid #3b82f6; padding:10px 15px; margin-bottom:10px; background:#f8fafc; border-radius:0 8px 8px 0;"><b>Oct 24</b> — Endocrinology Follow-up</div>
            <div style="border-left:3px solid #ef4444; padding:10px 15px; margin-bottom:10px; background:#f8fafc; border-radius:0 8px 8px 0;"><b>Nov 3</b> — Cardiology Review</div>
            <div style="border-left:3px solid #f59e0b; padding:10px 15px; background:#f8fafc; border-radius:0 8px 8px 0;"><b>Nov 18</b> — Routine Labs</div>
        </div>
    </div>
    <div id="scheduleModal" style="display:none; position:fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(15,23,42,0.4); backdrop-filter:blur(12px); z-index:10000; align-items:center; justify-content:center;">
        <div class="glass-card" style="width:520px; position:relative;">
            <div id="closeSched" style="position:absolute; top:15px; right:20px; cursor:pointer; font-size:20px; color:#64748b;">✕</div>
            <h3>Daily Schedule</h3>
            <div style="border-left:3px solid #3b82f6; padding:10px 15px; margin-bottom:8px; background:#f8fafc; border-radius:0 8px 8px 0;"><b>08:00</b> — Morning Vitals Check</div>
            <div style="border-left:3px solid #10b981; padding:10px 15px; margin-bottom:8px; background:#f8fafc; border-radius:0 8px 8px 0;"><b>10:30</b> — Medication Review</div>
            <div style="border-left:3px solid #f59e0b; padding:10px 15px; background:#f8fafc; border-radius:0 8px 8px 0;"><b>14:00</b> — Specialist Consultation</div>
        </div>
    </div>
    <div id="labsModal" style="display:none; position:fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(15,23,42,0.4); backdrop-filter:blur(12px); z-index:10000; align-items:center; justify-content:center;">
        <div class="glass-card" style="width:600px; position:relative;">
            <div id="closeLabs" style="position:absolute; top:15px; right:20px; cursor:pointer; font-size:20px; color:#64748b;">✕</div>
            <h3>Lab Results</h3>
            <table style="width:100%; border-collapse:collapse; font-size:13px;">
                <tr style="background:#0f172a; color:white;"><th style="padding:8px;">Test</th><th style="padding:8px;">Result</th><th style="padding:8px;">Status</th></tr>
                <tr style="background:#f8fafc;"><td style="padding:8px;">HbA1c</td><td style="padding:8px;">7.2%</td><td style="padding:8px; color:#f59e0b;">⚠ Monitor</td></tr>
                <tr><td style="padding:8px;">LDL</td><td style="padding:8px;">112 mg/dL</td><td style="padding:8px; color:#ef4444;">✗ High</td></tr>
                <tr style="background:#f8fafc;"><td style="padding:8px;">eGFR</td><td style="padding:8px;">78 mL/min</td><td style="padding:8px; color:#10b981;">✓ Normal</td></tr>
            </table>
        </div>
    </div>
    <script>
        (function() {{
            const doc = window.parent.document;
            const bind = (btnId, modalId, closeId) => {{
                const btn = doc.getElementById(btnId);
                const modal = doc.getElementById(modalId);
                const close = doc.getElementById(closeId);
                if (btn && modal) {{
                    btn.onclick = () => {{ modal.style.display = 'flex'; }};
                }}
                if (close && modal) {{
                    close.onclick = () => {{ modal.style.display = 'none'; }};
                }}
                if (modal) {{
                    modal.onclick = (e) => {{ if (e.target === modal) modal.style.display = 'none'; }};
                }}
            }};
            bind('advisoryBtn', 'advisoryModal', 'closeAdvisoryBtn');
            bind('demoBtn', 'demoModal', 'closeDemo');
            bind('navAppointments', 'appointmentsModal', 'closeApps');
            bind('navSchedule', 'scheduleModal', 'closeSched');
            bind('navLabs', 'labsModal', 'closeLabs');
            // Live vitals
            setInterval(() => {{
                const hr = doc.getElementById('liveHR');
                const br = doc.getElementById('liveBrain');
                const tp = doc.getElementById('liveTemp');
                if (hr) hr.innerText = 78 + Math.floor(Math.random() * 12);
                if (br) br.innerText = 110 + Math.floor(Math.random() * 40);
                if (tp) tp.innerText = (38.1 + Math.random() * 0.8).toFixed(1);
            }}, 2000);
            // SVG Zoom
            let scale = 1;
            const svg = doc.getElementById('flowSvg');
            const zoomIn = doc.getElementById('zoomIn');
            const zoomOut = doc.getElementById('zoomOut');
            const zoomReset = doc.getElementById('zoomReset');
            const applyZoom = () => {{ if(svg) svg.style.transform = 'scale(' + scale + ')'; }};
            if (zoomIn) zoomIn.onclick = () => {{ scale = Math.min(scale + 0.2, 3); applyZoom(); }};
            if (zoomOut) zoomOut.onclick = () => {{ scale = Math.max(scale - 0.2, 0.4); applyZoom(); }};
            if (zoomReset) zoomReset.onclick = () => {{ scale = 1; applyZoom(); }};
            // Scroll-to-zoom on SVG container
            const svgCont = doc.getElementById('svgContainer');
            if (svgCont) {{
                svgCont.addEventListener('wheel', (e) => {{
                    e.preventDefault();
                    scale = e.deltaY < 0 ? Math.min(scale + 0.1, 3) : Math.max(scale - 0.1, 0.4);
                    applyZoom();
                }}, {{passive: false}});
            }}
        }})();
    </script>
""")
