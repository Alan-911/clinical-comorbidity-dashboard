import streamlit as st
import pandas as pd
import os, base64, re
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

st.set_page_config(page_title="Clinical Intelligence Dashboard", page_icon="⚕️", layout="wide", initial_sidebar_state="collapsed")
if 'primary_diag' not in st.session_state: st.session_state['primary_diag'] = "All"
if 'secondary_diag' not in st.session_state: st.session_state['secondary_diag'] = "All"

def st_html(h): st.markdown(re.sub(r'\n\s*', ' ', h), unsafe_allow_html=True)

@st.cache_data
def _b64(f):
    with open(f,'rb') as fh: return base64.b64encode(fh.read()).decode()

base_path = os.path.dirname(os.path.abspath(__file__))
bg_html = ""
try:
    b64 = _b64(os.path.join(base_path,"visualizations","anatomical_model.png"))
    bg_html = f'<img src="data:image/png;base64,{b64}" style="position:fixed;top:55%;left:38%;height:85vh;z-index:0;opacity:0.8;pointer-events:none;animation:spin3D 30s linear infinite;transform-style:preserve-3d;">'
except: pass

def clean_fs(x):
    c = re.sub(r"frozenset|[{}()\[\]'\"]", "", str(x))
    return re.sub(r',\s*', ', ', re.sub(r'\s+', ' ', c)).strip().strip(",")

@st.cache_data
def get_data():
    rp = os.path.join(base_path,"data_processed","association_rules.csv")
    if os.path.exists(rp): rules = pd.read_csv(rp)
    else:
        try:
            df = pd.read_csv(os.path.join(base_path,"transactions","transactions.csv"))
            te = TransactionEncoder()
            arr = te.fit([str(i).split(",") for i in df["Items"]]).transform([str(i).split(",") for i in df["Items"]])
            fi = fpgrowth(pd.DataFrame(arr,columns=te.columns_), min_support=0.01, use_colnames=True)
            rules = association_rules(fi, metric="confidence", min_threshold=0.5)
            os.makedirs(os.path.dirname(rp), exist_ok=True)
            rules.to_csv(rp, index=False)
        except: return None
    items = set()
    for v in rules['antecedents'].tolist()+rules['consequents'].tolist():
        items.update([i.strip() for i in clean_fs(v).split(",") if i.strip()])
    return rules, sorted(items)

data = get_data()
if not data: st.stop()
rules_df, all_items = data

f = rules_df.copy()
if st.session_state['primary_diag'] != "All":
    f = f[f['antecedents'].apply(lambda x: st.session_state['primary_diag'] in str(x))]
if st.session_state['secondary_diag'] != "All":
    f = f[f['consequents'].apply(lambda x: st.session_state['secondary_diag'] in str(x))]
f = f.sort_values('lift', ascending=False)

# --- Pre-build all HTML strings ---
plan_html = ""
if len(f) > 0:
    for i, (_, row) in enumerate(f.nlargest(3,'lift').iterrows()):
        ant, con = clean_fs(row['antecedents']), clean_fs(row['consequents'])
        plan_html += f'<div class="gc" style="padding:14px;margin-bottom:10px;"><div class="ti"><div style="font-size:10px;color:#94a3b8;font-weight:700;">{14+i}:00</div><div style="font-size:13px;font-weight:700;margin:3px 0;">Review {con[:22]}{"…" if len(con)>22 else ""}</div><div style="font-size:11px;color:#64748b;">{ant[:24]}{"…" if len(ant)>24 else ""} &rarr; {con[:24]}{"…" if len(con)>24 else ""}</div></div></div>'
else:
    plan_html = '<div class="gc" style="padding:14px;"><div class="ti"><div style="font-size:13px;font-weight:700;">Standard Monitoring Plan</div></div></div>'

if len(f) > 0:
    top = f.iloc[0]
    insight = f'<div class="gc" style="background:linear-gradient(135deg,#0f172a,#1e293b);color:white;border:none;"><div style="font-size:10px;color:#94a3b8;font-weight:700;letter-spacing:1px;">TOP CLINICAL INSIGHT</div><div style="font-size:14px;font-weight:700;margin:8px 0;">{clean_fs(top["antecedents"])} &rarr; {clean_fs(top["consequents"])}</div><div style="font-size:11px;opacity:0.7;">LIFT: {top["lift"]:.2f} &bull; CONF: {top["confidence"]*100:.1f}%</div></div>'
else:
    insight = '<div class="gc" style="background:linear-gradient(135deg,#0f172a,#1e293b);color:white;border:none;"><div style="font-size:11px;color:#94a3b8;">No pattern selected</div></div>'

cmap_s, cmap_c = plt.get_cmap('Blues'), plt.get_cmap('Reds')
rows = ""
if len(f) > 0:
    for i, (_, row) in enumerate(f.nlargest(5,'lift').iterrows(), 1):
        sn = min(1.0, max(0.2, row['support']/0.2))
        cn = min(1.0, max(0.2, row['confidence']/1.0))
        bs, bc = mcolors.to_hex(cmap_s(sn)), mcolors.to_hex(cmap_c(cn))
        sc, cc = ("#fff" if sn>0.5 else "#000"), ("#fff" if cn>0.5 else "#000")
        rows += f'<tr><td style="font-weight:700;padding:5px 6px;">{i}</td><td style="padding:5px 6px;font-size:11px;">{clean_fs(row["antecedents"])[:24]} &rarr; {clean_fs(row["consequents"])[:20]}</td><td style="background:{bs};color:{sc};text-align:center;padding:5px 6px;">{row["support"]:.2f}</td><td style="background:{bc};color:{cc};text-align:center;padding:5px 6px;">{row["confidence"]:.2f}</td><td style="text-align:center;padding:5px 6px;font-weight:700;color:#7c3aed;">{row["lift"]:.1f}</td></tr>'

matrix = f'<table style="width:100%;border-collapse:collapse;font-size:12px;"><tr style="background:#f1f5f9;font-weight:700;"><th style="padding:5px 6px;">#</th><th style="padding:5px 6px;text-align:left;">Pattern</th><th style="padding:5px 6px;">Supp</th><th style="padding:5px 6px;">Conf</th><th style="padding:5px 6px;">Lift</th></tr>{rows}</table>'

# --- Pre-compute stats for middle column and dynamic SVG ---
total_rules = len(rules_df)
filtered_count = len(f)
avg_lift = round(f['lift'].mean(), 2) if len(f) > 0 else 0
avg_conf = round(f['confidence'].mean() * 100, 1) if len(f) > 0 else 0
max_lift_val = round(f['lift'].max(), 2) if len(f) > 0 else 0

if len(f) > 0:
    _ant = clean_fs(f.iloc[0]['antecedents'])
    _con_parts = [c.strip() for c in clean_fs(f.iloc[0]['consequents']).split(',')]
    svg_node0 = (_ant[:10] + '…') if len(_ant) > 10 else _ant
    svg_node1 = (_con_parts[0][:11] + '…') if len(_con_parts[0]) > 11 else _con_parts[0]
    svg_node2 = (_con_parts[1][:11] + '…') if len(_con_parts) > 1 and len(_con_parts[1]) > 11 else (_con_parts[1] if len(_con_parts) > 1 else 'Pathway')
else:
    svg_node0, svg_node1, svg_node2 = 'Profile', 'Condition', 'Pathway'

# Dynamic top conditions for advisory modal
_cond_set = []
if len(f) > 0:
    _seen = set()
    for _, _row in f.nlargest(6, 'lift').iterrows():
        for _item in clean_fs(_row['consequents']).split(','):
            _item = _item.strip()
            if _item and _item not in _seen:
                _seen.add(_item)
                _cond_set.append(_item)
top_conds = _cond_set[:4]
advisory_items_html = ''.join([f'<div style="padding:5px 0;border-bottom:1px solid #f1f5f9;font-size:13px;">&#9679; {c}</div>' for c in top_conds]) or '<div style="font-size:13px;color:#94a3b8;">No conditions found</div>'

# --- SINGLE ATOMIC HTML BLOCK: CSS + Nav + Modals + Full Layout ---
st_html(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
#MainMenu,footer,header{{visibility:hidden;}}
.block-container{{padding:0.5rem 1rem;max-width:100%;}}
html,body,[class*="css"]{{font-family:'Inter',sans-serif;color:#0f172a;}}
.stApp{{background-color:#f8fafc;background-image:linear-gradient(to right,#e2e8f0 1px,transparent 1px),linear-gradient(to bottom,#e2e8f0 1px,transparent 1px);background-size:40px 40px;}}
@keyframes spin3D{{from{{transform:translate(-38%,-50%) rotateY(0deg);}}to{{transform:translate(-38%,-50%) rotateY(360deg);}}}}
@keyframes moveEKG{{from{{background-position:0 0;}}to{{background-position:-100px 0;}}}}
.gc{{background:rgba(255,255,255,0.95);backdrop-filter:blur(20px);border:1px solid rgba(255,255,255,0.8);border-radius:15px;padding:18px;box-shadow:0 10px 30px rgba(0,0,0,0.07);margin-bottom:15px;}}
.vc{{flex:1;padding:14px;border-radius:14px;background:#fff;box-shadow:0 4px 15px rgba(0,0,0,0.05);}}
.ekg{{height:24px;width:100%;margin-top:8px;background-size:100px 100%;animation:moveEKG 1s linear infinite;opacity:0.5;}}
.ti{{margin-bottom:10px;padding-left:14px;border-left:2px solid #e2e8f0;position:relative;}}
.ti::before{{content:'';position:absolute;left:-6px;top:2px;width:10px;height:10px;border-radius:50%;background:#3b82f6;}}
.cd-ov{{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(15,23,42,0.55);backdrop-filter:blur(10px);z-index:9999;align-items:center;justify-content:center;}}
.cd-mo{{background:#fff;border-radius:20px;padding:32px;max-width:90%;position:relative;box-shadow:0 30px 80px rgba(0,0,0,0.2);max-height:85vh;overflow-y:auto;}}
.cd-cl{{position:absolute;top:14px;right:16px;cursor:pointer;font-size:22px;color:#64748b;background:none;border:none;line-height:1;}}
.ap{{border-radius:0 8px 8px 0;padding:10px 14px;margin-bottom:8px;background:#f8fafc;}}
[data-testid="stForm"]{{background:rgba(255,255,255,0.95);backdrop-filter:blur(20px);border:1px solid rgba(255,255,255,0.8);border-radius:15px;padding:4px 18px 18px;box-shadow:0 10px 30px rgba(0,0,0,0.07);margin-top:15px;}}
[data-testid="stForm"] label{{font-size:12px;font-weight:600;color:#64748b;}}
#navAppt,#navSched,#navLabs{{transition:color 0.2s;}}
#navAppt:hover,#navSched:hover,#navLabs:hover{{color:#3b82f6;cursor:pointer;}}
.mid-stat{{background:rgba(255,255,255,0.55);backdrop-filter:blur(12px);border:1px solid rgba(255,255,255,0.7);border-radius:14px;padding:14px 10px;text-align:center;margin-bottom:12px;box-shadow:0 4px 15px rgba(0,0,0,0.05);}}
</style>
{bg_html}

<div style="display:flex;align-items:center;justify-content:space-between;padding:10px 36px;background:rgba(255,255,255,0.75);backdrop-filter:blur(15px);border-radius:100px;box-shadow:0 4px 20px rgba(0,0,0,0.05);margin-bottom:18px;position:relative;z-index:10;">
  <div style="font-weight:700;font-size:19px;">&#9877; Clinical Comorbidity &amp; Treatment Patterns</div>
  <div style="display:flex;gap:20px;font-size:13px;font-weight:600;color:#64748b;">
    <span>Dashboard</span>
    <span id="navAppt" style="cursor:pointer;">Appointments</span>
    <span id="navSched" style="cursor:pointer;">Schedule</span>
    <span id="navLabs" style="cursor:pointer;">Labs</span>
  </div>
  <div style="text-align:right;"><div style="font-size:10px;font-weight:800;">PATIENT #2440</div><div style="font-size:9px;color:#3b82f6;">CONNECTED</div></div>
</div>

<div style="display:grid;grid-template-columns:1fr 1.3fr 1.6fr;gap:22px;position:relative;z-index:2;">
  <div>
    <h3 style="font-size:15px;font-weight:700;margin:0 0 12px;">Dynamic Care Plan</h3>
    {plan_html}
    <div id="demoCard" class="gc" style="cursor:pointer;text-align:center;padding:14px;"><h3 style="margin:0;font-size:14px;">Demographic Comorbidity Patterns</h3></div>
    <div id="advisoryCard" class="gc" style="border-left:4px solid #3b82f6;cursor:pointer;">
      <h3 style="font-size:14px;margin:0 0 6px;">Multi-Disciplinary Consult</h3>
      <p style="font-size:12px;color:#64748b;margin:0 0 8px;">AI-assisted specialist board recommendations.</p>
      <div style="font-size:11px;font-weight:700;color:#3b82f6;">View Insights &rarr;</div>
    </div>
    {insight}
  </div>
  <div style="padding-top:10px;">
    <div class="mid-stat"><div style="font-size:10px;color:#94a3b8;font-weight:700;letter-spacing:0.5px;">TOTAL RULES</div><div style="font-size:30px;font-weight:700;color:#0f172a;">{total_rules}</div></div>
    <div class="mid-stat"><div style="font-size:10px;color:#94a3b8;font-weight:700;letter-spacing:0.5px;">MATCHING</div><div style="font-size:30px;font-weight:700;color:#3b82f6;">{filtered_count}</div></div>
    <div class="mid-stat"><div style="font-size:10px;color:#94a3b8;font-weight:700;letter-spacing:0.5px;">MAX LIFT</div><div style="font-size:30px;font-weight:700;color:#7c3aed;">{max_lift_val}</div></div>
    <div class="mid-stat"><div style="font-size:10px;color:#94a3b8;font-weight:700;letter-spacing:0.5px;">AVG CONFIDENCE</div><div style="font-size:30px;font-weight:700;color:#10b981;">{avg_conf}%</div></div>
  </div>
  <div>
    <div style="display:flex;gap:10px;margin-bottom:15px;">
      <div class="vc"><div style="font-size:11px;color:#94a3b8;font-weight:600;">Total Visits</div><div style="font-size:24px;font-weight:700;">2,440</div><div class="ekg" style="background:linear-gradient(90deg,transparent,#3b82f6,transparent);"></div></div>
      <div class="vc"><div style="font-size:11px;color:#94a3b8;font-weight:600;">Avg Events</div><div style="font-size:24px;font-weight:700;">3.2</div><div class="ekg" style="background:linear-gradient(90deg,transparent,#eab308,transparent);"></div></div>
    </div>
    <div style="display:flex;gap:10px;margin-bottom:15px;">
      <div class="vc" style="font-size:13px;">Heart: <span id="liveHR">82</span>bpm &#10084;</div>
      <div class="vc" style="font-size:13px;">SpO2: <span id="liveBrain">98</span>% &#129695;</div>
      <div class="vc" style="font-size:13px;">Temp: <span id="liveTemp">38.5</span>&deg;C &#127777;</div>
    </div>
    <div class="gc">
      <h3 style="font-size:14px;font-weight:700;margin:0 0 12px;">Comorbidity Heatmap &amp; Graph Hybrid</h3>
      <div style="margin-bottom:15px;">{matrix}</div>
      <div style="display:flex;gap:14px;">
        <div id="svgContainer" style="flex:1.2;height:185px;background:#f8fafc;border-radius:12px;border:1px solid #e2e8f0;overflow:hidden;position:relative;">
          <div style="position:absolute;top:5px;right:5px;display:flex;gap:3px;z-index:10;">
            <button id="zoomIn" style="width:22px;height:22px;border-radius:50%;border:1px solid #e2e8f0;background:white;cursor:pointer;font-size:13px;line-height:1;">+</button>
            <button id="zoomOut" style="width:22px;height:22px;border-radius:50%;border:1px solid #e2e8f0;background:white;cursor:pointer;font-size:13px;line-height:1;">&minus;</button>
            <button id="zoomReset" style="width:22px;height:22px;border-radius:50%;border:1px solid #e2e8f0;background:white;cursor:pointer;font-size:10px;line-height:1;">&#8635;</button>
          </div>
          <svg id="flowSvg" width="100%" height="100%" viewBox="0 0 300 150" style="transform-origin:center;transition:transform 0.2s;">
            <defs><marker id="arr" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#94a3b8"/></marker></defs>
            <line x1="50" y1="75" x2="145" y2="38" stroke="#94a3b8" stroke-width="2" marker-end="url(#arr)"/>
            <line x1="50" y1="75" x2="145" y2="112" stroke="#94a3b8" stroke-width="2" marker-end="url(#arr)"/>
            <rect x="10" y="60" width="80" height="30" rx="15" fill="#3b82f6"/><text x="50" y="79" fill="white" font-size="9" text-anchor="middle">{svg_node0}</text>
            <rect x="150" y="23" width="90" height="30" rx="15" fill="#ef4444"/><text x="195" y="42" fill="white" font-size="9" text-anchor="middle">{svg_node1}</text>
            <rect x="150" y="97" width="90" height="30" rx="15" fill="#f59e0b"/><text x="195" y="116" fill="white" font-size="9" text-anchor="middle">{svg_node2}</text>
          </svg>
        </div>
        <div style="flex:1;padding-top:4px;min-width:110px;">
          <div style="font-size:10px;font-weight:800;color:#0f172a;letter-spacing:1px;margin-bottom:12px;border-bottom:1px solid #e2e8f0;padding-bottom:6px;">METRIC KEY</div>

          <div style="margin-bottom:11px;">
            <div style="font-size:10px;font-weight:700;color:#1d4ed8;margin-bottom:4px;">SUPPORT</div>
            <div style="display:flex;align-items:center;gap:5px;">
              <span style="font-size:9px;color:#94a3b8;">0</span>
              <div style="flex:1;height:8px;background:linear-gradient(to right,#eff6ff,#1d4ed8);border-radius:4px;border:1px solid #e2e8f0;"></div>
              <span style="font-size:9px;color:#1d4ed8;font-weight:600;">1</span>
            </div>
            <div style="font-size:9px;color:#94a3b8;margin-top:3px;line-height:1.4;">How often the pattern<br>appears in the dataset</div>
          </div>

          <div style="margin-bottom:11px;">
            <div style="font-size:10px;font-weight:700;color:#b91c1c;margin-bottom:4px;">CONFIDENCE</div>
            <div style="display:flex;align-items:center;gap:5px;">
              <span style="font-size:9px;color:#94a3b8;">0</span>
              <div style="flex:1;height:8px;background:linear-gradient(to right,#fef2f2,#b91c1c);border-radius:4px;border:1px solid #e2e8f0;"></div>
              <span style="font-size:9px;color:#b91c1c;font-weight:600;">1</span>
            </div>
            <div style="font-size:9px;color:#94a3b8;margin-top:3px;line-height:1.4;">Probability A leads<br>to B in same visit</div>
          </div>

          <div>
            <div style="font-size:10px;font-weight:700;color:#7c3aed;margin-bottom:4px;">LIFT</div>
            <div style="display:flex;align-items:center;gap:5px;">
              <div style="width:10px;height:10px;border-radius:2px;background:#ede9fe;border:1px solid #c4b5fd;flex-shrink:0;"></div>
              <span style="font-size:9px;color:#94a3b8;">&lt;1 negative</span>
            </div>
            <div style="display:flex;align-items:center;gap:5px;margin-top:3px;">
              <div style="width:10px;height:10px;border-radius:2px;background:#7c3aed;flex-shrink:0;"></div>
              <span style="font-size:9px;color:#94a3b8;">&gt;1 meaningful</span>
            </div>
            <div style="font-size:9px;color:#94a3b8;margin-top:3px;line-height:1.4;">Strength above<br>random co-occurrence</div>
          </div>
        </div>
      </div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1.4fr;gap:13px;">
      <div class="gc" style="border-top:4px solid #0f172a;">
        <h3 style="font-size:14px;margin:0 0 10px;">Pattern Selection</h3>
        <div style="font-size:12px;color:#475569;">
          <div style="margin-bottom:8px;"><div style="font-size:10px;font-weight:700;color:#94a3b8;letter-spacing:0.5px;margin-bottom:2px;">PRIMARY</div><div style="font-weight:600;">{st.session_state['primary_diag']}</div></div>
          <div><div style="font-size:10px;font-weight:700;color:#94a3b8;letter-spacing:0.5px;margin-bottom:2px;">SECONDARY</div><div style="font-weight:600;">{st.session_state['secondary_diag']}</div></div>
          <div style="margin-top:10px;font-size:10px;color:#94a3b8;">&#8595; Update filters below</div>
        </div>
      </div>
      <div class="gc">
        <h3 style="font-size:14px;margin:0 0 10px;">Algorithm Comparison</h3>
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <div><div style="font-size:10px;color:#94a3b8;">Apriori</div><div style="font-size:20px;font-weight:700;">1.2s</div></div>
          <div style="width:1px;height:28px;background:#e2e8f0;"></div>
          <div><div style="font-size:10px;color:#94a3b8;">FP-Growth</div><div style="font-size:20px;font-weight:700;">0.4s</div></div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- MODALS -->
<div id="apptModal" class="cd-ov"><div class="cd-mo" style="width:520px;"><button id="closeAppt" class="cd-cl">&#10005;</button><h3 style="margin-top:0;">Upcoming Appointments</h3><div class="ap" style="border-left:3px solid #3b82f6;"><b>Oct 24</b> &mdash; Endocrinology Follow-up</div><div class="ap" style="border-left:3px solid #ef4444;"><b>Nov 3</b> &mdash; Cardiology Review</div><div class="ap" style="border-left:3px solid #f59e0b;"><b>Nov 18</b> &mdash; Routine Labs</div></div></div>
<div id="schedModal" class="cd-ov"><div class="cd-mo" style="width:520px;"><button id="closeSched" class="cd-cl">&#10005;</button><h3 style="margin-top:0;">Daily Schedule</h3><div class="ap" style="border-left:3px solid #3b82f6;"><b>08:00</b> &mdash; Morning Vitals</div><div class="ap" style="border-left:3px solid #10b981;"><b>10:30</b> &mdash; Medication Review</div><div class="ap" style="border-left:3px solid #f59e0b;"><b>14:00</b> &mdash; Specialist Consultation</div></div></div>
<div id="labsModal" class="cd-ov"><div class="cd-mo" style="width:600px;"><button id="closeLabs" class="cd-cl">&#10005;</button><h3 style="margin-top:0;">Lab Results</h3><table style="width:100%;border-collapse:collapse;font-size:13px;"><tr style="background:#0f172a;color:white;"><th style="padding:8px;">Test</th><th style="padding:8px;">Result</th><th style="padding:8px;">Status</th></tr><tr style="background:#f8fafc;"><td style="padding:8px;">HbA1c</td><td style="padding:8px;">7.2%</td><td style="padding:8px;color:#f59e0b;">&#9888; Monitor</td></tr><tr><td style="padding:8px;">LDL</td><td style="padding:8px;">112 mg/dL</td><td style="padding:8px;color:#ef4444;">&#10007; High</td></tr><tr style="background:#f8fafc;"><td style="padding:8px;">eGFR</td><td style="padding:8px;">78 mL/min</td><td style="padding:8px;color:#10b981;">&#10003; Normal</td></tr></table></div></div>
<div id="advisoryModal" class="cd-ov"><div class="cd-mo" style="width:720px;"><button id="closeAdvisory" class="cd-cl">&#10005;</button><h2 style="margin-top:0;">Specialist Advisory Board</h2><hr style="border:0;border-top:1px solid #e2e8f0;margin:14px 0;"><div style="display:grid;grid-template-columns:1fr 1fr;gap:18px;"><div style="background:#f8fafc;padding:18px;border-radius:12px;"><h4 style="margin-top:0;">Top Consequent Conditions</h4>{advisory_items_html}</div><div style="background:#f8fafc;padding:18px;border-radius:12px;"><h4 style="margin-top:0;">Evidence Summary</h4><div style="font-size:13px;margin-bottom:8px;">&#128200; Max Lift: <b>{max_lift_val}</b></div><div style="font-size:13px;margin-bottom:8px;">&#127919; Avg Confidence: <b>{avg_conf}%</b></div><div style="font-size:13px;margin-bottom:8px;">&#128196; Matching Rules: <b>{filtered_count}</b></div><div style="margin-top:12px;font-size:12px;color:#64748b;">Multi-specialty review recommended for high-lift chains above {round(max_lift_val*0.8,1)}.</div></div></div></div></div>
<div id="demoModal" class="cd-ov"><div class="cd-mo" style="width:600px;"><button id="closeDemo" class="cd-cl">&#10005;</button><h3 style="margin-top:0;">Demographic Comorbidity Insights</h3><table style="width:100%;border-collapse:collapse;font-size:13px;"><tr style="background:#0f172a;color:white;"><th style="padding:8px;">Age Group</th><th style="padding:8px;">Condition</th><th style="padding:8px;">Confidence</th></tr><tr style="background:#f8fafc;"><td style="padding:8px;">Senior (65+)</td><td style="padding:8px;">Heart Disease</td><td style="padding:8px;">82%</td></tr><tr><td style="padding:8px;">Adult (40-64)</td><td style="padding:8px;">Diabetes</td><td style="padding:8px;">75%</td></tr><tr style="background:#f8fafc;"><td style="padding:8px;">Adult (40-64)</td><td style="padding:8px;">Hypertension</td><td style="padding:8px;">71%</td></tr></table></div></div>

<script>
(function(){{
  function show(id){{var m=document.getElementById(id);if(m)m.style.display='flex';}}
  function hide(id){{var m=document.getElementById(id);if(m)m.style.display='none';}}
  function bind(b,m,c){{
    var btn=document.getElementById(b),modal=document.getElementById(m),cls=document.getElementById(c);
    if(btn&&modal)btn.addEventListener('click',function(){{show(m);}});
    if(cls&&modal)cls.addEventListener('click',function(){{hide(m);}});
    if(modal)modal.addEventListener('click',function(e){{if(e.target===modal)hide(m);}});
  }}
  bind('navAppt','apptModal','closeAppt');
  bind('navSched','schedModal','closeSched');
  bind('navLabs','labsModal','closeLabs');
  bind('demoCard','demoModal','closeDemo');
  bind('advisoryCard','advisoryModal','closeAdvisory');
  setInterval(function(){{
    var hr=document.getElementById('liveHR'),br=document.getElementById('liveBrain'),tp=document.getElementById('liveTemp');
    if(hr)hr.innerText=78+Math.floor(Math.random()*12);
    if(br)br.innerText=96+Math.floor(Math.random()*4);
    if(tp)tp.innerText=(38.1+Math.random()*0.8).toFixed(1);
  }},2000);
  var scale=1,svg=document.getElementById('flowSvg');
  function applyZoom(){{if(svg)svg.style.transform='scale('+scale+')';}}
  var zi=document.getElementById('zoomIn'),zo=document.getElementById('zoomOut'),zr=document.getElementById('zoomReset');
  if(zi)zi.onclick=function(){{scale=Math.min(scale+0.2,3);applyZoom();}};
  if(zo)zo.onclick=function(){{scale=Math.max(scale-0.2,0.4);applyZoom();}};
  if(zr)zr.onclick=function(){{scale=1;applyZoom();}};
  var cont=document.getElementById('svgContainer');
  if(cont)cont.addEventListener('wheel',function(e){{e.preventDefault();scale=e.deltaY<0?Math.min(scale+0.1,3):Math.max(scale-0.1,0.4);applyZoom();}},{{passive:false}});
}})();
</script>
""")

# Pattern Selection form — Streamlit widget, placed after main render
with st.form("pattern_form"):
    c1, c2, c3 = st.columns([1, 1, 0.6])
    with c1:
        p = st.selectbox("Primary Diagnosis", ["All"] + all_items,
                         index=(["All"] + all_items).index(st.session_state['primary_diag']))
    with c2:
        s = st.selectbox("Secondary Condition", ["All"] + all_items,
                         index=(["All"] + all_items).index(st.session_state['secondary_diag']))
    with c3:
        submitted = st.form_submit_button("Apply Filters", type="primary", use_container_width=True)
    if submitted:
        st.session_state['primary_diag'], st.session_state['secondary_diag'] = p, s
        st.rerun()
