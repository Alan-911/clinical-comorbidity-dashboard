"""
Generate the project report PDF for the Clinical Comorbidity Dashboard.

Usage:
    python3 generate_report.py

Outputs: docs/Clinical_Comorbidity_Dashboard_Report.pdf
"""

from fpdf import FPDF
from datetime import datetime
import os

OUT = os.path.join(os.path.dirname(__file__), "Clinical_Comorbidity_Dashboard_Report.pdf")

# Color palette (matches the app)
SLATE_DARK = (15, 23, 42)
SLATE_MID = (71, 85, 105)
SLATE_LIGHT = (148, 163, 184)
BLUE = (59, 130, 246)
PURPLE = (124, 58, 237)
GREEN = (16, 185, 129)
AMBER = (245, 158, 11)
RED = (239, 68, 68)
BG_LIGHT = (248, 250, 252)
BG_CODE = (241, 245, 249)


class Report(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(20, 20, 20)
        self.alias_nb_pages()

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*SLATE_LIGHT)
        self.cell(0, 6, "Clinical Comorbidity Dashboard - Project Report", align="L")
        self.cell(0, 6, f"Page {self.page_no()} / {{nb}}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*SLATE_LIGHT)
        self.set_line_width(0.1)
        self.line(20, 28, 190, 28)
        self.ln(6)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*SLATE_LIGHT)
        self.cell(0, 5, "github.com/Alan-911/clinical-comorbidity-dashboard", align="C")

    # --- typography helpers ---
    def h1(self, text, color=SLATE_DARK):
        # Start each major section on a fresh page (unless cursor is near top)
        if self.get_y() > 40:
            self.add_page()
        self.set_font("Helvetica", "B", 20)
        self.set_text_color(*color)
        self.ln(4)
        self.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*BLUE)
        self.set_line_width(0.8)
        self.line(20, self.get_y(), 60, self.get_y())
        self.ln(6)

    def h2(self, text):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*BLUE)
        self.ln(3)
        self.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def h3(self, text):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*SLATE_DARK)
        self.ln(1)
        self.cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")

    def body(self, text, size=10):
        self.set_font("Helvetica", "", size)
        self.set_text_color(*SLATE_DARK)
        self.multi_cell(0, 5.2, text)
        self.ln(1)

    def bullet(self, label, desc):
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*BLUE)
        self.write(5, f"- {label}  ")
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*SLATE_DARK)
        self.write(5, desc)
        self.ln(7)

    def code_block(self, code, height_per_line=4.5):
        lines = code.strip("\n").split("\n")
        h = height_per_line * len(lines) + 4
        self.set_fill_color(*BG_CODE)
        self.set_draw_color(*SLATE_LIGHT)
        self.set_line_width(0.1)
        x, y = self.get_x(), self.get_y()
        self.rect(x, y, 170, h, style="FD")
        self.set_xy(x + 3, y + 2)
        self.set_font("Courier", "", 8.5)
        self.set_text_color(*SLATE_DARK)
        for line in lines:
            self.set_x(x + 3)
            self.cell(0, height_per_line, line, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def callout(self, title, text, color=BLUE):
        self.set_fill_color(*BG_LIGHT)
        self.set_draw_color(*color)
        self.set_line_width(0.3)
        x, y = self.get_x(), self.get_y()
        lines = len(text) // 85 + 1
        h = 8 + lines * 5
        self.rect(x, y, 170, h, style="FD")
        self.set_line_width(1.2)
        self.line(x, y, x, y + h)
        self.set_xy(x + 4, y + 2)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*color)
        self.cell(0, 5, title, new_x="LMARGIN", new_y="NEXT")
        self.set_x(x + 4)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*SLATE_DARK)
        self.multi_cell(160, 4.5, text)
        self.ln(3)

    def key_value_row(self, label, value, fill=False):
        if fill:
            self.set_fill_color(*BG_LIGHT)
        else:
            self.set_fill_color(255, 255, 255)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*SLATE_MID)
        self.cell(50, 7, f"  {label}", border=0, fill=True)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*SLATE_DARK)
        self.cell(120, 7, value, border=0, fill=True, new_x="LMARGIN", new_y="NEXT")


def build():
    pdf = Report()

    # ==========================================================================
    # COVER PAGE
    # ==========================================================================
    pdf.add_page()
    pdf.set_fill_color(*SLATE_DARK)
    pdf.rect(0, 0, 210, 120, style="F")

    pdf.set_xy(20, 35)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*SLATE_LIGHT)
    pdf.cell(0, 5, "SCHOOL PROJECT REPORT", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_x(20)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(255, 255, 255)
    pdf.multi_cell(170, 12, "Clinical Comorbidity & Treatment Patterns Dashboard")
    pdf.ln(2)

    pdf.set_x(20)
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(*BLUE)
    pdf.cell(0, 7, "A Data Mining Approach Using FP-Growth Association Rule Mining",
             new_x="LMARGIN", new_y="NEXT")

    # Meta info card
    pdf.set_xy(20, 140)
    pdf.set_draw_color(*SLATE_LIGHT)
    pdf.set_fill_color(*BG_LIGHT)
    pdf.rect(20, 140, 170, 90, style="FD")

    pdf.set_xy(28, 148)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*SLATE_MID)
    pdf.cell(0, 6, "PROJECT DETAILS", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    details = [
        ("Project Type", "Academic data-mining coursework"),
        ("Domain", "Healthcare informatics"),
        ("Algorithm", "FP-Growth (frequent pattern mining)"),
        ("Dataset", "2,440 clinical visit transactions"),
        ("Framework", "Streamlit (Python 3)"),
        ("Repository", "github.com/Alan-911/clinical-comorbidity-dashboard"),
        ("Live App", "clinical-comorbidity-dashboard.streamlit.app"),
        ("Date", datetime.now().strftime("%B %Y")),
    ]
    for label, value in details:
        pdf.set_x(28)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*BLUE)
        pdf.cell(42, 6, label)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*SLATE_DARK)
        pdf.cell(0, 6, value, new_x="LMARGIN", new_y="NEXT")

    pdf.set_xy(20, 250)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(*SLATE_LIGHT)
    pdf.cell(0, 5, "Discovering hidden patterns in patient comorbidities through unsupervised", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, "association rule mining to support clinical awareness and research.", align="C")

    # ==========================================================================
    # TABLE OF CONTENTS
    # ==========================================================================
    pdf.h1("Table of Contents")
    toc = [
        ("1.  Executive Summary", 3),
        ("2.  Introduction: The Clinical Problem", 3),
        ("3.  Why Data Mining?", 4),
        ("4.  Association Rule Mining: Theory", 5),
        ("5.  Algorithm Choice: Apriori vs FP-Growth", 7),
        ("6.  System Architecture", 8),
        ("7.  Technology Stack", 9),
        ("8.  Data Pipeline", 11),
        ("9.  Implementation Walkthrough", 12),
        ("10. User Interface & Features", 13),
        ("11. Deployment", 14),
        ("12. Limitations", 14),
        ("13. Future Work", 15),
        ("14. Conclusion", 15),
        ("15. References", 16),
    ]
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*SLATE_DARK)
    for item, page in toc:
        pdf.cell(150, 8, item)
        pdf.set_text_color(*BLUE)
        pdf.cell(0, 8, str(page), align="R", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(*SLATE_DARK)

    # ==========================================================================
    # 1. EXECUTIVE SUMMARY
    # ==========================================================================
    pdf.h1("1. Executive Summary")
    pdf.body(
        "This report documents the design, implementation, and deployment of an interactive clinical "
        "comorbidity analytics dashboard built as a data-mining coursework project. The system applies "
        "the FP-Growth association rule mining algorithm to 2,440 clinical visit transactions to "
        "discover which medical conditions tend to co-occur in patients, then surfaces the resulting "
        "rules through a polished, interactive web interface."
    )
    pdf.body(
        "The project demonstrates the end-to-end data-mining workflow: from raw transaction data "
        "through algorithm selection, rule generation, filtering, visualization, and deployment. "
        "It is not a medical device and makes no clinical claims; it exists as an educational "
        "demonstration of how unsupervised pattern discovery can extract meaningful structure from "
        "healthcare-style data."
    )
    pdf.callout(
        "KEY OUTPUT",
        "A library of association rules of the form X -> Y with quantified support, "
        "confidence, and lift metrics, rendered in a responsive dashboard where a user can "
        "filter by primary and secondary diagnosis, inspect the top patterns, and drill "
        "into patient-context modals (Dashboard, Appointments, Schedule, Labs, Demographics).",
        BLUE,
    )

    # ==========================================================================
    # 2. INTRODUCTION
    # ==========================================================================
    pdf.h1("2. Introduction: The Clinical Problem")
    pdf.h2("2.1 What are comorbidities?")
    pdf.body(
        "A comorbidity is the simultaneous presence of two or more medical conditions in a single "
        "patient. Real patients rarely present with only one condition in isolation. A diabetic "
        "patient may also have hypertension; a senior with heart disease may also develop chronic "
        "kidney disease. The relationships among these conditions are often predictable and "
        "statistically meaningful - yet they are not always obvious from casual inspection of the data."
    )
    pdf.h2("2.2 Why this matters")
    pdf.body(
        "Understanding which conditions cluster together supports three broad goals: (1) preventive "
        "screening - if condition A strongly predicts B, then a patient presenting with A should be "
        "screened for B; (2) treatment planning - multi-condition patients need coordinated care "
        "across specialties; (3) research - novel or unexpected comorbidity patterns can inform "
        "epidemiology and etiological hypotheses."
    )
    pdf.h2("2.3 The data-mining opportunity")
    pdf.body(
        "Clinical visit data - where each visit records the set of conditions or events observed - "
        "maps naturally onto the 'transaction' abstraction used in market-basket analysis. This "
        "opens the door to mature, well-studied pattern-mining algorithms originally developed for "
        "retail analytics, now reframed for healthcare insight."
    )

    # ==========================================================================
    # 3. WHY DATA MINING?
    # ==========================================================================
    pdf.h1("3. Why Data Mining?")
    pdf.body(
        "Data mining is the process of discovering patterns, correlations, and anomalies in large "
        "datasets using techniques from statistics, machine learning, and database systems. "
        "It differs from classical statistics in intent: rather than testing a pre-formulated "
        "hypothesis, data mining is exploratory - it surfaces structure you did not necessarily "
        "anticipate."
    )
    pdf.h2("3.1 Why it fits this problem")
    pdf.bullet("Scale:",
               "Manually eyeballing 2,440 visits to find recurring patterns is infeasible. "
               "Algorithms can scan the entire dataset in a fraction of a second.")
    pdf.bullet("Combinatorics:",
               "With dozens of possible conditions, the number of possible multi-condition "
               "combinations explodes. Only systematic algorithms can enumerate and score them all.")
    pdf.bullet("Objectivity:",
               "Quantified metrics (support, confidence, lift) replace subjective judgment. "
               "Two analysts running the same algorithm on the same data get the same rules.")
    pdf.bullet("Reproducibility:",
               "The pipeline is deterministic: raw data -> encoded transactions -> FP-tree -> "
               "rule set. Anyone can rerun it and verify the results.")
    pdf.h2("3.2 What we extract")
    pdf.body(
        "The data-mining task here is Association Rule Mining - a form of unsupervised learning. "
        "We want rules of the shape: 'If a patient has condition set X, they frequently also have "
        "condition set Y, with confidence c and lift l.' These rules are interpretable, actionable, "
        "and directly useful for the downstream visualization."
    )
    pdf.callout(
        "DATA MINING IN ONE SENTENCE",
        "Data mining turns a pile of raw, unlabeled transactions into a ranked catalog of "
        "human-readable rules describing what tends to co-occur and how strongly.",
        PURPLE,
    )

    # ==========================================================================
    # 4. ASSOCIATION RULE MINING: THEORY
    # ==========================================================================
    pdf.h1("4. Association Rule Mining: Theory")
    pdf.h2("4.1 The transaction model")
    pdf.body(
        "Each clinical visit is modeled as a transaction - a set of items (conditions or events). "
        "Example: Visit #42 = {Hypertension, Obesity, Diabetes}. The full dataset is a list of "
        "such transactions. From this list we mine frequent itemsets (combinations of items that "
        "appear together often) and then derive association rules from those itemsets."
    )
    pdf.h2("4.2 The three core metrics")
    pdf.body("A rule X -> Y is scored by three quantities:")
    pdf.code_block(
        "Support(X -> Y)    = count(X AND Y) / N                   (freq. of the pattern)\n"
        "Confidence(X -> Y) = count(X AND Y) / count(X)             (P(Y | X))\n"
        "Lift(X -> Y)       = Confidence(X -> Y) / Support(Y)       (vs. random baseline)"
    )
    pdf.bullet("Support:",
               "How often the combined itemset appears in the whole dataset. Low support "
               "means the rule is statistically marginal, even if the confidence is high.")
    pdf.bullet("Confidence:",
               "If a patient has X, what fraction of the time do they also have Y? This is "
               "the directional, conditional probability.")
    pdf.bullet("Lift:",
               "The amount by which X and Y co-occur more often than they would under "
               "independence. Lift = 1 means no association; lift > 1 means positive "
               "co-occurrence; lift < 1 means X suppresses Y.")
    pdf.h2("4.3 A worked example")
    pdf.body(
        "Suppose in 1,000 visits: 200 contain Diabetes, 150 contain Heart Disease, and 120 contain "
        "both. Then for the rule Diabetes -> Heart Disease:"
    )
    pdf.code_block(
        "Support    = 120 / 1000 = 0.12  (12% of visits show both)\n"
        "Confidence = 120 / 200  = 0.60  (60% of diabetics also have heart disease)\n"
        "Lift       = 0.60 / (150/1000) = 0.60 / 0.15 = 4.00"
    )
    pdf.body(
        "A lift of 4.0 means diabetics are 4x more likely to have heart disease than a randomly "
        "chosen patient. That is a strong, clinically meaningful signal, and the kind of rule our "
        "dashboard surfaces at the top of the Pattern Matrix."
    )
    pdf.h2("4.4 Thresholds used in this project")
    pdf.key_value_row("min_support", "0.01  (pattern appears in at least 1% of visits)", fill=True)
    pdf.key_value_row("min_confidence", "0.50  (rule is correct at least half the time)", fill=False)
    pdf.key_value_row("Ranking metric", "Lift (descending)", fill=True)
    pdf.ln(3)
    pdf.body(
        "These are conservative-enough values to avoid returning every possible trivial pair while "
        "still producing a rich enough rule set (>300 rules) to make the dashboard interesting."
    )

    # ==========================================================================
    # 5. ALGORITHM CHOICE
    # ==========================================================================
    pdf.h1("5. Algorithm Choice: Apriori vs FP-Growth")
    pdf.body(
        "The classical association-mining algorithm is Apriori (Agrawal & Srikant, 1994). "
        "FP-Growth (Han et al., 2000) was introduced to address Apriori's scalability problems. "
        "Our system uses FP-Growth. The comparison below justifies that choice."
    )
    pdf.h2("5.1 Apriori")
    pdf.body(
        "Apriori iteratively generates candidate itemsets: first all size-1, then size-2, and so "
        "on. At each level it scans the entire dataset to count supports, prunes infrequent "
        "candidates, and extends the survivors. The dataset is scanned k times for itemsets of "
        "size k. The number of candidates grows combinatorially, which becomes a bottleneck on "
        "dense, multi-item transactions typical of clinical data."
    )
    pdf.h2("5.2 FP-Growth")
    pdf.body(
        "FP-Growth (Frequent Pattern Growth) avoids candidate generation entirely. It builds a "
        "compressed prefix-tree (FP-tree) that encodes the transaction database in a single scan. "
        "Frequent itemsets are then mined recursively by traversing the tree. The result: fewer "
        "passes over the data, no exponential candidate explosion, and linear-like behavior as "
        "the dataset grows."
    )
    pdf.h2("5.3 Side-by-side")
    pdf.set_fill_color(*BG_CODE)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*SLATE_DARK)
    pdf.cell(40, 8, "Property", fill=True, border=0)
    pdf.cell(65, 8, "Apriori", fill=True, border=0)
    pdf.cell(65, 8, "FP-Growth", fill=True, border=0, new_x="LMARGIN", new_y="NEXT")
    rows = [
        ("Data scans", "k scans (one per itemset size)", "2 scans total"),
        ("Candidate generation", "Yes - exponential in k", "No candidates at all"),
        ("Memory", "Flat lists of candidates", "Compressed FP-tree"),
        ("Performance", "Slow on dense data", "Fast on dense data"),
        ("Our benchmark", "~1.2s on 2,440 visits", "~0.4s on 2,440 visits"),
    ]
    pdf.set_font("Helvetica", "", 9)
    for i, (a, b, c) in enumerate(rows):
        if i % 2 == 0:
            pdf.set_fill_color(*BG_LIGHT)
        else:
            pdf.set_fill_color(255, 255, 255)
        pdf.cell(40, 7, a, fill=True, border=0)
        pdf.cell(65, 7, b, fill=True, border=0)
        pdf.cell(65, 7, c, fill=True, border=0, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.callout(
        "WHY FP-GROWTH",
        "Clinical transactions are dense (many conditions per visit) and the condition "
        "vocabulary is moderate (dozens of items). Both properties make Apriori struggle "
        "and FP-Growth shine. For real-time, interactive dashboards, the roughly 3x speedup "
        "is what keeps the UI responsive after each filter change.",
        GREEN,
    )

    # ==========================================================================
    # 6. SYSTEM ARCHITECTURE
    # ==========================================================================
    pdf.h1("6. System Architecture")
    pdf.body(
        "The system is a single-page Streamlit application (app.py) that performs three tasks: "
        "(1) load or generate association rules; (2) filter them based on UI state; (3) render "
        "the filtered rules as a dashboard. A clean separation of concerns is enforced by "
        "Streamlit's caching and session-state primitives."
    )
    pdf.h2("6.1 Data flow")
    pdf.code_block(
        "transactions.csv                       (raw input)\n"
        "      |\n"
        "      v\n"
        "TransactionEncoder  -->  one-hot DataFrame\n"
        "      |\n"
        "      v\n"
        "fpgrowth (min_support=0.01)  -->  frequent itemsets\n"
        "      |\n"
        "      v\n"
        "association_rules (min_conf=0.5)  -->  rules DataFrame\n"
        "      |\n"
        "      v\n"
        "association_rules.csv                  (cached to disk)\n"
        "      |\n"
        "      v\n"
        "Streamlit UI  ------->  filters, stats, heatmap, SVG graph, modals"
    )
    pdf.h2("6.2 Component map")
    pdf.bullet("Data layer:", "CSV files in transactions/ and data_processed/")
    pdf.bullet("Mining layer:", "mlxtend's fpgrowth + association_rules functions")
    pdf.bullet("State layer:", "st.session_state for filter selections (primary/secondary diagnosis)")
    pdf.bullet("Render layer:", "Streamlit components + inline HTML/CSS for the custom look")
    pdf.bullet("Interaction layer:", "@st.dialog modals for Dashboard, Appointments, Schedule, "
                                     "Labs, Demographic, Advisory, Patient views")

    # ==========================================================================
    # 7. TECHNOLOGY STACK
    # ==========================================================================
    pdf.h1("7. Technology Stack")
    pdf.body(
        "Every tool used in the project, what it does, and why it was chosen."
    )

    tools = [
        ("Python 3",
         "General-purpose language",
         "Python's ecosystem is the de-facto standard for data science. Every library in the stack is either Python-native or has a first-class Python binding."),
        ("Streamlit",
         "Web application framework for data apps",
         "Streamlit turns Python scripts into interactive web apps with no JavaScript or HTML required from the developer. It handles state, re-execution, caching, and routing internally. Ideal for rapid data-app prototyping."),
        ("pandas",
         "Tabular data manipulation",
         "pandas is used to load the CSV, manipulate the rules DataFrame, apply filters, sort by lift, and extract top-N rows. Its DataFrame abstraction makes the filtering and slicing code short and readable."),
        ("mlxtend",
         "Machine learning extensions library",
         "Provides the production-quality FP-Growth and association_rules implementations. Also provides TransactionEncoder, which one-hot encodes lists-of-lists into the boolean matrix the algorithm expects. Saves us from writing the algorithm from scratch."),
        ("matplotlib",
         "Plotting and colormaps",
         "We use matplotlib's colormaps ('Blues' and 'Reds') to color the support/confidence cells of the Pattern Matrix heatmap. No full plots are drawn; only the colormap utility is imported."),
        ("numpy",
         "Numerical operations",
         "Transitive dependency of pandas and mlxtend. Used implicitly."),
        ("base64",
         "Binary-to-text encoding",
         "The anatomical background PNG is read as bytes, base64-encoded, and embedded directly as a data URI in the page's HTML. This eliminates a separate image request and lets the image render instantly with the rest of the page."),
        ("GitHub",
         "Version control and collaboration",
         "The entire codebase is stored at github.com/Alan-911/clinical-comorbidity-dashboard. Git history provides a full audit trail of every change, and GitHub hosts the remote repository that Streamlit Cloud pulls from on each deploy."),
        ("Streamlit Community Cloud",
         "Zero-config deployment",
         "Streamlit Cloud connects directly to the GitHub repository. Every push to the main branch automatically triggers a rebuild of the live app. No Docker, no CI pipeline, no server management required."),
    ]

    for name, short, long in tools:
        pdf.h3(name)
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*PURPLE)
        pdf.cell(0, 5, short, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*SLATE_DARK)
        pdf.multi_cell(0, 5, long)
        pdf.ln(2)

    # ==========================================================================
    # 8. DATA PIPELINE
    # ==========================================================================
    pdf.h1("8. Data Pipeline")
    pdf.h2("8.1 Input")
    pdf.body(
        "The raw input is transactions/transactions.csv - a two-column file where each row has a "
        "visit identifier and a comma-separated list of conditions observed at that visit."
    )
    pdf.h2("8.2 Encoding")
    pdf.body(
        "TransactionEncoder takes a list of item-lists and produces a boolean DataFrame: one row "
        "per transaction, one column per unique item, True if that item appeared in that "
        "transaction. This is the standard input format for frequent-pattern miners."
    )
    pdf.code_block(
        "te = TransactionEncoder()\n"
        "arr = te.fit(item_lists).transform(item_lists)\n"
        "df  = pd.DataFrame(arr, columns=te.columns_)"
    )
    pdf.h2("8.3 Mining")
    pdf.body(
        "fpgrowth returns the set of frequent itemsets - every combination of items that meets the "
        "minimum support threshold. association_rules then converts those itemsets into directional "
        "rules (X -> Y) with support, confidence, and lift attached."
    )
    pdf.code_block(
        "freq_itemsets = fpgrowth(df, min_support=0.01, use_colnames=True)\n"
        "rules = association_rules(freq_itemsets,\n"
        "                          metric='confidence',\n"
        "                          min_threshold=0.5)"
    )
    pdf.h2("8.4 Caching")
    pdf.body(
        "Mining is expensive - we do it only once. The rules DataFrame is written to "
        "data_processed/association_rules.csv. On subsequent app starts, the app reads that CSV "
        "directly and skips the mining step. Additionally, Streamlit's @st.cache_data decorator "
        "keeps the rules in memory across reruns, so filter changes are instant."
    )

    # ==========================================================================
    # 9. IMPLEMENTATION
    # ==========================================================================
    pdf.h1("9. Implementation Walkthrough")
    pdf.h2("9.1 File structure")
    pdf.code_block(
        "clinical-comorbidity-dashboard/\n"
        "  app.py                              (main application, ~550 lines)\n"
        "  requirements.txt                    (streamlit, pandas, numpy, mlxtend, matplotlib)\n"
        "  transactions/\n"
        "    transactions.csv                  (raw input)\n"
        "  data_processed/\n"
        "    association_rules.csv             (cached mining output)\n"
        "  visualizations/\n"
        "    anatomical_model.png              (decorative background)\n"
        "  docs/\n"
        "    Clinical_Comorbidity_Dashboard_Report.pdf   (this report)\n"
        "    generate_report.py                (report generator script)"
    )
    pdf.h2("9.2 Session state")
    pdf.body(
        "The app keeps two pieces of UI state: the primary diagnosis filter and the secondary "
        "condition filter. Both default to 'All'. They are stored in st.session_state so they "
        "persist across reruns, and updated by the Pattern Selection form at the bottom of the "
        "page."
    )
    pdf.h2("9.3 Filtering")
    pdf.body(
        "Filtering is a simple substring match over the stringified antecedent / consequent "
        "columns. The result is sorted by lift descending, and the top 3 rows drive the Dynamic "
        "Care Plan, the Top Clinical Insight banner, and the top-5 Pattern Matrix heatmap."
    )
    pdf.code_block(
        "f = rules_df.copy()\n"
        "if primary_diag != 'All':\n"
        "    f = f[f['antecedents'].apply(lambda x: primary_diag in str(x))]\n"
        "if secondary_diag != 'All':\n"
        "    f = f[f['consequents'].apply(lambda x: secondary_diag in str(x))]\n"
        "f = f.sort_values('lift', ascending=False)"
    )
    pdf.h2("9.4 Heatmap coloring")
    pdf.body(
        "Each row of the top-5 matrix has its support and confidence cells colored by matplotlib "
        "colormaps (Blues for support, Reds for confidence). Higher values produce darker cells, "
        "which lets the eye pick out strong rules at a glance without reading numbers."
    )
    pdf.code_block(
        "cmap_s = plt.get_cmap('Blues')\n"
        "cmap_c = plt.get_cmap('Reds')\n"
        "bs = mcolors.to_hex(cmap_s(row['support']/0.2))\n"
        "bc = mcolors.to_hex(cmap_c(row['confidence']/1.0))"
    )
    pdf.h2("9.5 Modal dialogs")
    pdf.body(
        "Streamlit 1.31+ introduced @st.dialog, a decorator that turns a function into a popup "
        "modal triggered by a button click. This avoids the need for custom JavaScript or overlay "
        "hacks. Our app has seven dialogs: Dashboard, Patient, Appointments, Schedule, Labs, "
        "Advisory, and Demographic - all sharing a consistent dark-gradient header for visual "
        "cohesion."
    )

    # ==========================================================================
    # 10. UI & FEATURES
    # ==========================================================================
    pdf.h1("10. User Interface & Features")
    pdf.h2("10.1 Layout")
    pdf.body(
        "The app uses a three-column main layout (ratios 1 : 1.3 : 1.6). The left column shows "
        "the Dynamic Care Plan and two card buttons (Demographic Patterns, Multi-Disciplinary "
        "Consult). The middle column shows four large stat cards (Total Rules, Matching, Max "
        "Lift, Avg Confidence). The right column holds the Vital Signs row, the Pattern Matrix, "
        "and the SVG Graph Hybrid."
    )
    pdf.h2("10.2 Top navigation")
    pdf.body(
        "A pill-shaped white nav bar (implemented via st.container with a key-targeted CSS class) "
        "holds four link-styled buttons: Dashboard, Appointments, Schedule, Labs. The right end "
        "shows a patient badge (#2440 - Connected). Each button opens the corresponding modal."
    )
    pdf.h2("10.3 Modals")
    pdf.bullet("Dashboard:",
               "Top-level KPIs, active filters, system info. Dark header: 'N rules analyzed'.")
    pdf.bullet("Appointments:",
               "Four upcoming visits with priority badges, days-until counters, and a "
               "rule-linked justification referencing the top-lift pattern.")
    pdf.bullet("Schedule:",
               "Today's clinical tasks on a vertical timeline with live NOW status chips "
               "(Done / In Progress / Upcoming) computed from datetime.now().")
    pdf.bullet("Labs:",
               "Three grouped panels (Metabolic, Cardiac, Renal) with reference ranges, "
               "SVG trend sparklines, and a clinical note tying the abnormal values to "
               "the mined rule. Includes a CSV download.")
    pdf.bullet("Demographic:",
               "Three age cohorts (Young 18-39, Adult 40-64, Senior 65+) ordered ascending "
               "by age, with per-cohort progress bars for the top conditions.")
    pdf.bullet("Advisory:",
               "Multi-specialty consult summary with top consequents and evidence stats.")
    pdf.bullet("Patient:",
               "Avatar, connection status, monitoring focus, next review.")
    pdf.h2("10.4 Design language")
    pdf.body(
        "The UI uses glass-morphism (translucent cards with backdrop-blur), the Inter typeface, "
        "a soft gray grid-paper background, and a rotating 3D anatomical illustration "
        "(transform: rotateY + CSS keyframes). All seven modals share the same dark-gradient "
        "header banner so they feel like panels from a single section rather than unrelated popups."
    )
    pdf.callout(
        "DESIGN PHILOSOPHY",
        "Every visual element is intentional: the gray grid evokes anatomical diagrams; "
        "the dark header banners establish modal identity; the color-coded left borders "
        "(green / blue / purple / amber) communicate semantic categories (young / adult / "
        "senior; routine / urgent; support / confidence / lift). The goal is a dashboard "
        "that a non-technical viewer recognizes as professional within the first second.",
        PURPLE,
    )

    # ==========================================================================
    # 11. DEPLOYMENT
    # ==========================================================================
    pdf.h1("11. Deployment")
    pdf.body(
        "The project uses a push-to-deploy workflow: committing to the main branch on GitHub "
        "triggers an automatic rebuild on Streamlit Community Cloud."
    )
    pdf.code_block(
        "git add app.py\n"
        "git commit -m '<message>'\n"
        "git push origin main\n"
        "# ~1-2 minutes later: https://clinical-comorbidity-dashboard.streamlit.app is live"
    )
    pdf.body(
        "Streamlit Cloud reads requirements.txt to install dependencies, then runs "
        "streamlit run app.py. No Dockerfile, no CI configuration, and no server maintenance "
        "are required."
    )

    # ==========================================================================
    # 12. LIMITATIONS
    # ==========================================================================
    pdf.h1("12. Limitations")
    pdf.bullet("Dataset scale:",
               "2,440 visits is adequate for a coursework demonstration but too small for any "
               "serious inference. Statistically stable rules on clinical data typically "
               "require tens to hundreds of thousands of transactions.")
    pdf.bullet("No temporality:",
               "Association rules treat each visit as an unordered bag of conditions. Real "
               "comorbidity is sequential (hypertension -> kidney disease over years), and "
               "that temporal structure is discarded.")
    pdf.bullet("Substring filtering:",
               "The UI filter uses simple substring matching over stringified itemsets. This "
               "can over-match related-but-distinct conditions. Proper vocabulary-based "
               "filtering (ICD-10, SNOMED) would be needed for production use.")
    pdf.bullet("No confounder adjustment:",
               "Lift and confidence are unadjusted. In reality, rules like 'Obesity -> "
               "Diabetes' are confounded by age, lifestyle, genetics - none of which the "
               "algorithm sees.")
    pdf.bullet("Not a medical device:",
               "The HIPAA-compliant / ISO-certified badges in the footer are cosmetic; the "
               "system has not undergone any regulatory review. It is strictly educational.")

    # ==========================================================================
    # 13. FUTURE WORK
    # ==========================================================================
    pdf.h1("13. Future Work")
    pdf.body(
        "Natural extensions of this work, in rough order of effort:"
    )
    pdf.bullet("Larger dataset:",
               "Integrate a public benchmark such as MIMIC-IV (60,000+ ICU stays) to make "
               "the mined rules statistically meaningful.")
    pdf.bullet("Temporal mining:",
               "Replace FP-Growth with a sequential-pattern miner like PrefixSpan or "
               "SPADE to capture condition-progression order.")
    pdf.bullet("Proper vocabularies:",
               "Code conditions with ICD-10 or SNOMED CT instead of free-text strings.")
    pdf.bullet("Supervised layer:",
               "Use the mined rules as features in a risk-prediction model (gradient "
               "boosting, logistic regression) for quantitative patient-level predictions.")
    pdf.bullet("Subgroup analysis:",
               "Stratify rules by age, sex, geography, and test for fairness across groups.")
    pdf.bullet("FHIR integration:",
               "Wrap the rule engine as a CDS Hook so the results could, in principle, "
               "surface inside an EHR at the point of care.")

    # ==========================================================================
    # 14. CONCLUSION
    # ==========================================================================
    pdf.h1("14. Conclusion")
    pdf.body(
        "This project delivers a complete, working data-mining pipeline end-to-end: it takes raw "
        "clinical transactions, applies the FP-Growth frequent-pattern algorithm via mlxtend, "
        "derives association rules with support / confidence / lift metrics, persists them, and "
        "exposes them through a polished interactive dashboard built with Streamlit."
    )
    pdf.body(
        "The primary educational value is in the end-to-end integration: every layer of a typical "
        "data-mining stack is represented (data acquisition, encoding, mining, filtering, "
        "visualization, deployment), and every layer is implemented with a widely-used, "
        "industry-standard tool. The dashboard is not a clinical product, but it is a faithful "
        "demonstration of how association rule mining can turn undifferentiated transaction data "
        "into interpretable, quantified, and visually accessible insight."
    )
    pdf.callout(
        "FINAL TAKEAWAY",
        "Data mining does not replace clinical expertise - but it reliably surfaces "
        "patterns that would be invisible to manual inspection, and the metrics "
        "(support, confidence, lift) give the human reader a quantified basis for "
        "deciding which of those patterns are worth paying attention to.",
        GREEN,
    )

    # ==========================================================================
    # 15. REFERENCES
    # ==========================================================================
    pdf.h1("15. References")
    refs = [
        "Agrawal, R. & Srikant, R. (1994). Fast algorithms for mining association rules in "
        "large databases. Proceedings of the 20th International Conference on Very Large Data "
        "Bases (VLDB '94), 487-499.",
        "Han, J., Pei, J. & Yin, Y. (2000). Mining frequent patterns without candidate "
        "generation. ACM SIGMOD Record, 29(2), 1-12.",
        "Raschka, S. (2018). MLxtend: Providing machine learning and data science utilities "
        "and extensions to Python's scientific computing stack. Journal of Open Source "
        "Software, 3(24), 638.",
        "The pandas development team. (2024). pandas-dev/pandas: Pandas. Zenodo.",
        "Streamlit Inc. (2024). Streamlit Documentation. streamlit.io/docs",
        "Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in Science "
        "& Engineering, 9(3), 90-95.",
    ]
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*SLATE_DARK)
    for i, r in enumerate(refs, 1):
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*BLUE)
        pdf.cell(8, 5, f"[{i}]")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*SLATE_DARK)
        pdf.multi_cell(162, 5, r)
        pdf.ln(1)

    # Save
    pdf.output(OUT)
    print(f"Wrote {OUT}")
    print(f"Size: {os.path.getsize(OUT):,} bytes")
    print(f"Pages: {pdf.pages_count}")


if __name__ == "__main__":
    build()
