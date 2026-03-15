EV SmartHub transforms passive EV charging stations into intelligent decision hubs. The core innovation is a Physics-Informed Digital Twin methodology that replaces statistically incoherent raw dataset labels with domain-grounded targets, a duck-curve energy model, time-of-use pricing formula, physics-based charging duration, and a battery RUL degradation function. An AutoML engine benchmarks Linear Regression, Decision Tree, Random Forest, Gradient Boosting, SVR, and MLP for four tasks: Wait Time, Load Forecasting, Price Optimisation, and Maintenance RUL. All models are served via a 7-tab Gradio dashboard deployable on Google Colab.

## 📒 Notebook Sections

---

### 1️⃣ ⚡ Business Problem
> Charging stations are passive power outlets. EV SmartHub makes them intelligent.

| Silent Risk | Current Problem | SmartHub Solution |
|---|---|---|
| 🔥 Battery Safety | No real-time hazard detection | AI risk classification |
| 💸 Pricing Inefficiency | Flat-rate ignores battery wear | Physics-based dynamic pricing |
| ⚡ Grid Instability | Unmanaged peak clustering | ML load forecasting |
| 🔧 Reactive Maintenance | Run-to-failure model | RUL-based predictive alerts |

---

### 2️⃣ 📦 Data Description
> Two source datasets from Kaggle

| Dataset | Veriables | Observations | Key Variables | Source |
|---|---|---|---|---|
| ⚡ EV Charging Patterns | 20 Columns | 1320  rows | SOC, Charging Rate, Duration, Cost, Hour | [ev_charging_patterns.csv](https://www.kaggle.com/datasets/valakhorasani/electric-vehicle-charging-patterns) |
| 🔧 Battery Maintenance Logs | 30 Columns | 175,393 rows | Charge Cycles, Temperature, RUL | [EV_Predictive_Maintenance_Dataset_15min.csv](https://www.kaggle.com/datasets/datasetengineer/eviot-predictivemaint-dataset) |


> ⚠️ **Critical Finding:** Both raw datasets produced **negative R²** baselines —
> meaning models trained on raw labels performed *worse than just guessing the mean*.
> This triggered the Physics-Informed Engineering approach.

---

### 3️⃣ 🔬 Baseline Diagnostic Pass
> Intentionally training on raw data to prove the data problem

- 📉 `Charging_Cost` had **zero correlation** with energy consumed
- 📉 `Charging_Duration` had **zero correlation** with charger power
- 📉 `RUL` showed **no logical relationship** to charge cycles
- 🚨 IQR outlier check on maintenance data flagged **173,474 outlier rows**
  out of 175,000 — leaving only **1,919 usable rows!**

> 💡 **Conclusion:** The problem wasn't the algorithms — it was the data.
> We stopped training on noise and pivoted to **Physics-Informed Target Engineering**.

---

### 4️⃣ ⚛️ Physics-Informed Target Engineering
> The `preprocess_and_repair_data()` function — the heart of EV SmartHub

| 🆕 Engineered Target | 📐 Formula / Logic | 💡 Why |
|---|---|---|
| `Energy_Consumed` | Duck curve: low overnight → peak 16:00–22:00 | Reflects real-world EV grid patterns |
| `Charging_Cost` | `Energy × base_rate × peak_factor + noise` | Time-of-use pricing economics |
| `Charging_Duration` | `Energy / Charging_Rate × noise_factor` | Basic physics: P = E/t |
| `RUL` | `3000 − (Cycles × 3.0) − (max(Temp−35, 0) × 5)` clipped 0–3000 | Battery degradation formula from literature |

> ✅ **Result after engineering:** All correlations became strong, positive, and physically meaningful.
> R² shifted from **negative → clearly positive** across all four tasks.

---

### 5️⃣ 🤖 AutoML Benchmarking Layer
> Every task gets its own model competition — best performer wins

**4 Decision Services:**

| Task | Target Variable | Features Used |
|---|---|---|
| ⏱️ Wait Time Prediction | `Charging_Duration` | SOC, Capacity, Charging Rate, Energy |
| ⚡ Load Forecasting | `Energy_Consumed` | Hour, Day Code |
| 💰 Price Optimisation | `Charging_Cost` | Energy, Hour, Day Code |
| 🔧 Maintenance RUL | Engineered `RUL` | Battery Temp, Charge Cycles |

**Models Benchmarked Per Task:**

| Model | Type |
|---|---|
| 📏 Linear Regression | Baseline & interpretability |
| 🌿 Decision Tree | Non-linear, high variance reference |
| 🌲 Random Forest (default) | Bagging ensemble |
| 🌲 Random Forest (tuned) | Optimised bagging |
| 🚀 Gradient Boosting | Boosting ensemble |
| 📡 SVR | Non-linear, non-tree baseline |
| 🧠 MLP Neural Network | Small neural net baseline |

> 🏆 **Winner:** Tree-based ensembles consistently topped every task leaderboard —
> confirming non-linear physics relationships can't be captured by linear models alone.

---

### 6️⃣ 📊 EDA Dashboard Outputs
> After engineering, the data finally behaves as physics dictates

- 🔥 Strong positive correlation: `Energy` ↔ `Cost` ↔ `Duration`
- 📉 Strong negative correlation: `ChargeCycles` ↔ `RUL` (batteries die with use)
- 🦆 Bar chart of avg. energy by hour reveals clear **duck curve** pattern
- 🌐 3D Plotly surface: `Hour × Energy → Cost` — peak hours visibly raise price
- 📈 Scatter: `Cycles vs RUL` shows smooth monotonic degradation line

---

### 7️⃣ 🖥️ 7-Tab Gradio Dashboard
> Deployed entirely in Google Colab — no backend or frontend stack needed

| Tab | Purpose | User Input | Output |
|---|---|---|---|
| 1️⃣ Data Setup | Upload CSVs or generate synthetic data | File upload / button | Repaired data preview |
| 2️⃣ EDA Dashboard | Visualise engineered relationships | Button | 6 diagnostic plots |
| 3️⃣ AutoML | Train & compare all models per task | Dropdown (task) | Leaderboard + bar chart |
| 4️⃣ Wait Time | How long will this session take? | SOC, Capacity, Rate | Duration (hrs:mins) |
| 5️⃣ Load Forecast | What load will this slot impose? | Hour, Day | Predicted kWh |
| 6️⃣ Price Optimiser | What is the fair price for this session? | Energy, Hour, Day | Cost (RM) + RM/kWh |
| 7️⃣ Maintenance RUL | How much life is left in this battery? | Temp, Cycles | RUL + Health Status |

```python
# Launch the app in Google Colab
app.launch(share=True)  # Generates a public shareable URL instantly
