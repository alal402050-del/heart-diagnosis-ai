import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. LOAD DATA & TRAIN MODEL
# ==========================================

# Load the dataset from the separate CSV file
try:
    df = pd.read_csv('heart.csv')
except FileNotFoundError:
    print("Error: 'heart.csv' not found. Please create the file with your data.")
    exit()

# Features based on your dataset
features_list = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
target = 'HeartDisease'

# Encode Categorical Data
encoders = {}
categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df[features_list]
y = df[target]

# Train Naive Bayes Model
model = GaussianNB()
model.fit(X, y)
print("Naive Bayes Model Trained Successfully on heart.csv!")

# ==========================================
# 2. HTML INTERFACE (Arabic Dark Theme)
# ==========================================
html_content = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="UTF-8" />
  <title>توقّع مرض القلب</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root {
      --bg: #020617;
      --card-bg: #1e293b;
      --primary: #2563eb;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --border: #334155;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; min-height: 100vh;
      font-family: system-ui, -apple-system, sans-serif;
      background: var(--bg); color: var(--text-main);
      display: flex; align-items: center; justify-content: center;
      padding: 20px;
    }
    .container {
      width: 100%; max-width: 900px;
      display: grid; grid-template-columns: 1.5fr 1fr; gap: 20px;
    }
    @media (max-width: 768px) { .container { grid-template-columns: 1fr; } }
    
    .card {
      background: var(--card-bg); border-radius: 16px;
      padding: 24px; border: 1px solid var(--border);
      box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    h1 { margin: 0 0 10px 0; font-size: 1.25rem; color: #60a5fa; }
    p { margin: 0 0 20px 0; font-size: 0.85rem; color: var(--text-muted); }
    
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .full { grid-column: 1 / -1; }
    
    label { display: block; font-size: 0.8rem; margin-bottom: 4px; color: #cbd5e1; }
    input, select {
      width: 100%; padding: 10px; border-radius: 8px;
      border: 1px solid var(--border);
      background: #0f172a; color: white;
      font-size: 0.9rem; outline: none;
    }
    input:focus, select:focus { border-color: var(--primary); }
    
    button {
      width: 100%; margin-top: 20px; padding: 12px;
      background: var(--primary); color: white;
      border: none; border-radius: 8px; font-weight: bold;
      cursor: pointer; transition: 0.2s;
    }
    button:hover { background: #1d4ed8; }
    button:disabled { opacity: 0.7; }

    /* Result Side */
    .result-box {
      text-align: center; display: none;
      padding: 20px; border-radius: 12px;
      margin-top: 20px;
    }
    .safe { background: rgba(34, 197, 94, 0.2); border: 1px solid #22c55e; color: #4ade80; }
    .risk { background: rgba(239, 68, 68, 0.2); border: 1px solid #ef4444; color: #fca5a5; }
    .big-percent { font-size: 2.5rem; font-weight: 800; margin: 10px 0; }
  </style>
</head>
<body>

  <div class="container">
    <!-- INPUT FORM -->
    <div class="card">
      <h1>تحليل صحة القلب (Naive Bayes)</h1>
      <p>أدخل البيانات الطبية للتحليل بواسطة الذكاء الاصطناعي.</p>
      
      <form id="form">
        <div class="grid">
          <div><label>العمر (Age)</label><input id="age" type="number" required></div>
          <div>
            <label>الجنس (Sex)</label>
            <select id="sex"><option value="M">ذكر</option><option value="F">أنثى</option></select>
          </div>
          
          <div class="full">
            <label>نوع ألم الصدر (Chest Pain)</label>
            <select id="cp">
              <option value="ASY">ASY (بدون أعراض)</option>
              <option value="NAP">NAP (ألم غير ذبحية)</option>
              <option value="ATA">ATA (ذبحية غير نمطية)</option>
              <option value="TA">TA (ذبحية نمطية)</option>
            </select>
          </div>

          <div><label>ضغط الدم (RestingBP)</label><input id="bp" type="number" required></div>
          <div><label>الكوليسترول (Cholesterol)</label><input id="chol" type="number" required></div>
          
          <div>
            <label>سكر الصيام (FastingBS)</label>
            <select id="fbs">
              <option value="0">أقل من 120 (0)</option>
              <option value="1">أعلى من 120 (1)</option>
            </select>
          </div>

          <div>
            <label>تخطيط القلب (RestingECG)</label>
            <select id="ecg">
              <option value="Normal">Normal (طبيعي)</option>
              <option value="ST">ST (تغيرات ST)</option>
              <option value="LVH">LVH (تضخم البطين)</option>
            </select>
          </div>

          <div><label>أقصى نبض (MaxHR)</label><input id="maxhr" type="number" required></div>
          
          <div>
            <label>ذبحة مع الجهد (ExAngina)</label>
            <select id="exang"><option value="N">لا (No)</option><option value="Y">نعم (Yes)</option></select>
          </div>

          <div><label>Oldpeak</label><input id="oldpeak" type="number" step="0.1" required></div>
          
          <div>
            <label>ميل ST (ST Slope)</label>
            <select id="slope">
              <option value="Flat">Flat (مسطح)</option>
              <option value="Up">Up (صاعد)</option>
              <option value="Down">Down (هابط)</option>
            </select>
          </div>
        </div>

        <button type="submit" id="btn">تحليل البيانات</button>
      </form>
    </div>

    <!-- RESULT SIDE -->
    <div class="card" style="display:flex; flex-direction:column; justify-content:center;">
      <h2 style="margin-top:0; font-size:1.1rem; color:#94a3b8;">النتيجة</h2>
      <div id="default-msg" style="text-align:center; color:#64748b; margin-top:20px;">
        بانتظار إدخال البيانات...
      </div>
      
      <div id="result-box" class="result-box">
        <div id="result-title" style="font-weight:bold; font-size:1.2rem;"></div>
        <div id="result-percent" class="big-percent"></div>
        <div id="result-desc" style="font-size:0.9rem; opacity:0.9;"></div>
      </div>
    </div>
  </div>

  <script>
    document.getElementById("form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const btn = document.getElementById("btn");
      btn.innerHTML = "جاري التحليل...";
      btn.disabled = true;

      const data = {
        Age: Number(document.getElementById("age").value),
        Sex: document.getElementById("sex").value,
        ChestPainType: document.getElementById("cp").value,
        RestingBP: Number(document.getElementById("bp").value),
        Cholesterol: Number(document.getElementById("chol").value),
        FastingBS: Number(document.getElementById("fbs").value),
        RestingECG: document.getElementById("ecg").value,
        MaxHR: Number(document.getElementById("maxhr").value),
        ExerciseAngina: document.getElementById("exang").value,
        Oldpeak: Number(document.getElementById("oldpeak").value),
        ST_Slope: document.getElementById("slope").value
      };

      try {
        const res = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(data)
        });
        const json = await res.json();
        
        document.getElementById("default-msg").style.display = "none";
        const box = document.getElementById("result-box");
        box.style.display = "block";
        
        const percent = Math.round(json.probability * 100);
        document.getElementById("result-percent").innerText = percent + "%";

        if (json.label === 1) {
          box.className = "result-box risk";
          document.getElementById("result-title").innerText = "خطر محتمل";
          document.getElementById("result-desc").innerText = "النموذج يشير إلى احتمالية وجود مرض القلب.";
        } else {
          box.className = "result-box safe";
          document.getElementById("result-title").innerText = "سليم (خطر منخفض)";
          document.getElementById("result-desc").innerText = "النموذج يشير إلى عدم وجود مؤشرات قوية لمرض القلب.";
        }

      } catch (err) {
        alert("حدث خطأ في الاتصال بالخادم");
        console.error(err);
      } finally {
        btn.innerHTML = "تحليل البيانات";
        btn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""

# ==========================================
# 3. FASTAPI BACKEND
# ==========================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HeartInput(BaseModel):
    Age: int
    Sex: str
    ChestPainType: str
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    RestingECG: str
    MaxHR: int
    ExerciseAngina: str
    Oldpeak: float
    ST_Slope: str

@app.get("/", response_class=HTMLResponse)
def home():
    return html_content

@app.post("/predict")
def predict(data: HeartInput):
    try:
        # Encode categorical inputs using the encoders we trained earlier
        sex_e = encoders['Sex'].transform([data.Sex])[0]
        cp_e = encoders['ChestPainType'].transform([data.ChestPainType])[0]
        ecg_e = encoders['RestingECG'].transform([data.RestingECG])[0]
        ex_e = encoders['ExerciseAngina'].transform([data.ExerciseAngina])[0]
        slope_e = encoders['ST_Slope'].transform([data.ST_Slope])[0]

        # Create input array in exact order of CSV columns
        features = np.array([[
            data.Age,
            sex_e,
            cp_e,
            data.RestingBP,
            data.Cholesterol,
            data.FastingBS,
            ecg_e,
            data.MaxHR,
            ex_e,
            data.Oldpeak,
            slope_e
        ]])

        # Predict
        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1] # Probability of '1' (Heart Disease)

        return {"label": int(prediction), "probability": float(prob)}
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)