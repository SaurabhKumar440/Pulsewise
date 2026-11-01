import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { Activity, ArrowRight, Bell, CheckCircle2, ChevronRight, CircleHelp, ClipboardCheck, HeartPulse, Info, LogOut, Menu, ShieldCheck, Sparkles, UserRound, X } from "lucide-react";
import "./styles.css";

const initialForm = { gender: "Female", age: 45, bmi: 28, bmiUnknown: false, hypertension: false, heartDisease: false, smokingHistory: "never", hasLabs: false, hba1c: 5.7, bloodGlucose: 110 };

async function readJsonResponse(response) {
  const body = await response.text();
  if (!body.trim()) {
    throw new Error(`The server returned an empty response (HTTP ${response.status}). Please make sure the API is running.`);
  }
  let data;
  try {
    data = JSON.parse(body);
  } catch {
    throw new Error(`The server returned an invalid response (HTTP ${response.status}).`);
  }
  if (!response.ok) {
    throw new Error(data.message || `Request failed (HTTP ${response.status}).`);
  }
  return data;
}

function App() {
  const [user, setUser] = useState(() => JSON.parse(localStorage.getItem("diabetesUser") || "null"));
  const [login, setLogin] = useState({ email: "", password: "" });
  const [loginError, setLoginError] = useState("");
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [active, setActive] = useState("Overview");
  const [mobileNav, setMobileNav] = useState(false);

  const update = (key, value) => setForm((current) => ({ ...current, [key]: value }));
  const submitLogin = async (event) => {
    event.preventDefault(); setLoginError("");
    try {
      const response = await fetch("/api/auth/login", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(login) });
      const data = await readJsonResponse(response);
      localStorage.setItem("diabetesUser", JSON.stringify(data.user)); setUser(data.user);
    } catch (error) { setLoginError(error.message); }
  };
  const assess = async (event) => {
    event.preventDefault(); setLoading(true);
    try {
      const response = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...form, heartDisease: form.heartDisease, featureSet: form.hasLabs ? "full" : "screening" }) });
      const data = await readJsonResponse(response);
      setResult(data); setActive("Assessment");
    } catch (error) { window.alert(error.message); } finally { setLoading(false); }
  };
  if (!user) return <Login login={login} setLogin={setLogin} submitLogin={submitLogin} error={loginError} />;
  return <div className="app-shell">
    <aside className={`sidebar ${mobileNav ? "open" : ""}`}>
      <div className="brand"><div className="brand-mark"><HeartPulse size={22} /></div><div><strong>Pulsewise</strong><small>Health intelligence</small></div><button className="icon-btn close-nav" onClick={() => setMobileNav(false)}><X size={20} /></button></div>
      <div className="nav-label">Workspace</div>
      {["Overview", "Assessment", "History"].map((item) => <button key={item} className={`nav-item ${active === item ? "active" : ""}`} onClick={() => { setActive(item); setMobileNav(false); }}><Activity size={18} />{item}<ChevronRight size={15} /></button>)}
      <div className="nav-label settings-label">Support</div>
      <button className="nav-item"><CircleHelp size={18} />How it works</button>
      <div className="sidebar-bottom"><div className="privacy"><ShieldCheck size={18} /><div><strong>Your privacy matters</strong><span>Data stays in this session.</span></div></div><button className="nav-item logout" onClick={() => { localStorage.removeItem("diabetesUser"); setUser(null); }}><LogOut size={18} />Sign out</button></div>
    </aside>
    <main className="main-content">
      <header className="topbar"><button className="icon-btn menu-btn" onClick={() => setMobileNav(true)}><Menu size={22} /></button><div className="crumb"><span>Workspace</span><ChevronRight size={14} /><strong>{active}</strong></div><div className="top-actions"><button className="icon-btn notification"><Bell size={19} /><i /></button><div className="avatar">{user.name[0].toUpperCase()}</div><span className="user-name">{user.name}</span></div></header>
      <section className="page">
        <div className="welcome"><div><p className="eyebrow">SATURDAY, OCTOBER 31, 2026</p><h1>{active === "Overview" ? <>Good morning, <em>{user.name.split(" ")[0]}</em></> : active}</h1><p className="subhead">A clearer view of your metabolic health starts here.</p></div><button className="primary-btn" onClick={() => setActive("Assessment")}><Sparkles size={17} />Start assessment</button></div>
        {active === "Overview" && <Overview result={result} setActive={setActive} />}
        {active === "Assessment" && <Assessment form={form} update={update} assess={assess} loading={loading} result={result} />}
        {active === "History" && <History result={result} setActive={setActive} />}
      </section>
    </main>
  </div>;
}

function Login({ login, setLogin, submitLogin, error }) {
  return <div className="login-page"><div className="login-art"><div className="art-glow" /><div className="login-brand"><div className="brand-mark"><HeartPulse size={22} /></div><strong>Pulsewise</strong></div><div className="art-copy"><p className="eyebrow">PROACTIVE HEALTH, MADE SIMPLE</p><h1>Small signals.<br /><em>Better decisions.</em></h1><p>Understand your diabetes risk with a thoughtful, science-led screening experience.</p></div><div className="testimonial"><div className="quote">“The clarity I needed to take the next step.”</div><small>— Pulsewise member</small></div></div><div className="login-panel"><div className="mobile-login-brand"><div className="brand-mark"><HeartPulse size={22} /></div><strong>Pulsewise</strong></div><div className="login-form"><p className="eyebrow">WELCOME BACK</p><h2>Your health, at a glance.</h2><p className="form-copy">Sign in to continue to your personal health dashboard.</p><form onSubmit={submitLogin}><label>Email address<input type="email" placeholder="you@example.com" value={login.email} onChange={(e) => setLogin({ ...login, email: e.target.value })} required /></label><label>Password<div className="password-wrap"><input type="password" placeholder="Enter your password" value={login.password} onChange={(e) => setLogin({ ...login, password: e.target.value })} required minLength="6" /><span>Forgot?</span></div></label>{error && <div className="error">{error}</div>}<button className="primary-btn full" type="submit">Sign in <ArrowRight size={17} /></button></form><div className="demo-hint">Demo mode · any valid email and 6+ character password</div></div></div></div>;
}

function Overview({ result, setActive }) {
  return <><div className="stat-grid"><div className="stat-card accent"><div className="stat-icon"><Activity size={20} /></div><div className="stat-label">Latest assessment</div><strong>{result ? `${(result.probability * 100).toFixed(1)}%` : "Not yet taken"}</strong><span>{result ? (result.highRisk ? "Follow-up recommended" : "Within lower-risk range") : "Start your first check-in"}</span></div><div className="stat-card"><div className="stat-icon lavender"><ClipboardCheck size={20} /></div><div className="stat-label">Assessment status</div><strong>{result ? "Complete" : "Ready when you are"}</strong><span>{result ? "Updated just now" : "Takes about 2 minutes"}</span></div><div className="stat-card"><div className="stat-icon peach"><ShieldCheck size={20} /></div><div className="stat-label">Your data</div><strong>Private</strong><span>Stored only for this session</span></div></div><div className="dashboard-grid"><div className="welcome-card"><div className="card-kicker">YOUR NEXT BEST STEP</div><h2>A healthier future starts with knowing your baseline.</h2><p>Our assessment uses your everyday health information to estimate risk and help you decide whether a clinical blood test could be useful.</p><button className="text-btn" onClick={() => setActive("Assessment")}>Take assessment <ArrowRight size={16} /></button><div className="line-art"><div /><div /><div /></div></div><div className="insight-card"><div className="card-heading"><h3>How Pulsewise works</h3><Info size={17} /></div><Step n="01" title="Share your context" text="A few details about your health and lifestyle." /><Step n="02" title="Get a clear signal" text="A model trained on 79,000+ adult records." /><Step n="03" title="Choose your next step" text="Use the result to have a better conversation." /></div></div><div className="disclaimer"><Info size={16} /><span><strong>Educational screening only.</strong> Pulsewise is not a diagnosis or a substitute for professional medical advice.</span></div></>;
}
function Step({ n, title, text }) { return <div className="step"><span>{n}</span><div><strong>{title}</strong><p>{text}</p></div><ChevronRight size={15} /></div>; }

function Assessment({ form, update, assess, loading, result }) {
  return <div className="assessment-layout"><form className="assessment-form" onSubmit={assess}><div className="section-title"><div><p className="eyebrow">STEP 1 OF 1</p><h2>Tell us about yourself</h2><p>Use your most recent information. It does not need to be perfect.</p></div><span className="secure"><ShieldCheck size={15} />Secure session</span></div><div className="input-section"><h3>About you</h3><div className="segmented"><span className="field-label">Sex</span><div>{["Female", "Male"].map((x) => <button type="button" className={form.gender === x ? "selected" : ""} onClick={() => update("gender", x)} key={x}>{x}</button>)}</div></div><div className="two-col"><Field label="Age" suffix="years"><input type="number" min="18" max="80" value={form.age} onChange={(e) => update("age", e.target.value)} /></Field><Field label="BMI" suffix="kg/m²"><input type="number" min="12" max="60" step=".1" value={form.bmi} disabled={form.bmiUnknown} onChange={(e) => update("bmi", e.target.value)} /></Field></div><label className="check-row"><input type="checkbox" checked={form.bmiUnknown} onChange={(e) => update("bmiUnknown", e.target.checked)} /><span>BMI is not known</span><small>We'll use a typical value</small></label></div><div className="input-section"><h3>Health context</h3><Toggle label="Diagnosed hypertension" checked={form.hypertension} onChange={(v) => update("hypertension", v)} /><Toggle label="Diagnosed heart disease" checked={form.heartDisease} onChange={(v) => update("heartDisease", v)} /><div className="select-row"><label>Smoking history<select value={form.smokingHistory} onChange={(e) => update("smokingHistory", e.target.value)}><option value="never">Never smoked</option><option value="former">Former smoker</option><option value="current">Current smoker</option><option value="unknown">Not known</option></select></label></div></div><div className="input-section lab-section"><div className="lab-heading"><div><h3>Recent blood test results</h3><p>Have your HbA1c and glucose values handy?</p></div><button type="button" className={`switch ${form.hasLabs ? "on" : ""}`} onClick={() => update("hasLabs", !form.hasLabs)}><i /></button></div>{form.hasLabs && <div className="two-col"><Field label="HbA1c" suffix="%"><input type="number" min="3.5" max="9" step=".1" value={form.hba1c} onChange={(e) => update("hba1c", e.target.value)} /></Field><Field label="Blood glucose" suffix="mg/dL"><input type="number" min="80" max="300" value={form.bloodGlucose} onChange={(e) => update("bloodGlucose", e.target.value)} /></Field></div>}</div><button className="primary-btn submit-btn" disabled={loading}>{loading ? "Calculating..." : "Calculate my risk"} <ArrowRight size={17} /></button></form><ResultCard result={result} /></div>;
}
function Field({ label, suffix, children }) { return <label className="field"><span>{label}</span><div>{children}<small>{suffix}</small></div></label>; }
function Toggle({ label, checked, onChange }) { return <div className="toggle-row"><span>{label}</span><button type="button" className={`switch ${checked ? "on" : ""}`} onClick={() => onChange(!checked)}><i /></button></div>; }
function ResultCard({ result }) { if (!result) return <div className="result-empty"><div className="empty-icon"><Sparkles size={24} /></div><h3>Your result will appear here</h3><p>Complete the assessment to see your estimated risk, contributing factors, and a suggested next step.</p></div>; return <div className="result-panel"><div className="result-top"><p className="eyebrow">YOUR RESULT</p><span className={`result-pill ${result.highRisk ? "high" : "low"}`}>{result.highRisk ? "Needs follow-up" : "Lower risk"}</span></div><h2>{(result.probability * 100).toFixed(1)}<small>%</small></h2><p className="result-description">Estimated probability based on your {result.featureSet === "full" ? "health and lab" : "health"} information.</p><div className="meter"><div style={{ width: `${Math.min(result.probability * 100 * 1.8, 100)}%` }} /></div><div className="meter-labels"><span>Lower</span><span>Threshold {(result.threshold * 100).toFixed(1)}%</span><span>Higher</span></div><div className="result-note"><Info size={16} /><span>{result.highRisk ? "This is a screening flag, not a diagnosis. Consider arranging a blood test and speaking with a clinician." : "Keep up healthy habits and stay on a routine screening schedule."}</span></div><h3 className="drivers-title">Main contributing factors</h3>{result.drivers.map((driver) => <div className="driver" key={driver.name}><span>{driver.name}</span><div className="driver-bar"><i className={driver.value > 0 ? "positive" : ""} style={{ width: `${Math.min(Math.abs(driver.value) * 100, 100)}%` }} /></div><b className={driver.value > 0 ? "positive-text" : ""}>{driver.value > 0 ? "+" : ""}{driver.value.toFixed(2)}</b></div>)}</div>; }
function History({ result, setActive }) { return <div className="history-card"><div className="card-heading"><div><p className="eyebrow">PRIVATE SESSION HISTORY</p><h2>Your assessments</h2></div><button className="primary-btn" onClick={() => setActive("Assessment")}>New assessment</button></div>{result ? <div className="history-row"><div className="history-date"><strong>Today</strong><span>{result.featureSet === "full" ? "Full assessment" : "Screening assessment"}</span></div><div className={`result-pill ${result.highRisk ? "high" : "low"}`}>{result.highRisk ? "Needs follow-up" : "Lower risk"}</div><strong>{(result.probability * 100).toFixed(1)}%</strong><ChevronRight size={17} /></div> : <div className="empty-history"><ClipboardCheck size={28} /><h3>No assessments yet</h3><p>Your completed assessments will appear here.</p></div>}</div>; }

createRoot(document.getElementById("root")).render(<App />);
