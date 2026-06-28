import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BookOpen,
  CheckCircle2,
  ClipboardList,
  History,
  Loader2,
  RefreshCcw,
  UserRound,
} from "lucide-react";
import React, { useEffect, useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_URL || (import.meta.env.DEV ? "http://localhost:5000/api" : "/api");

const defaults = {
  studentReference: "Learner-001",
  code_module: "FFF",
  code_presentation: "2014J",
  studied_credits: 60,
  days_registered_before_start: 30,
  total_clicks: 140,
  active_days: 8,
  days_since_last_activity: 7,
  due_assessments: 2,
  submitted_due_assessments: 1,
  mean_score: 58,
  late_submissions: 1,
  module_presentation_length: 269,
};

const engagedSample = {
  ...defaults,
  studentReference: "Learner-Strong",
  total_clicks: 640,
  active_days: 24,
  days_since_last_activity: 0,
  due_assessments: 2,
  submitted_due_assessments: 2,
  mean_score: 84,
  late_submissions: 0,
};

const atRiskSample = {
  ...defaults,
  studentReference: "Learner-Watch",
  total_clicks: 55,
  active_days: 3,
  days_since_last_activity: 14,
  due_assessments: 3,
  submitted_due_assessments: 1,
  mean_score: 42,
  late_submissions: 1,
};

function NumberField({ label, name, value, onChange, min = 0, max }) {
  return (
    <label className="field" htmlFor={name}>
      <span>{label}</span>
      <input id={name} name={name} type="number" min={min} max={max} value={value} onChange={onChange} />
    </label>
  );
}

function TextField({ label, name, value, onChange }) {
  return (
    <label className="field" htmlFor={name}>
      <span>{label}</span>
      <input id={name} name={name} type="text" value={value} onChange={onChange} />
    </label>
  );
}

function riskCopy(result) {
  if (!result) {
    return {
      title: "No prediction yet",
      body: "Enter the learner's early activity and submit the form.",
    };
  }

  if (result.riskBand === "high") {
    return {
      title: "High dropout risk",
      body: "This learner should be contacted soon.",
    };
  }

  if (result.riskBand === "medium") {
    return {
      title: "Medium dropout risk",
      body: "A support nudge is recommended.",
    };
  }

  if (result.riskBand === "watch") {
    return {
      title: "Watch list",
      body: "Keep monitoring engagement and assessment progress.",
    };
  }

  return {
    title: "Low dropout risk",
    body: "The learner currently looks steady.",
  };
}

function ResultCard({ result, loading }) {
  const copy = riskCopy(result);
  const probability = result ? Math.round(result.dropoutProbability * 100) : 0;
  const band = result?.riskBand || "empty";

  return (
    <aside className="result-card">
      <div className="result-top">
        <span className="label">Prediction</span>
        <span className={`status-pill ${band}`}>{result ? result.riskBand : "Ready"}</span>
      </div>

      <div className="score-row">
        <strong>{loading ? "..." : `${probability}%`}</strong>
        <span>dropout probability</span>
      </div>

      <div className="meter" aria-hidden="true">
        <span className={band} style={{ width: `${probability}%` }} />
      </div>

      <h2>{copy.title}</h2>
      <p>{copy.body}</p>

      {result?.recommendations?.length ? (
        <div className="next-steps">
          {result.recommendations.slice(0, 2).map((item) => (
            <div key={item}>
              <CheckCircle2 size={16} />
              <span>{item}</span>
            </div>
          ))}
        </div>
      ) : null}

      {result?.explanations?.length ? (
        <div className="reasons">
          <h3>Top reasons</h3>
          {result.explanations
            .filter((item) => item.feature !== "explanation_unavailable")
            .slice(0, 4)
            .map((item) => (
              <div key={`${item.feature}-${item.contribution}`}>
                <span>{item.label}</span>
                <b className={item.contribution >= 0 ? "risk" : "safe"}>
                  {item.direction}
                </b>
              </div>
            ))}
        </div>
      ) : null}

      {result?.thresholdStrategy?.strategy ? (
        <p className="policy-note">{result.thresholdStrategy.strategy}</p>
      ) : null}

      <div className="storage-note">
        <History size={15} />
        <span>Recent predictions stay in this session</span>
      </div>
    </aside>
  );
}

function HistoryList({ predictions }) {
  return (
    <section className="history-card">
      <div className="section-heading">
        <History size={18} />
        <h2>Recent</h2>
      </div>
      {predictions.length ? (
        <div className="history-list">
          {predictions.slice(0, 5).map((item) => (
            <article key={item._id || item.createdAt}>
              <div>
                <strong>{item.studentReference || "Learner"}</strong>
                <span>{item.result?.riskBand || "prediction"} risk</span>
              </div>
              <b className={item.result?.riskBand || "watch"}>
                {Math.round((item.result?.dropoutProbability || 0) * 100)}%
              </b>
            </article>
          ))}
        </div>
      ) : (
        <p className="empty">No predictions yet.</p>
      )}
    </section>
  );
}

export default function App() {
  const [form, setForm] = useState(defaults);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const completionRate = useMemo(() => {
    const due = Number(form.due_assessments) || 0;
    const submitted = Number(form.submitted_due_assessments) || 0;
    return due ? Math.min(Math.round((submitted / due) * 100), 100) : 0;
  }, [form.due_assessments, form.submitted_due_assessments]);

  async function loadHistory() {
    try {
      const response = await fetch(`${API_BASE}/predictions`);
      if (response.ok) {
        const data = await response.json();
        setHistory(data.predictions || []);
      }
    } catch {
      setHistory([]);
    }
  }

  useEffect(() => {
    loadHistory();
  }, []);

  function handleChange(event) {
    const { name, value, type } = event.target;
    setForm((current) => ({
      ...current,
      [name]: type === "number" ? Number(value) : value,
    }));
  }

  function buildPayload() {
    const activeDays = Number(form.active_days) || 0;
    const totalClicks = Number(form.total_clicks) || 0;
    const submitted = Number(form.submitted_due_assessments) || 0;
    const due = Number(form.due_assessments) || 0;
    const meanScore = Number(form.mean_score) || 0;
    const lastActivityDay = 30 - (Number(form.days_since_last_activity) || 0);

    return {
      ...form,
      last_activity_day: lastActivityDay,
      first_activity_day: activeDays ? 0 : 30,
      unique_sites_visited: Math.max(1, Math.round(activeDays * 1.5)),
      max_clicks_in_day: activeDays ? Math.ceil(totalClicks / activeDays) : 0,
      std_clicks_per_active_day: activeDays ? Math.round(totalClicks / activeDays / 2) : 0,
      clicks_pre_course: Math.round(totalClicks * 0.18),
      clicks_days_00_07: Math.round(totalClicks * 0.32),
      clicks_days_08_14: Math.round(totalClicks * 0.28),
      clicks_days_15_cutoff: Math.round(totalClicks * 0.22),
      resource_clicks: Math.round(totalClicks * 0.22),
      forum_clicks: Math.round(totalClicks * 0.18),
      homepage_clicks: Math.round(totalClicks * 0.18),
      content_clicks: Math.round(totalClicks * 0.25),
      subpage_clicks: Math.round(totalClicks * 0.12),
      url_clicks: Math.round(totalClicks * 0.05),
      due_weight: due * 10,
      assessment_submissions: submitted,
      on_time_submissions: Math.max(submitted - Number(form.late_submissions || 0), 0),
      min_score: meanScore,
      max_score: meanScore,
      submitted_weight: submitted * 10,
      weighted_score: meanScore,
    };
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildPayload()),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.message || "Prediction failed.");
      }
      setResult(data.prediction.result);
      loadHistory();
    } catch (err) {
      setError(err.message || "Prediction failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app">
      <header className="app-header">
        <div>
          <span>Early dropout detection</span>
          <h1>Online Course Dropout Detector</h1>
        </div>
        <div className="header-stat">
          <Activity size={18} />
          <b>{completionRate}%</b>
          <small>assessment completion</small>
        </div>
      </header>

      <div className="layout">
        <form className="form-card" onSubmit={handleSubmit}>
          <div className="form-title">
            <div>
              <span className="label">Learner details</span>
              <h2>Simple Prediction Form</h2>
            </div>
            <button type="button" className="icon-btn" onClick={() => setForm(defaults)} title="Reset form">
              <RefreshCcw size={18} />
            </button>
          </div>

          <section>
            <div className="section-heading">
              <UserRound size={18} />
              <h3>Learner</h3>
            </div>
            <div className="grid one">
              <TextField label="Learner ID" name="studentReference" value={form.studentReference} onChange={handleChange} />
            </div>
          </section>

          <section>
            <div className="section-heading">
              <BookOpen size={18} />
              <h3>Activity</h3>
            </div>
            <div className="grid two">
              <NumberField label="Total clicks in first 30 days" name="total_clicks" value={form.total_clicks} onChange={handleChange} />
              <NumberField label="Active days" name="active_days" value={form.active_days} onChange={handleChange} />
              <NumberField label="Days since last activity" name="days_since_last_activity" value={form.days_since_last_activity} onChange={handleChange} />
              <NumberField label="Studied credits" name="studied_credits" value={form.studied_credits} onChange={handleChange} />
            </div>
          </section>

          <section>
            <div className="section-heading">
              <ClipboardList size={18} />
              <h3>Assessment</h3>
            </div>
            <div className="grid two">
              <NumberField label="Assessments due" name="due_assessments" value={form.due_assessments} onChange={handleChange} />
              <NumberField label="Assessments submitted" name="submitted_due_assessments" value={form.submitted_due_assessments} onChange={handleChange} />
              <NumberField label="Average score" name="mean_score" value={form.mean_score} onChange={handleChange} max={100} />
              <NumberField label="Late submissions" name="late_submissions" value={form.late_submissions} onChange={handleChange} />
            </div>
          </section>

          <div className="sample-row">
            <button type="button" onClick={() => setForm(engagedSample)}>Use engaged sample</button>
            <button type="button" onClick={() => setForm(atRiskSample)}>Use at-risk sample</button>
          </div>

          {error ? (
            <div className="error">
              <AlertTriangle size={17} />
              <span>{error}</span>
            </div>
          ) : null}

          <button className="submit" type="submit" disabled={loading}>
            {loading ? <Loader2 className="spin" size={18} /> : <ArrowRight size={18} />}
            Predict
          </button>
        </form>

        <div className="side">
          <ResultCard result={result} loading={loading} />
          <HistoryList predictions={history} />
        </div>
      </div>
    </main>
  );
}
