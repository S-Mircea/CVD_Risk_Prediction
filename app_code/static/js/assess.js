// Borough risk context and London averages are rendered in by the server (borough_info.py)
const boroughRiskData = window.HEARTSCAPE.boroughs;
const LONDON_AVG = window.HEARTSCAPE.londonAvg;
const CIRC = 2 * Math.PI * 72;

// Bands follow clinical practice: NICE offers statins from 10% 10-year risk
const LEVELS = {
    'Low Risk': { tag: 'LOW', headline: 'Low risk', tone: '#5FBF8F', soft: '#1F3A2D', desc: 'Under 5% over 10 years. Keep up healthy habits and re-check every few years.' },
    'Moderate Risk': { tag: 'MODERATE', headline: 'Worth keeping an eye on', tone: '#F2A33A', soft: '#3D2F17', desc: 'Between 5% and 10%. Lifestyle changes can bring this down; mention it at your next check-up.' },
    'High Risk': { tag: 'HIGH', headline: 'Talk to your GP', tone: '#FF7A6E', soft: '#40211F', desc: 'Between 10% and 20%. At this level NICE guidance suggests discussing treatment such as statins.' },
    'Very High Risk': { tag: 'VERY HIGH', headline: 'Please seek advice soon', tone: '#FF7A6E', soft: '#40211F', desc: '20% or more over 10 years. Book an appointment with your GP to review your risk factors.' }
};

const fmtPct = (v) => (v < 10 ? v.toFixed(1) : Math.round(v)) + '%';

const $ = (id) => document.getElementById(id);
const form = $('assessmentForm');

function tierFor(multiplier) {
    if (multiplier >= 1.10) return { cls: 'high', label: 'Higher-exposure borough' };
    if (multiplier >= 1.00) return { cls: 'medium', label: 'Medium-exposure borough' };
    return { cls: 'low', label: 'Lower-exposure borough' };
}

$('borough').addEventListener('change', (e) => {
    const chip = $('borough-chip');
    const data = boroughRiskData[e.target.value];
    if (!data) { chip.hidden = true; return; }
    const tier = tierFor(data.multiplier);
    chip.className = 'chip ' + tier.cls;
    chip.textContent = `${tier.label} · ×${data.multiplier.toFixed(2)} multiplier`;
    chip.hidden = false;
});

function missingFields() {
    const missing = [];
    form.querySelectorAll('[required]').forEach((el) => {
        if (el.type === 'radio') {
            if (!form.querySelector(`input[name="${el.name}"]:checked`)) missing.push(el.closest('fieldset').querySelector('legend').textContent);
        } else if (!el.value) {
            missing.push(form.querySelector(`label[for="${el.id}"]`).childNodes[0].textContent.trim());
        }
    });
    return missing;
}

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const err = $('formError');
    const missing = missingFields();
    if (missing.length) {
        err.textContent = 'Please complete: ' + missing.join(', ') + '.';
        err.classList.add('show');
        return;
    }
    err.classList.remove('show');

    const btn = $('submitBtn');
    btn.disabled = true;
    btn.classList.add('loading');
    $('submitLabel').textContent = 'Analysing…';

    try {
        const response = await fetch('/assess_risk', { method: 'POST', body: new FormData(form) });
        const data = await response.json();
        if (data.success) {
            displayResults(data.result);
            if (window.matchMedia('(max-width: 1180px)').matches) $('result').scrollIntoView({ behavior: 'smooth' });
        } else {
            err.textContent = 'Something went wrong: ' + data.error;
            err.classList.add('show');
        }
    } catch (error) {
        err.textContent = 'Could not reach the server: ' + error.message;
        err.classList.add('show');
    } finally {
        btn.disabled = false;
        btn.classList.remove('loading');
        $('submitLabel').textContent = 'Recalculate';
    }
});

function countUp(el, to) {
    const start = performance.now();
    const dur = 1200;
    const step = (now) => {
        const t = Math.min((now - start) / dur, 1);
        el.textContent = fmtPct(to * (1 - Math.pow(1 - t, 3)));
        if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
}

function displayResults(result) {
    const pct = result.risk_probability * 100;
    const level = LEVELS[result.risk_level] || LEVELS['Moderate Risk'];

    $('resultCard').classList.remove('empty');
    $('result-time').textContent = 'Updated ' + new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    countUp($('risk-percentage'), pct);

    const arc = $('gauge-arc');
    arc.style.stroke = level.tone;
    requestAnimationFrame(() => { arc.style.strokeDashoffset = CIRC * (1 - Math.min(pct, 100) / 100); });

    const pill = $('risk-level-pill');
    pill.textContent = level.tag;
    pill.style.background = level.soft;
    pill.style.color = level.tone;
    $('risk-label').textContent = level.headline;
    $('risk-description').textContent = level.desc;

    const env = result.environmental_data || {};
    const envPoints = pct - result.base_risk * 100;
    const base = $('risk-base');
    base.textContent = `Framingham ${fmtPct(result.base_risk * 100)} · borough adjustment ${envPoints >= 0 ? '+' : '−'}${Math.abs(envPoints).toFixed(1)} pts`;
    base.hidden = false;

    const notes = [];
    if (result.age_note) notes.push(result.age_note);
    if (result.hdl_assumed) notes.push('No HDL value given, so a typical 50 mg/dL (1.3 mmol/L) was assumed. Adding yours makes the estimate more accurate.');
    $('risk-note').textContent = notes.join(' ');
    $('risk-note').hidden = notes.length === 0;

    renderDrivers(result.drivers || [], level.tone);
    updateEnvironment(env);
    updateAdvice(result);
}

function renderDrivers(drivers, tone) {
    const box = $('drivers');
    box.innerHTML = '';
    const shown = drivers.filter((d) => Math.abs(d.points) >= 0.05).sort((a, b) => b.points - a.points);
    const max = Math.max(1, ...shown.map((d) => Math.abs(d.points)));
    if (!shown.length) {
        const p = document.createElement('p');
        p.className = 'result-desc';
        p.textContent = 'None of your answers add measurable risk beyond your age and sex.';
        box.appendChild(p);
        return;
    }
    shown.forEach((d) => {
        const row = document.createElement('div');
        row.className = 'factor';
        const name = document.createElement('span');
        name.textContent = d.name;
        const bar = document.createElement('div');
        bar.className = 'bar';
        const fill = document.createElement('i');
        if (d.points < 0) fill.className = 'neg';
        else fill.style.background = d.points >= 2 ? tone : '#8C8880';
        bar.appendChild(fill);
        const val = document.createElement('span');
        val.className = 'mono';
        val.textContent = (d.points > 0 ? '+' : '−') + Math.abs(d.points).toFixed(1);
        row.append(name, bar, val);
        box.appendChild(row);
        setTimeout(() => { fill.style.width = (Math.abs(d.points) / max * 100) + '%'; }, 150);
    });
}

function updateEnvironment(env) {
    const borough = env.borough || $('borough').value;
    const data = boroughRiskData[borough];
    if (!borough) return;
    $('env-borough').textContent = borough;
    $('env-mult').textContent = data ? `×${data.multiplier.toFixed(2)} risk multiplier` : 'Environmental profile';
    const fmt = (v, d = 1) => (typeof v === 'number' && !isNaN(v)) ? v.toFixed(d) : '–';
    $('pm25-val').textContent = fmt(env.pm25);
    $('no2-val').textContent = fmt(env.no2);
    $('green-val').textContent = fmt(env.green_space);
    $('pm25-card').classList.toggle('warn', env.pm25 > LONDON_AVG.pm25);
    $('no2-card').classList.toggle('warn', env.no2 > LONDON_AVG.no2);
    $('env-risk-text').textContent = data ? data.description : '';
    $('environmental-info').hidden = false;
}

function updateAdvice(result) {
    const adviceEl = $('llm-advice');
    const recs = $('recs');
    recs.innerHTML = '';
    if (result.llm_advice) {
        adviceEl.textContent = result.llm_advice;
        adviceEl.hidden = false;
        $('advice-source').textContent = result.llm_available ? 'Llama · local' : 'Fallback';
    } else {
        adviceEl.hidden = true;
        $('advice-source').textContent = 'Clinical guidance';
    }
    (result.recommendations || []).forEach((text) => {
        const li = document.createElement('li');
        li.textContent = text;
        recs.appendChild(li);
    });
    $('advice-panel').hidden = !(result.llm_advice || (result.recommendations || []).length);
}

$('resetBtn').addEventListener('click', () => {
    form.reset();
    $('borough-chip').hidden = true;
    $('formError').classList.remove('show');
    $('resultCard').classList.add('empty');
    $('risk-percentage').textContent = '--';
    $('result-time').textContent = 'Awaiting answers';
    $('gauge-arc').style.strokeDashoffset = CIRC;
    $('gauge-arc').style.stroke = '#8C8880';
    const pill = $('risk-level-pill');
    pill.textContent = 'PENDING'; pill.style.background = ''; pill.style.color = '';
    $('risk-label').textContent = 'Ready when you are';
    $('risk-description').textContent = 'Complete the five short sections and we will estimate your 10-year cardiovascular risk, adjusted for your borough.';
    $('risk-base').hidden = true;
    $('risk-note').hidden = true;
    $('drivers').innerHTML = '';
    $('environmental-info').hidden = true;
    $('advice-panel').hidden = true;
    $('submitLabel').textContent = 'Calculate my risk';
});

// A borough passed in the URL (from the Borough data page) is already selected; show its chip
if ($('borough').value) $('borough').dispatchEvent(new Event('change'));
