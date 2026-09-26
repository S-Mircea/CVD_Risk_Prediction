// Borough risk context and London averages are rendered in by the server (borough_info.py)
const boroughRiskData = window.HEARTSCAPE.boroughs;
const LONDON_AVG = window.HEARTSCAPE.londonAvg;
const CIRC = 2 * Math.PI * 72;

const LEVELS = {
    'Very Low Risk': { tag: 'VERY LOW', headline: 'Very low risk', tone: '#5FBF8F', soft: '#1F3A2D', desc: 'Well below average. Keep doing what you are doing and stay on top of routine check-ups.' },
    'Low Risk': { tag: 'LOW', headline: 'Low risk', tone: '#5FBF8F', soft: '#1F3A2D', desc: 'Your 10-year risk is low. Continue healthy habits and keep an eye on your environment.' },
    'Moderate Risk': { tag: 'MODERATE', headline: 'Worth acting on', tone: '#F2A33A', soft: '#3D2F17', desc: 'Your risk is moderate. Lifestyle changes can move this number; discuss it with your GP.' },
    'High Risk': { tag: 'HIGH', headline: 'Talk to your GP', tone: '#FF7A6E', soft: '#40211F', desc: 'Several markers are elevated together. A clinical review is worthwhile.' },
    'Very High Risk': { tag: 'VERY HIGH', headline: 'Please seek advice', tone: '#FF7A6E', soft: '#40211F', desc: 'Your estimated risk is high. Book an appointment with a healthcare professional soon.' }
};

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
        el.textContent = Math.round(to * (1 - Math.pow(1 - t, 3))) + '%';
        if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
}

function displayResults(result) {
    const pct = Math.round(result.risk_probability * 100);
    const level = LEVELS[result.risk_level] || LEVELS['Moderate Risk'];

    $('resultCard').classList.remove('empty');
    $('result-time').textContent = 'Updated ' + new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    countUp($('risk-percentage'), pct);

    const arc = $('gauge-arc');
    arc.style.stroke = level.tone;
    requestAnimationFrame(() => { arc.style.strokeDashoffset = CIRC * (1 - pct / 100); });

    const pill = $('risk-level-pill');
    pill.textContent = level.tag;
    pill.style.background = level.soft;
    pill.style.color = level.tone;
    $('risk-label').textContent = level.headline;
    $('risk-description').textContent = level.desc;

    updateFactorBars(level.tone);
    updateEnvironment(result.environmental_data || {});
    updateAdvice(result);
}

function updateFactorBars(tone) {
    const num = (id, fallback) => parseFloat($(id).value) || fallback;
    const radio = (name) => (form.querySelector(`input[name="${name}"]:checked`) || {}).value;
    const clamp = (v) => Math.round(Math.min(Math.max(v, 4), 100));

    const multiplier = (boroughRiskData[$('borough').value] || { multiplier: 1 }).multiplier;
    let life = 20;
    if (radio('smoker') === 'Yes') life += 35;
    if (radio('activity') === 'Low') life += 20; else if (radio('activity') === 'High') life -= 10;
    if (radio('stress') === 'High') life += 15;
    if ($('alcohol').value === 'Heavy') life += 15;
    if (num('sleep_hours', 7) < 6) life += 10;

    const values = {
        age: clamp((num('age', 45) - 30) * 2),
        bp: clamp((num('systolic_bp', 120) - 110) * 1.5 + (radio('high_bp') === 'Yes' ? 20 : 0)),
        chol: clamp((num('cholesterol', 200) - 170) * 0.6),
        env: clamp((multiplier - 0.8) * 200),
        life: clamp(life)
    };
    setTimeout(() => {
        Object.entries(values).forEach(([key, v]) => {
            const bar = $(key + '-bar');
            bar.style.width = v + '%';
            bar.style.background = v >= 50 ? tone : '#8C8880';
            $(key + '-val').textContent = v + '%';
        });
    }, 150);
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
    $('environmental-info').hidden = true;
    $('advice-panel').hidden = true;
    $('submitLabel').textContent = 'Calculate my risk';
});

// A borough passed in the URL (from the Borough data page) is already selected; show its chip
if ($('borough').value) $('borough').dispatchEvent(new Event('change'));
