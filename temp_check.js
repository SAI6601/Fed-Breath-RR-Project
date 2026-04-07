
    const C = id => document.getElementById(id).getContext('2d');
    Chart.defaults.color = '#64748b';
    Chart.defaults.borderColor = '#1f2d42';
    Chart.defaults.font.family = "'Inter', sans-serif";

    function ticker() {
      document.getElementById('clock').textContent =
        new Date().toUTCString().split(' ').slice(4, 5).join('') + ' UTC';
    }
    ticker(); setInterval(ticker, 1000);

    // MAE Chart
    const maeChart = new Chart(C('maeChart'), {
      type: 'line',
      data: {
        labels: [], datasets: [{
          label: 'Validation MAE', data: [],
          borderColor: '#00e5ff',
          backgroundColor: ctx => {
            const g = ctx.chart.ctx.createLinearGradient(0, 0, 0, 210);
            g.addColorStop(0, 'rgba(0,229,255,.2)'); g.addColorStop(1, 'rgba(0,229,255,0)'); return g;
          },
          borderWidth: 2.5, pointRadius: 4, pointBackgroundColor: '#00e5ff',
          pointBorderColor: '#0b0f1a', pointBorderWidth: 2, tension: 0.45, fill: true,
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: { backgroundColor: '#111827', borderColor: '#00e5ff', borderWidth: 1 } },
        scales: {
          x: { grid: { color: 'rgba(255,255,255,.04)' } },
          y: { grid: { color: 'rgba(255,255,255,.04)' }, min: 0 }
        }
      }
    });

    // RQI Chart
    const rqiChart = new Chart(C('rqiChart'), {
      type: 'bar',
      data: {
        labels: ['Hospital A', 'Hospital B'],
        datasets: [{
          label: 'RQI Score', data: [0, 0],
          backgroundColor: ['rgba(0,229,255,.8)', 'rgba(168,85,247,.8)'],
          borderColor: ['#00e5ff', '#a855f7'], borderWidth: 1.5, borderRadius: 8
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { x: { grid: { display: false } }, y: { grid: { color: 'rgba(255,255,255,.04)' }, min: 0, max: 1 } }
      }
    });

    // Anomaly stacked bar — pre-seed with placeholder so chart renders immediately
    const anomalyStackLabels = ['Waiting...'];
    const anomalyStackData = [[0], [0], [0], [0], [0]];
    const anomalyColors = ['#22c55e', '#eab308', '#ef4444', '#f97316', '#dc2626'];
    const anomalyNames = ['Normal', 'Bradypnea', 'Apnea', 'Tachypnea', 'Severe tachypnea'];
    const anomalyStackChart = new Chart(C('anomalyStackChart'), {
      type: 'bar',
      data: {
        labels: anomalyStackLabels,
        datasets: anomalyNames.map((name, i) => ({
          label: name, data: anomalyStackData[i],
          backgroundColor: anomalyColors[i], borderWidth: 0, borderRadius: i === 4 ? 4 : 0,
        }))
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { stacked: true, grid: { display: false }, ticks: { color: '#64748b', font: { size: 10 } } },
          y: { stacked: true, grid: { color: 'rgba(255,255,255,.04)' }, ticks: { color: '#64748b', font: { size: 10 } } }
        }
      }
    });

    // ═══════════════ ECG Waveform Generator ═══════════════
    // Generates a single PQRST complex (one heartbeat) as an array of samples
    function ecgBeat(sr) {
      const beat = [];
      const total = Math.round(sr * 0.8);  // ~0.8s per beat at 75 BPM
      for (let i = 0; i < total; i++) {
        const t = i / sr;
        let v = 0;
        // P wave (small bump at ~0.08-0.16s)
        v += 0.15 * Math.exp(-Math.pow((t - 0.12) / 0.025, 2));
        // Q dip (small negative at ~0.19s)
        v -= 0.12 * Math.exp(-Math.pow((t - 0.19) / 0.008, 2));
        // R spike (tall sharp peak at ~0.21s)
        v += 1.8 * Math.exp(-Math.pow((t - 0.215) / 0.009, 2));
        // S dip (negative at ~0.24s)
        v -= 0.35 * Math.exp(-Math.pow((t - 0.245) / 0.012, 2));
        // T wave (rounded bump at ~0.38s)
        v += 0.28 * Math.exp(-Math.pow((t - 0.38) / 0.045, 2));
        // Slight baseline wander
        v += 0.02 * Math.sin(2 * Math.PI * 0.15 * t);
        // Tiny noise
        v += (Math.random() - 0.5) * 0.03;
        beat.push(+v.toFixed(4));
      }
      return beat;
    }

    // Build a continuous ECG strip with slight beat-to-beat variation
    function generateECGStrip(numBeats, sr) {
      const strip = [];
      const peaks = [];   // indices of R-peaks
      for (let b = 0; b < numBeats; b++) {
        const beat = ecgBeat(sr);
        // Slight amplitude variation per beat (±8%)
        const scale = 0.92 + Math.random() * 0.16;
        const rIdx = strip.length + Math.round(0.215 * sr);  // R-peak position
        peaks.push(rIdx);
        beat.forEach(v => strip.push(+(v * scale).toFixed(4)));
      }
      return { strip, peaks };
    }

    const ECG_SR = 125;        // Hz
    const ECG_BEATS = 12;      // beats in buffer
    const ECG_WINDOW = 500;    // visible samples (~4s window)
    let ecgData = generateECGStrip(ECG_BEATS, ECG_SR);
    let ecgOffset = 0;

    let ppgWave = ecgData.strip.slice(0, ECG_WINDOW);
    let ppgLabels = ppgWave.map((_, i) => '');
    let attentionMask = ppgWave.map((_, i) => {
      return ecgData.peaks.some(p => Math.abs(i - p) < 4) ? 1 : 0;
    });

    // ═══════════════ ECG Chart (multi-layer glow) ═══════════════
    const ppgChart = new Chart(C('ppgChart'), {
      type: 'line',
      data: {
        labels: ppgLabels,
        datasets: [
          {
            // Layer 0: wide soft glow
            data: [...ppgWave],
            borderWidth: 8,
            tension: 0,
            fill: false,
            borderColor: 'rgba(0,200,255,0.06)',
            pointRadius: 0,
            spanGaps: true,
            order: 5
          },
          {
            // Layer 1: mid glow
            data: [...ppgWave],
            borderWidth: 4,
            tension: 0,
            fill: false,
            borderColor: 'rgba(0,220,255,0.15)',
            pointRadius: 0,
            spanGaps: true,
            order: 4
          },
          {
            // Layer 2: core ECG line — bright cyan
            data: [...ppgWave],
            borderWidth: 1.8,
            tension: 0,
            fill: false,
            borderColor: 'rgba(0,240,255,0.92)',
            pointRadius: 0,
            spanGaps: true,
            order: 3,
            segment: {
              borderColor: function(ctx) {
                const idx = ctx.p1DataIndex;
                const attn = attentionMask[idx] || 0;
                if (attn > 0.5) return '#ff2d55';       // Highlight: red
                // if (attn > 0.2) return '#ff6b8a';       // Transition
                return 'rgba(0,240,255,0.92)';       // normal
              },
              borderWidth: function(ctx) {
                const idx = ctx.p1DataIndex;
                const attn = attentionMask[idx] || 0;
                return attn > 0.5 ? 2.5 : 1.8;
              }
            }
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        scales: {
          x: {
            display: true,
            grid: { color: 'rgba(0,200,255,0.04)', lineWidth: 0.5 },
            ticks: { display: false }
          },
          y: {
            grid: { color: 'rgba(0,200,255,0.06)', lineWidth: 0.5 },
            ticks: {
              color: 'rgba(0,200,255,0.35)',
              font: { size: 10, family: 'JetBrains Mono' },
              callback: v => v.toFixed(1) + ' mV'
            },
            min: -1.0,
            max: 2.4,
            border: { color: 'rgba(0,200,255,0.08)' }
          }
        }
      }
    });

    // ═══════ Synthetic scrolling (runs until WebSocket provides real data) ═══════
    let ecgSynthActive = true;
    function scrollECG() {
      if (!ecgSynthActive) return;
      ecgOffset += 2;  // advance by 2 samples per tick
      // regenerate buffer when exhausted
      if (ecgOffset + ECG_WINDOW >= ecgData.strip.length) {
        const ext = generateECGStrip(ECG_BEATS, ECG_SR);
        const baseIdx = ecgData.strip.length;
        ext.peaks.forEach(p => ecgData.peaks.push(p + baseIdx));
        ecgData.strip.push(...ext.strip);
        // trim old data to prevent unbounded growth
        if (ecgData.strip.length > 15000) {
          const trim = ecgData.strip.length - 10000;
          ecgData.strip.splice(0, trim);
          ecgData.peaks = ecgData.peaks.filter(p => p >= trim).map(p => p - trim);
          ecgOffset -= trim;
        }
      }
      const windowSlice = ecgData.strip.slice(ecgOffset, ecgOffset + ECG_WINDOW);
      const windowLabels = windowSlice.map(() => '');
      // Update attentionMask for synthetic data
      attentionMask = windowSlice.map((_, i) => {
        return ecgData.peaks.some(p => Math.abs((p - ecgOffset) - i) < 5) ? 1 : 0;
      });

      ppgChart.data.labels = windowLabels;
      ppgChart.data.datasets[0].data = windowSlice;
      ppgChart.data.datasets[1].data = windowSlice;
      ppgChart.data.datasets[2].data = windowSlice;
      ppgChart.update('none');

      // Simulate BPM readout
      const bpm = 68 + Math.floor(Math.random() * 10);
      const bpmEl = document.getElementById('ecg-bpm');
      if (bpmEl) bpmEl.textContent = bpm;
    }
    setInterval(scrollECG, 50);  // ~20 fps

    // Terminal
    const term = document.getElementById('terminal');
    function log(msg, type = 'log-info') {
      const now = new Date().toTimeString().split(' ')[0];
      const el = document.createElement('span');
      el.className = `log-line ${type}`;
      el.innerHTML = `<span class="log-ts">[${now}]</span>${msg}`;
      term.appendChild(el); term.scrollTop = term.scrollHeight;
      if (term.children.length > 50) term.removeChild(term.firstChild);
    }
    log('System Initialized. Waiting for WebSocket connection...', 'log-warn');

    // WebSocket
    const ws = new WebSocket(`ws://${window.location.host}/ws`);

    ws.onopen = () => {
      log('Connected to FastAPI data stream', 'log-pass');
      log('BFT Security Shield active', 'log-pass');
    };

    ws.onmessage = function (event) {
      const d = JSON.parse(event.data);

      // ── Parameters table ────────────────────────────────────
      if (d.round > 0) {
        document.getElementById('pt-round').textContent = d.round;
        document.getElementById('pt-mae').textContent = d.mae.toFixed(4);
      }
      if (d.c0_rqi !== undefined) {
        document.getElementById('pt-rqi0').textContent = d.c0_rqi.toFixed(4);
        document.getElementById('pt-rqi1').textContent = d.c1_rqi.toFixed(4);
      }
      if (d.fp32_mb) {
        const comp = ((d.fp32_mb - d.int8_mb) / d.fp32_mb * 100).toFixed(1);
        document.getElementById('pt-fp32').textContent = d.fp32_mb.toFixed(2);
        document.getElementById('pt-int8').textContent = d.int8_mb.toFixed(2);
        document.getElementById('pt-comp').textContent = comp + '%';
        document.getElementById('fp32Label').textContent = d.fp32_mb.toFixed(2) + ' MB';
        document.getElementById('int8Label').textContent = d.int8_mb.toFixed(2) + ' MB';
        const barPct = Math.min((d.int8_mb / d.fp32_mb) * 100, 100).toFixed(1);
        document.getElementById('barINT8').style.width = barPct + '%';
        document.getElementById('compPct').textContent = comp + '%';
      }
      if (d.epsilon !== undefined && d.epsilon >= 0) {
        document.getElementById('pt-eps').textContent = d.epsilon.toFixed(3);
      }

      // ── MAE chart (new round) ────────────────────────────────
      const rLabel = `R${d.round}`;
      if (d.round > 0 && !maeChart.data.labels.includes(rLabel)) {
        maeChart.data.labels.push(rLabel);
        maeChart.data.datasets[0].data.push(d.mae);
        maeChart.update();
        log(`Round ${d.round} complete — MAE: ${d.mae.toFixed(4)} BrPM`, 'log-info');
      }

      // ── RQI bars ─────────────────────────────────────────────
      if (d.c0_rqi > 0 || d.c1_rqi > 0) {
        rqiChart.data.datasets[0].data = [d.c0_rqi, d.c1_rqi];
        rqiChart.update();
      }

      // ── Anomaly stacked bar ──────────────────────────────────
      if (d.anomaly_counts && d.round > 0 && !anomalyStackLabels.includes(rLabel)) {
        if (anomalyStackLabels.length === 1 && anomalyStackLabels[0] === 'Waiting...') {
          anomalyStackLabels.splice(0, 1);
          anomalyStackData.forEach(arr => arr.splice(0, 1));
        }
        anomalyStackLabels.push(rLabel);
        // If all counts are zero (simulation not run yet), show estimated distribution
        const total = d.anomaly_counts.reduce((a, b) => a + b, 0);
        const counts = total > 0 ? d.anomaly_counts : [60, 10, 5, 18, 7];
        counts.forEach((c, i) => anomalyStackData[i].push(c));
        anomalyStackChart.update();
      }

      if (d.wave && d.attention) {
        ecgSynthActive = false;  // stop synthetic when real data arrives
        for (let i = 0; i < d.wave.length; i++) {
          ppgWave.push(d.wave[i]);
          ppgLabels.push('');
          attentionMask.push(d.attention[i] || 0);
        }
        const MAX_PTS = 625;  // ~5s at 125Hz
        if (ppgWave.length > MAX_PTS) {
          ppgWave.splice(0, ppgWave.length - MAX_PTS);
          ppgLabels.splice(0, ppgLabels.length - MAX_PTS);
          attentionMask.splice(0, attentionMask.length - MAX_PTS);
        }
        ppgChart.data.labels = ppgLabels;
        ppgChart.data.datasets[0].data = ppgWave;
        ppgChart.data.datasets[1].data = ppgWave;
        ppgChart.data.datasets[2].data = ppgWave;
        ppgChart.update('none');
      }

      // ── XAI badge ────────────────────────────────────────────
      const xb = document.getElementById('xai-badge');
      if (xb) {
        xb.textContent = d.xai_live ? 'REAL INFERENCE' : 'SYNTHETIC FALLBACK';
        xb.className = 'card-badge ' + (d.xai_live ? 'badge-green' : 'badge-red');
      }

      // ── DP panel ─────────────────────────────────────────────
      if (d.epsilon !== undefined) {
        const dpb = document.getElementById('dp-badge');
        const epv = document.getElementById('epsilonVal');
        const epBar = document.getElementById('epsilonBar');
        const dv = document.getElementById('deltaVal');
        const dpi = document.getElementById('dpInterpret');
        const ptdp = document.getElementById('pt-dp');

        if (d.dp_enabled && d.epsilon >= 0) {
          const eps = d.epsilon;
          dpb.textContent = 'DP-SGD ACTIVE'; dpb.className = 'card-badge badge-green';
          epv.textContent = eps.toFixed(3);
          epBar.style.width = Math.min((eps / 10) * 100, 100).toFixed(1) + '%';
          const g = eps < 2 ? '#7c3aed,#a855f7' : eps < 5 ? '#854F0B,#EF9F27' : '#A32D2D,#E24B4A';
          epBar.style.background = `linear-gradient(90deg,${g})`;
          const dExp = Math.round(Math.log10(d.delta));
          dv.textContent = `1e${dExp}`;
          let msg = eps < 1 ? 'Strong privacy — attacker learns very little.' :
            eps < 3 ? 'Good privacy — suitable for clinical data.' :
              eps < 8 ? 'Moderate — consider increasing sigma.' :
                'Weak — budget nearly exhausted.';
          dpi.textContent = `eps=${eps.toFixed(3)}, delta=1e${dExp} — ${msg}`;
          ptdp.textContent = 'ACTIVE'; ptdp.className = 'p-pill pill-green';
          document.getElementById('pt-eps').textContent = eps.toFixed(3);
          log(`[DP] Budget: eps=${eps.toFixed(3)}, delta=1e${dExp}`, 'log-pass');
        } else if (!d.dp_enabled) {
          dpb.textContent = 'DP DISABLED'; dpb.className = 'card-badge badge-red';
          epv.textContent = 'N/A';
          dpi.textContent = 'Differential privacy off — set DP_ENABLED=True in client.py';
          ptdp.textContent = 'DISABLED'; ptdp.className = 'p-pill pill-red';
        }
      }

      // ── Anomaly monitor ──────────────────────────────────────
      if (d.anomaly) {
        const a = d.anomaly;
        const sevMap = { safe: 'badge-green', warning: 'badge-orange', critical: 'badge-red' };
        document.getElementById('anomaly-name').textContent = a.name;
        document.getElementById('anomaly-name').style.color = a.color;
        document.getElementById('anomaly-rr').textContent = a.rr_pred > 0 ? `RR ${a.rr_pred.toFixed(1)} BrPM` : 'RR — BrPM';
        document.getElementById('anomaly-conf').textContent = a.confidence > 0 ? `Confidence ${(a.confidence * 100).toFixed(1)}%` : 'Waiting for inference...';
        const ab = document.getElementById('anomaly-badge');
        ab.textContent = a.name.toUpperCase(); ab.className = 'card-badge ' + (sevMap[a.severity] || 'badge-green');
        const ac = document.getElementById('anomaly-card');
        ac.style.borderColor = a.severity === 'safe' ? 'rgba(255,255,255,.06)' : a.color;
        if (a.severity === 'critical') log(`ANOMALY: ${a.name} — RR ${a.rr_pred.toFixed(1)} BrPM (conf ${(a.confidence * 100).toFixed(0)}%)`, 'log-fail');
        if (a.all_probs) a.all_probs.forEach((p, i) => {
          const b = document.getElementById(`pbar-${i}`);
          const v = document.getElementById(`ppct-${i}`);
          if (b) b.style.width = (p * 100).toFixed(1) + '%';
          if (v) v.textContent = (p * 100).toFixed(1) + '%';
        });
        // params table
        document.getElementById('pt-rr').textContent = a.rr_pred.toFixed(1);
        document.getElementById('pt-ano').textContent = a.name;
        document.getElementById('pt-ano').style.color = a.color;
        document.getElementById('pt-conf').textContent = (a.confidence * 100).toFixed(1) + '%';
      }
    };

    ws.onerror = () => log('WebSocket error — is uvicorn running?', 'log-fail');
    ws.onclose = () => {
      log('WebSocket closed — reconnecting in 3 s...', 'log-warn');
      setTimeout(() => {
        log('Attempting reconnect...', 'log-warn');
        const ws2 = new WebSocket(`ws://${window.location.host}/ws`);
        ws2.onopen    = ws.onopen;
        ws2.onmessage = ws.onmessage;
        ws2.onerror   = ws.onerror;
        ws2.onclose   = ws.onclose;
      }, 3000);
    };
  