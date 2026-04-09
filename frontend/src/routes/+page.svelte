<script>
  import { parseCV, recommendProjects, optimizeResume } from '$lib/api';
  import {
    step,
    parsedProjects, parseStatus, parseProgress, parseError, resetParse,
    jdText, roleType, charLimit, maxBullets, topK,
    recommendations, recommendStatus, recommendError, selectedIds, resetRecommend,
    genStatus, tokenBuffer, bullets, genError, elapsed, resetGeneration,
    isUploading, isRecommending, isGenerating,
  } from '$lib/stores';

  // ── Step 1: Upload & Parse ────────────────────────────────────────────────
  let dragOver  = false;
  let fileInput;

  function onDrop(e) {
    e.preventDefault(); dragOver = false;
    const f = e.dataTransfer?.files?.[0];
    if (f) uploadFile(f);
  }

  function onFileChange(e) {
    const f = e.target.files?.[0];
    if (f) uploadFile(f);
  }

  async function uploadFile(file) {
    resetParse();
    resetRecommend();
    parseStatus.set('uploading');
    parseProgress.set(`Reading ${file.name}…`);

    await parseCV(file, {
      onProgress: ({ message }) => { parseStatus.set('streaming'); parseProgress.set(message); },
      onProject:  ({ project }) => parsedProjects.update(ps => [...ps, project]),
      onDone:     ()            => parseStatus.set('done'),
      onError:    msg           => { parseStatus.set('error'); parseError.set(msg); },
    });
  }

  // ── Fact editing ──────────────────────────────────────────────────────────
  let expandedProject = null;

  function toggleExpand(id) {
    expandedProject = expandedProject === id ? null : id;
  }

  function updateFact(projectId, fi, field, value) {
    parsedProjects.update(ps => ps.map(p => {
      if (p.project_id !== projectId) return p;
      const facts = [...p.core_facts];
      facts[fi] = { ...facts[fi], [field]: value };
      return { ...p, core_facts: facts };
    }));
  }

  function updateFactArray(projectId, fi, field, raw) {
    updateFact(projectId, fi, field, raw.split(',').map(s => s.trim()).filter(Boolean));
  }

  function goToJD() {
    if ($parsedProjects.length > 0) step.set(2);
  }

  // ── Step 2: JD → Recommend → Generate ────────────────────────────────────
  async function analyseJD() {
    if ($jdText.length < 50) return;
    resetRecommend();
    recommendStatus.set('loading');

    try {
      const resp = await recommendProjects({
        job_description: $jdText,
        projects:        $parsedProjects,
        top_k:           $topK,
      });

      recommendations.set(resp.recommendations ?? []);
      selectedIds.set(new Set(
        (resp.recommendations ?? [])
          .filter(r => r.recommended)
          .map(r => r.project_id)
      ));
      recommendStatus.set('done');
    } catch (err) {
      recommendError.set(err.message);
      recommendStatus.set('error');
    }
  }

  function toggleProject(id) {
    selectedIds.update(s => {
      if (s.has(id)) s.delete(id); else s.add(id);
      return new Set(s);
    });
  }

  async function generate() {
    resetGeneration();
    genStatus.set('streaming');

    const selectedProjects = $parsedProjects.filter(p => $selectedIds.has(p.project_id));
    const payload = {
      job_description:  $jdText,
      target_role_type: $roleType,
      constraints: {
        target_char_limit:       Number($charLimit),
        tolerance:               2,
        bullet_prefix:           '•',
        max_bullets_per_project: Number($maxBullets),
      },
      projects: selectedProjects,
    };

    step.set(3);
    await optimizeResume(payload, {
      onToken:  t => tokenBuffer.update(b => b + t),
      onBullet: b => { bullets.update(arr => [...arr, b]); tokenBuffer.set(''); },
      onDone:   d => { elapsed.set(d?.elapsed_seconds ?? 0); genStatus.set('done'); },
      onError:  e => { genError.set(e); genStatus.set('error'); },
    });
  }

  // ── Copy helpers ──────────────────────────────────────────────────────────
  let copiedIdx = null;

  async function copyBullet(text, idx) {
    await navigator.clipboard.writeText(text);
    copiedIdx = idx;
    setTimeout(() => copiedIdx = null, 2000);
  }

  async function copyAll() {
    const grouped = {};
    for (const b of $bullets) {
      const pid = b.metadata?.project_id ?? 'other';
      if (!grouped[pid]) grouped[pid] = [];
      grouped[pid].push(b.text);
    }
    const text = Object.values(grouped).map(bs => bs.join('\n')).join('\n\n');
    await navigator.clipboard.writeText(text);
    copiedIdx = -1;
    setTimeout(() => copiedIdx = null, 2000);
  }

  // ── Score helpers ─────────────────────────────────────────────────────────
  function scoreBar(score) { return `${Math.round(score * 100)}%`; }

  function scoreColour(score) {
    if (score >= 0.7) return 'var(--green)';
    if (score >= 0.4) return 'var(--amber)';
    return 'var(--muted)';
  }

  const ROLES = [
    { value: 'ml_engineering',       label: 'ML Engineering' },
    { value: 'data_science',         label: 'Data Science' },
    { value: 'software_engineering', label: 'Software Engineering' },
    { value: 'quant_finance',        label: 'Quant / Finance' },
    { value: 'product_management',   label: 'Product Management' },
    { value: 'general',              label: 'General' },
  ];

  $: groupedBullets = (() => {
    const map = new Map();
    for (const b of $bullets) {
      const pid   = b.metadata?.project_id ?? 'other';
      const title = $parsedProjects.find(p => p.project_id === pid)?.title ?? pid;
      if (!map.has(pid)) map.set(pid, { title, bullets: [] });
      map.get(pid).bullets.push(b);
    }
    return [...map.values()];
  })();

  $: selectedCount = [...$selectedIds].filter(id => $parsedProjects.some(p => p.project_id === id)).length;
</script>

<!-- ═══════════════════════════════════════════════════════════════════════════
     STEP 1 — Upload CV
     ═══════════════════════════════════════════════════════════════════════ -->
{#if $step === 1}
<div class="fade-in step-container">

  <!-- Hero -->
  <div class="hero-section">
    <div class="hero-badge">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
        <polyline points="14 2 14 8 20 8"/>
      </svg>
      <span>Step 1 of 3</span>
    </div>
    <h1 class="hero-title">Upload your <span class="gradient-text">CV</span></h1>
    <p class="hero-subtitle">
      Drop a <span class="mono">.docx</span> or <span class="mono">.pdf</span> — our AI extracts every project, metric, and tool automatically.
    </p>
  </div>

  <!-- Drop zone -->
  {#if $parseStatus === 'idle' || $parseStatus === 'error'}
  <div
    class="dropzone-wrapper" class:active={dragOver}
    on:dragover|preventDefault={() => dragOver = true}
    on:dragleave={() => dragOver = false}
    on:drop={onDrop}
  >
    <div
      class="dropzone-inner"
      on:click={() => fileInput.click()}
      on:keydown={e => e.key === 'Enter' && fileInput.click()}
      role="button" tabindex="0"
    >
      <div class="drop-icon float-icon">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="url(#dropGrad)" stroke-width="1.5">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="17 8 12 3 7 8"/>
          <line x1="12" y1="3" x2="12" y2="15"/>
          <defs>
            <linearGradient id="dropGrad" x1="3" y1="3" x2="21" y2="21">
              <stop offset="0%" stop-color="#7c3aed"/>
              <stop offset="100%" stop-color="#06b6d4"/>
            </linearGradient>
          </defs>
        </svg>
      </div>
      <p class="drop-text">Drop your resume here or <span class="gradient-text">browse files</span></p>
      <p class="drop-hint mono">.docx (recommended) or .pdf · max 10 MB</p>
      <input bind:this={fileInput} type="file" accept=".docx,.pdf" class="hidden" on:change={onFileChange} id="cv-upload" />
    </div>
  </div>

  {#if $parseStatus === 'error'}
    <div class="error-banner">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>
      </svg>
      <span>{$parseError}</span>
    </div>
  {/if}
  {/if}

  <!-- Parsing progress -->
  {#if $parseStatus === 'uploading' || $parseStatus === 'streaming'}
  <div class="glass-lg pulse-glow parse-progress">
    <div class="progress-header">
      <div class="spinner"></div>
      <span class="mono" style="color:var(--accent-light);font-size:0.8125rem">{$parseProgress}</span>
    </div>
    {#if $parsedProjects.length > 0}
      <div class="stagger parsed-list">
        {#each $parsedProjects as p}
          <div class="parsed-item slide-in">
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
              <circle cx="8" cy="8" r="7" stroke="var(--green)" stroke-width="1.5" fill="var(--green-dim)"/>
              <path d="M5 8.5l2 2 4-4.5" stroke="var(--green)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <span>{p.title}</span>
            <span class="chip mono">{p.core_facts.length} facts</span>
          </div>
        {/each}
      </div>
    {/if}
  </div>
  {/if}

  <!-- Parsed project list -->
  {#if $parseStatus === 'done' && $parsedProjects.length > 0}
  <div class="stagger projects-section">
    <div class="section-header">
      <div>
        <h2 class="section-title">
          {$parsedProjects.length} project{$parsedProjects.length !== 1 ? 's' : ''} extracted
        </h2>
        <p class="section-subtitle">Review and optionally edit extracted facts before proceeding.</p>
      </div>
      <button class="btn-ghost" on:click={resetParse}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/>
        </svg>
        Re-upload
      </button>
    </div>

    {#each $parsedProjects as project (project.project_id)}
    {@const expanded = expandedProject === project.project_id}
    <div class="project-card surface-card" class:project-card--expanded={expanded}>
      <div class="project-card-main">
        <div class="project-info">
          <p class="project-title">{project.title}</p>
          <div class="project-chips">
            <span class="chip">{project.core_facts.length} facts</span>
            {#each [...new Set(project.core_facts.flatMap(f => f.tools))].slice(0,4) as tool}
              <span class="chip chip-accent">{tool}</span>
            {/each}
            {#each project.core_facts.flatMap(f => f.metrics).slice(0,2) as m}
              <span class="chip chip-amber">{m}</span>
            {/each}
          </div>
        </div>
        <button class="expand-btn mono" on:click={() => toggleExpand(project.project_id)}>
          {expanded ? '▲ collapse' : '▼ edit facts'}
        </button>
      </div>

      {#if expanded}
      <div class="project-facts">
        <p class="facts-hint">Optional — edit if the parser missed something.</p>
        {#each project.core_facts as fact, fi}
        <div class="fact-card">
          <span class="chip mono" style="font-size:0.6rem">{fact.fact_id}</span>
          <textarea rows="2" value={fact.text}
            on:change={e => updateFact(project.project_id, fi, 'text', e.target.value)}
            class="field mono" style="font-size:0.75rem;resize:none;min-height:3rem"></textarea>
          <div class="fact-fields">
            <div>
              <label for="tools-{project.project_id}-{fi}" class="field-label">Tools</label>
              <input id="tools-{project.project_id}-{fi}" class="field" style="font-size:0.75rem"
                value={fact.tools.join(', ')}
                on:change={e => updateFactArray(project.project_id, fi, 'tools', e.target.value)} />
            </div>
            <div>
              <label for="metrics-{project.project_id}-{fi}" class="field-label">Metrics</label>
              <input id="metrics-{project.project_id}-{fi}" class="field" style="font-size:0.75rem;background:var(--amber-dim);border-color:var(--amber-border);color:var(--amber)"
                value={fact.metrics.join(', ')}
                on:change={e => updateFactArray(project.project_id, fi, 'metrics', e.target.value)} />
            </div>
          </div>
        </div>
        {/each}
      </div>
      {/if}
    </div>
    {/each}

    <div style="display:flex;justify-content:flex-end;padding-top:0.5rem">
      <button class="btn-primary" on:click={goToJD}>
        Continue to Job Description
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-left:6px">
          <line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>
        </svg>
      </button>
    </div>
  </div>
  {/if}

</div>

<!-- ═══════════════════════════════════════════════════════════════════════════
     STEP 2 — JD + AI Recommendation + Generate
     ═══════════════════════════════════════════════════════════════════════ -->
{:else if $step === 2}
<div class="fade-in step-container">

  <div class="hero-section">
    <div class="hero-badge">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="21" x2="9" y2="9"/>
      </svg>
      <span>Step 2 of 3</span>
    </div>
    <h1 class="hero-title">Paste the <span class="gradient-text">Job Description</span></h1>
    <p class="hero-subtitle">Our AI scores your projects against the JD and recommends the best matches.</p>
  </div>

  <!-- JD textarea -->
  <div class="jd-section">
    <label for="jd-input" class="field-label" style="font-size:0.8125rem;font-weight:600">Job Description</label>
    <textarea
      id="jd-input"
      bind:value={$jdText}
      rows="8"
      placeholder={"Paste the full job description here…\n\nWe are looking for a Senior ML Engineer with expertise in Python, SARIMA forecasting, and production MLOps pipelines…"}
      class="field jd-textarea"
    ></textarea>
    <div style="display:flex;justify-content:flex-end;margin-top:0.375rem">
      <span class="mono" style="font-size:0.7rem;color:{$jdText.length < 50 ? 'var(--red)' : 'var(--muted)'}">
        {$jdText.length} chars {$jdText.length < 50 ? '· min 50' : ''}
      </span>
    </div>
  </div>

  <!-- Settings -->
  <div class="surface-card settings-panel">
    <p class="settings-title mono">SETTINGS</p>
    <div class="settings-grid">
      <div>
        <label for="role-select" class="field-label">Role type</label>
        <select id="role-select" bind:value={$roleType} class="field" style="font-size:0.75rem">
          {#each ROLES as r}<option value={r.value}>{r.label}</option>{/each}
        </select>
      </div>
      <div>
        <label for="char-limit" class="field-label">Target chars <span class="chip-amber" style="font-size:0.55rem">±2</span></label>
        <input id="char-limit" type="number" bind:value={$charLimit} min="60" max="300" class="field mono" style="font-size:0.75rem" />
      </div>
      <div>
        <label for="max-bullets" class="field-label">Bullets / project</label>
        <input id="max-bullets" type="number" bind:value={$maxBullets} min="1" max="8" class="field mono" style="font-size:0.75rem" />
      </div>
      <div>
        <label for="top-k" class="field-label">Projects to recommend</label>
        <input id="top-k" type="number" bind:value={$topK} min="1" max="6" class="field mono" style="font-size:0.75rem" />
      </div>
    </div>
  </div>

  <!-- Analyse button -->
  {#if $recommendStatus === 'idle' || $recommendStatus === 'error'}
  <button class="btn-primary" style="width:100%;padding:0.875rem" disabled={$jdText.length < 50} on:click={analyseJD}>
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right:8px">
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
    </svg>
    Analyse JD — Find Best Projects
  </button>
  {#if $recommendStatus === 'error'}
    <div class="error-banner">
      <span>⚠ {$recommendError}</span>
      <button class="btn-ghost" style="padding:0.25rem 0.75rem;font-size:0.7rem" on:click={analyseJD}>Retry</button>
    </div>
  {/if}
  {/if}

  <!-- Loading -->
  {#if $recommendStatus === 'loading'}
  <div class="glass-lg pulse-glow" style="padding:1.5rem;display:flex;align-items:center;gap:1rem">
    <div class="spinner"></div>
    <div>
      <p style="font-size:0.875rem;font-weight:500;color:var(--text)">Analysing job description…</p>
      <p style="font-size:0.75rem;color:var(--muted);margin-top:0.25rem">Scoring {$parsedProjects.length} projects against the JD</p>
    </div>
  </div>
  {/if}

  <!-- Recommendations -->
  {#if $recommendStatus === 'done' && $recommendations.length > 0}
  <div class="stagger reco-section">
    <div class="section-header">
      <div>
        <h2 class="section-title">AI Recommendation</h2>
        <p class="section-subtitle">Toggle projects to include in your optimized resume.</p>
      </div>
      <button class="btn-ghost" on:click={() => recommendStatus.set('idle')}>Re-analyse</button>
    </div>

    {#each $recommendations as rec}
    {@const selected = $selectedIds.has(rec.project_id)}
    {@const isRecommended = rec.recommended}
    <div class="reco-card surface-card"
         class:reco-card--selected={selected}
         class:reco-card--recommended={isRecommended && selected}
         style="opacity:{!isRecommended && !selected ? '0.5' : '1'}">

      <div class="reco-card-inner">
        <!-- Toggle -->
        <button
          class="toggle-btn"
          class:toggle-btn--on={selected}
          class:toggle-btn--green={selected && isRecommended}
          on:click={() => toggleProject(rec.project_id)}
          aria-label="Toggle {rec.title}"
        >
          {#if selected}
            <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
              <path d="M3 8.5l3.5 3.5 6.5-8" stroke="#fff" stroke-width="2" stroke-linecap="round"/>
            </svg>
          {/if}
        </button>

        <div class="reco-info">
          <div style="display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap">
            <p class="project-title" style="font-size:0.8125rem">{rec.title}</p>
            {#if isRecommended}
              <span class="chip chip-green">#{rec.rank} recommended</span>
            {:else}
              <span class="chip">#{rec.rank}</span>
            {/if}
          </div>

          {#if isRecommended && rec.reason}
            <p style="font-size:0.75rem;color:var(--text-secondary);margin-top:0.5rem;line-height:1.5">
              💡 {rec.reason}
            </p>
          {/if}

          <!-- Score bar -->
          <div style="display:flex;align-items:center;gap:0.75rem;margin-top:0.625rem">
            <div class="score-track">
              <div class="score-fill" style="width:{scoreBar(rec.score)};background:{scoreColour(rec.score)}"></div>
            </div>
            <span class="mono" style="font-size:0.7rem;color:{scoreColour(rec.score)};flex-shrink:0">
              {Math.round(rec.score * 100)}%
            </span>
          </div>

          <div class="project-chips" style="margin-top:0.5rem">
            {#each rec.matched_skills.slice(0,4) as skill}
              <span class="chip chip-accent">{skill}</span>
            {/each}
            {#each rec.top_metrics.slice(0,2) as m}
              <span class="chip chip-amber">{m}</span>
            {/each}
          </div>
        </div>
      </div>
    </div>
    {/each}

    <div style="padding-top:0.75rem;display:flex;flex-direction:column;gap:0.5rem">
      {#if selectedCount === 0}
        <p style="font-size:0.75rem;text-align:center;color:var(--red)">Select at least one project to continue.</p>
      {/if}
      <button class="btn-primary" style="width:100%;padding:0.875rem" disabled={selectedCount === 0} on:click={generate}>
        Generate bullets for {selectedCount} project{selectedCount !== 1 ? 's' : ''}
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-left:8px">
          <line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>
        </svg>
      </button>
    </div>
  </div>
  {/if}

  <div style="padding-top:0.5rem">
    <button class="btn-ghost" on:click={() => step.set(1)}>
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right:4px">
        <line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/>
      </svg>
      Back
    </button>
  </div>

</div>

<!-- ═══════════════════════════════════════════════════════════════════════════
     STEP 3 — Results
     ═══════════════════════════════════════════════════════════════════════ -->
{:else if $step === 3}
<div class="fade-in step-container">

  <div class="results-header">
    <div>
      <h1 class="hero-title" style="font-size:1.5rem">Your <span class="gradient-text">Bullets</span></h1>
      {#if $genStatus === 'streaming'}
        <p style="font-size:0.8125rem;color:var(--muted);margin-top:0.25rem">Generating<span class="cursor-blink"></span></p>
      {:else if $genStatus === 'done'}
        <p style="font-size:0.8125rem;color:var(--muted);margin-top:0.25rem">
          {$bullets.length} bullet{$bullets.length !== 1 ? 's' : ''} · {groupedBullets.length} project{groupedBullets.length !== 1 ? 's' : ''} · <span class="mono">{$elapsed}s</span>
        </p>
      {/if}
    </div>
    <div style="display:flex;gap:0.5rem;flex-wrap:wrap;flex-shrink:0">
      {#if $bullets.length > 1}
        <button class="btn-ghost" on:click={copyAll}>
          {copiedIdx === -1 ? '✓ Copied all' : 'Copy all'}
        </button>
      {/if}
      <button class="btn-ghost" on:click={() => { step.set(2); resetGeneration(); }}>← Projects</button>
      <button class="btn-ghost" on:click={() => { step.set(1); resetGeneration(); resetParse(); }}>Start over</button>
    </div>
  </div>

  <!-- Live typewriter -->
  {#if $tokenBuffer}
    <div class="glass-lg pulse-glow" style="padding:1.25rem;border-color:rgba(124,58,237,0.2)">
      <p class="mono" style="font-size:0.8125rem;color:var(--accent-light);line-height:1.6">{$tokenBuffer}<span class="cursor-blink"></span></p>
    </div>
  {/if}

  <!-- Bullets grouped by project -->
  {#each groupedBullets as group}
  <div class="bullet-group stagger">
    <div class="bullet-group-header">
      <h2 style="font-size:0.8125rem;font-weight:600;color:var(--text)">{group.title}</h2>
      <div class="bullet-group-line"></div>
      <span class="chip mono">{group.bullets.length}</span>
    </div>

    {#each group.bullets as bullet, i}
    {@const globalIdx = $bullets.indexOf(bullet)}
    {@const withinTol = bullet.metadata?.within_tolerance}
    <div class="bullet-card surface-card" class:bullet-card--warn={!withinTol}>
      <div class="bullet-text">
        <p class="mono" style="font-size:0.8125rem;line-height:1.65;color:var(--text)">{bullet.text}</p>
      </div>
      <div class="bullet-meta">
        <div class="project-chips">
          <span class="chip mono">{bullet.metadata?.char_count}/{bullet.metadata?.char_target} chars</span>
          <span class="chip {withinTol ? 'chip-green' : 'chip-amber'}">
            {withinTol ? '✓ within ±2' : '⚠ outside ±2'}
          </span>
          <span class="chip mono">{bullet.metadata?.iterations_taken} iter</span>
          {#if bullet.metadata?.jd_tone}
            <span class="chip">{bullet.metadata.jd_tone.replace(/_/g,' ')}</span>
          {/if}
        </div>
        <button
          class="copy-btn mono"
          class:copy-btn--done={copiedIdx === globalIdx}
          on:click={() => copyBullet(bullet.text, globalIdx)}
        >
          {copiedIdx === globalIdx ? '✓ Copied' : 'Copy'}
        </button>
      </div>
    </div>
    {/each}
  </div>
  {/each}

  <!-- Empty streaming state -->
  {#if $genStatus === 'streaming' && $bullets.length === 0 && !$tokenBuffer}
    <div class="glass-lg pulse-glow" style="padding:3rem;text-align:center">
      <div class="spinner" style="margin:0 auto 1rem"></div>
      <p style="font-size:0.8125rem;color:var(--muted)">Scoring facts, retrieving style exemplars…</p>
    </div>
  {/if}

  {#if $genStatus === 'error'}
    <div class="error-banner">⚠ {$genError}</div>
  {/if}

</div>
{/if}

<style>
  /* ── Layout ─────────────────────────────────────────────────────────── */
  .step-container {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }

  /* ── Hero ────────────────────────────────────────────────────────────── */
  .hero-section {
    display: flex;
    flex-direction: column;
    gap: 0.625rem;
    padding-bottom: 0.25rem;
  }

  .hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.375rem;
    font-size: 0.7rem;
    font-weight: 500;
    color: var(--accent-light);
    background: var(--accent-dim);
    border: 1px solid rgba(124,58,237,0.15);
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
    width: fit-content;
    font-family: 'JetBrains Mono', monospace;
  }

  .hero-title {
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1.2;
    color: var(--text);
  }

  .hero-subtitle {
    font-size: 0.875rem;
    color: var(--text-secondary);
    line-height: 1.6;
    max-width: 36rem;
  }

  /* ── Drop zone ──────────────────────────────────────────────────────── */
  .drop-icon {
    margin-bottom: 1.25rem;
    display: flex;
    justify-content: center;
  }
  .drop-text {
    font-size: 0.9375rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 0.5rem;
    text-align: center;
  }
  .drop-hint {
    font-size: 0.7rem;
    color: var(--muted);
    text-align: center;
  }

  /* ── Parse progress ─────────────────────────────────────────────────── */
  .parse-progress { padding: 1.25rem; }
  .progress-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }
  .parsed-list {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    padding-top: 0.75rem;
    margin-top: 0.75rem;
    border-top: 1px solid var(--border);
  }
  .parsed-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.8125rem;
    color: var(--text-secondary);
  }

  /* ── Section headers ────────────────────────────────────────────────── */
  .section-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.25rem;
  }
  .section-title {
    font-size: 0.9375rem;
    font-weight: 600;
    color: var(--text);
  }
  .section-subtitle {
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 0.25rem;
  }

  .projects-section {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  /* ── Project cards ──────────────────────────────────────────────────── */
  .project-card { transition: border-color 0.2s; }
  .project-card:hover { border-color: var(--border-hi); }
  .project-card-main {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    padding: 1rem 1.25rem;
  }
  .project-info { flex: 1; min-width: 0; }
  .project-title {
    font-size: 0.875rem;
    font-weight: 600;
    color: var(--text);
  }
  .project-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.375rem;
    margin-top: 0.5rem;
  }
  .expand-btn {
    font-size: 0.65rem;
    color: var(--muted);
    background: none;
    border: none;
    cursor: pointer;
    flex-shrink: 0;
    padding: 0.25rem;
    transition: color 0.2s;
  }
  .expand-btn:hover { color: var(--text); }

  .project-facts {
    border-top: 1px solid var(--border);
    padding: 1rem 1.25rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    background: rgba(0,0,0,0.15);
  }
  .facts-hint {
    font-size: 0.7rem;
    color: var(--muted);
  }
  .fact-card {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    padding: 0.75rem;
    background: rgba(255,255,255,0.02);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }
  .fact-fields {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
  }
  .field-label {
    font-size: 0.7rem;
    color: var(--muted);
    margin-bottom: 0.25rem;
    display: block;
  }

  /* ── JD section ─────────────────────────────────────────────────────── */
  .jd-textarea {
    font-size: 0.8125rem;
    line-height: 1.6;
    margin-top: 0.5rem;
    resize: none;
  }

  /* ── Settings panel ─────────────────────────────────────────────────── */
  .settings-panel { padding: 1rem 1.25rem; }
  .settings-title {
    font-size: 0.625rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin-bottom: 0.75rem;
  }
  .settings-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.75rem;
  }
  @media (min-width: 640px) {
    .settings-grid { grid-template-columns: repeat(4, 1fr); }
  }

  /* ── Recommendation cards ───────────────────────────────────────────── */
  .reco-section {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }
  .reco-card { transition: all 0.2s; }
  .reco-card--selected { border-color: rgba(124,58,237,0.25); }
  .reco-card--recommended { border-color: var(--green-border); }
  .reco-card-inner {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    padding: 1rem 1.25rem;
  }

  .toggle-btn {
    width: 22px; height: 22px;
    border-radius: 6px;
    border: 1.5px solid var(--border-hi);
    background: rgba(255,255,255,0.03);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 2px;
    transition: all 0.2s;
  }
  .toggle-btn--on {
    background: var(--accent);
    border-color: var(--accent);
  }
  .toggle-btn--green {
    background: var(--green);
    border-color: var(--green);
  }

  .reco-info { flex: 1; min-width: 0; }

  .score-track {
    flex: 1;
    height: 4px;
    border-radius: 2px;
    background: rgba(255,255,255,0.05);
    overflow: hidden;
  }
  .score-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.6s cubic-bezier(.16,1,.3,1);
  }

  /* ── Results ────────────────────────────────────────────────────────── */
  .results-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
  }

  .bullet-group {
    display: flex;
    flex-direction: column;
    gap: 0.625rem;
  }
  .bullet-group-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }
  .bullet-group-line {
    flex: 1;
    height: 1px;
    background: var(--border);
  }

  .bullet-card { transition: border-color 0.2s; }
  .bullet-card--warn::before {
    background: linear-gradient(135deg, var(--amber) 0%, rgba(245,158,11,0.3) 100%);
  }
  .bullet-text { padding: 1.25rem 1.25rem 0.75rem; }
  .bullet-meta {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    flex-wrap: wrap;
    padding: 0.75rem 1.25rem;
    border-top: 1px solid var(--border);
  }

  .copy-btn {
    font-size: 0.7rem;
    padding: 0.35rem 0.75rem;
    border-radius: 6px;
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    color: var(--muted);
    cursor: pointer;
    transition: all 0.2s;
    flex-shrink: 0;
  }
  .copy-btn:hover {
    color: var(--text);
    border-color: var(--border-hi);
  }
  .copy-btn--done {
    background: var(--green-dim);
    color: var(--green);
    border-color: var(--green-border);
  }

  /* ── Error banner ───────────────────────────────────────────────────── */
  .error-banner {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem 1rem;
    border-radius: var(--radius-sm);
    background: var(--red-dim);
    border: 1px solid var(--red-border);
    color: #fca5a5;
    font-size: 0.8125rem;
  }
</style>
