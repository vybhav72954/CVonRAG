<script>
  import { onMount } from 'svelte';
  import { get } from 'svelte/store';
  import { base } from '$app/paths';
  import { parseCV, recommendProjects, optimizeResume, checkHealth, abortInFlight, clearAuth } from '$lib/api';
  import {
    step,
    idToken, userEmail,
    parsedProjects, parseStatus, parseProgress, parseError, parseWarnings, resetParse,
    jdText, roleType, charLimit, maxBullets, topK,
    recommendations, recommendStatus, recommendError, selectedIds, resetRecommend,
    genStatus, tokenBuffer, bullets, genError, elapsed, resetGeneration,
    isUploading, isRecommending, isGenerating,
  } from '$lib/stores';

  // ── Google Sign-In ───────────────────────────────────────────────────────
  // Until the user signs in, every gated backend call would 401 — so step 1
  // hides its upload UI behind the sign-in card below. The Google ID token
  // (1-hour expiry) is sessionStorage-persisted via the idToken store; on
  // expiry the backend returns 401 and api.js's clearAuth() flips us back
  // to this card.

  const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_OAUTH_CLIENT_ID ?? '';
  const GOOGLE_HD        = import.meta.env.VITE_GOOGLE_OAUTH_HD ?? '';
  let gsiButtonRef;       // DOM target for renderButton
  let gsiError = '';      // surfaced when the GIS library can't load / init

  /** Decode the JWT payload client-side. Used ONLY to extract `email` for
   *  display in the header chip — the backend re-verifies the signature on
   *  every request, so this is not a trust boundary. */
  function _decodeEmail(jwt) {
    try {
      const payload = jwt.split('.')[1];
      // base64url → base64
      const b64 = payload.replace(/-/g, '+').replace(/_/g, '/');
      const json = atob(b64.padEnd(b64.length + ((4 - b64.length % 4) % 4), '='));
      return JSON.parse(json).email ?? '';
    } catch {
      return '';
    }
  }

  function handleCredentialResponse(response) {
    const token = response?.credential ?? '';
    if (!token) {
      gsiError = 'Sign-in failed: no credential returned.';
      return;
    }
    idToken.set(token);
    userEmail.set(_decodeEmail(token));
    gsiError = '';
  }

  /** Wait for the GIS script (loaded async in app.html) to be ready, then
   *  initialize and render the sign-in button. Two-tier wait:
   *    1. Fast poll (30 × 100ms = 3s) — covers the common case where the
   *       script is already loaded by the time the gate card mounts.
   *    2. Slower poll (30 × 1s = 30s) — flaky-network fallback. The async
   *       defer script tag can take noticeably longer to load on slow home
   *       connections, and giving up after 3s strands the user with an
   *       error they can't recover from without a page refresh.
   *  gsiError is only set after the full 33s budget is exhausted, and is
   *  cleared on a successful retry so a transient slow load doesn't leave
   *  a stale message visible after the button renders. */
  async function initGoogleSignIn() {
    if (!GOOGLE_CLIENT_ID) {
      gsiError = 'VITE_GOOGLE_OAUTH_CLIENT_ID is not set — sign-in disabled.';
      return;
    }
    // Tier 1: fast poll.
    for (let i = 0; i < 30; i++) {  // up to ~3s
      if (window.google?.accounts?.id) break;
      await new Promise(r => setTimeout(r, 100));
    }
    // Tier 2: slower poll for flaky networks.
    for (let i = 0; i < 30 && !window.google?.accounts?.id; i++) {  // up to +30s
      await new Promise(r => setTimeout(r, 1000));
    }
    if (!window.google?.accounts?.id) {
      gsiError = 'Google sign-in library failed to load after 30s. Check your network connection or browser settings, then refresh.';
      return;
    }
    gsiError = '';  // clear any stale slow-load message before we render
    try {
      window.google.accounts.id.initialize({
        client_id: GOOGLE_CLIENT_ID,
        callback: handleCredentialResponse,
        // hd is a UX hint — restricts the account picker. Server-side hd
        // claim verification in app/auth.py is the enforcement.
        hd: GOOGLE_HD || undefined,
        auto_select: false,
        ux_mode: 'popup',
        context: 'signin',
      });
      if (gsiButtonRef) {
        window.google.accounts.id.renderButton(gsiButtonRef, {
          theme: 'filled_black',
          size: 'large',
          type: 'standard',
          text: 'signin_with',
          shape: 'pill',
          logo_alignment: 'left',
        });
      }
    } catch (err) {
      gsiError = `Sign-in init failed: ${err.message ?? err}`;
    }
  }

  function signOut() {
    // Abort any in-flight gated requests first — otherwise an /optimize
    // stream started under the old token keeps burning quota and mutates
    // stores after the user is effectively signed out.
    abortInFlight('parse');
    abortInFlight('recommend');
    abortInFlight('optimize');
    // Reset transient status flags so a stale "streaming…" banner doesn't
    // linger after sign-out. Persisted state (parsedProjects, bullets, …)
    // deliberately survives — if the user signs back in, they can keep working.
    parseStatus.set('idle');
    parseProgress.set('');
    parseError.set('');
    parseWarnings.set([]);
    recommendStatus.set('idle');
    recommendError.set('');
    genStatus.set('idle');
    tokenBuffer.set('');
    genError.set('');
    clearAuth();
    step.set(1);
  }

  // ── Backend health check (F6) — also re-runs on focus/visibility ────────
  let backendDown = false;
  let backendMsg  = '';

  async function refreshHealth() {
    const { ok, reason } = await checkHealth();
    backendDown = !ok;
    backendMsg  = ok ? '' : reason;
  }

  // Refresh-during-streaming guard: persistent stores already preserve parsed
  // projects, JD, settings, and completed bullets across reloads, but the
  // in-flight LLM stream cannot be resumed — those tokens are gone the moment
  // the page unloads. Show the browser's native confirmation prompt so an
  // accidental Cmd-R/F5 mid-generation doesn't burn a Groq quota silently.
  function onBeforeUnload(e) {
    const streaming =
      get(genStatus)    === 'streaming' ||
      get(parseStatus)  === 'uploading' ||
      get(parseStatus)  === 'streaming' ||
      get(recommendStatus) === 'loading';
    if (streaming) {
      e.preventDefault();
      e.returnValue = '';  // Chrome requires returnValue to be set
      return '';
    }
  }

  onMount(() => {
    refreshHealth();
    const onVisible = () => { if (document.visibilityState === 'visible') refreshHealth(); };
    document.addEventListener('visibilitychange', onVisible);
    window.addEventListener('focus', refreshHealth);
    window.addEventListener('beforeunload', onBeforeUnload);
    return () => {
      document.removeEventListener('visibilitychange', onVisible);
      window.removeEventListener('focus', refreshHealth);
      window.removeEventListener('beforeunload', onBeforeUnload);
    };
  });

  // Re-init the Google sign-in button whenever the gate is open AND the
  // button slot mounts. The `gsiButtonRef` reactive trigger fires on first
  // mount and after sign-out (when the card re-renders).
  $: if (gsiButtonRef && !$idToken) initGoogleSignIn();

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
      // F2: distinguish a per-project error (stream continues, projects exist) from
      // a fatal error (no projects parsed, stream broken). The first is a warning
      // the user should see alongside their results; the second is fatal.
      onError:    msg           => {
        const haveAny = $parsedProjects.length > 0;
        parseWarnings.update(w => [...w, msg]);
        if (!haveAny) {
          parseStatus.set('error');
          parseError.set(msg);
        }
      },
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

    await recommendProjects(
      { job_description: $jdText, projects: $parsedProjects, top_k: $topK },
      {
        onDone(resp) {
          recommendations.set(resp.recommendations ?? []);
          selectedIds.set(new Set(
            (resp.recommendations ?? [])
              .filter(r => r.recommended)
              .map(r => r.project_id)
          ));
          recommendStatus.set('done');
        },
        onError(msg) {
          recommendError.set(msg);
          recommendStatus.set('error');
        },
      },
    );
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

<!-- ── Backend-down banner ──────────────────────────────────────────────── -->
{#if backendDown}
<div class="error-banner" style="margin-bottom:1.5rem">
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
    <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
  </svg>
  <span>Backend unreachable: {backendMsg}. Make sure the API server is running.</span>
</div>
{/if}

<!-- ── Signed-in chip (visible on every step once signed in) ────────────── -->
{#if $idToken}
<div class="invite-chip" aria-label="Signed-in account">
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
    <circle cx="12" cy="7" r="4"/>
  </svg>
  <span class="mono">Signed in: <strong>{$userEmail || '…'}</strong></span>
  <button class="invite-change" type="button" on:click={signOut}>sign out</button>
</div>
{/if}

<!-- ═══════════════════════════════════════════════════════════════════════════
     SIGN-IN GATE: shown when no Google ID token is held. Hides the rest of
     the wizard until the user signs in with their institutional account.
     ═══════════════════════════════════════════════════════════════════════ -->
{#if !$idToken}
<div class="fade-in step-container">
  <div class="hero-section" style="align-items:center;text-align:center">
    <div class="hero-badge">
      <div class="badge-dot"></div>
      <span>Sign in</span>
    </div>
    <h1 class="hero-title">Sign in with your <span class="gradient-text">institutional Google account</span></h1>
    <p class="hero-subtitle">
      CVonRAG is restricted to your batch's Google Workspace. Your sign-in
      tracks your daily quota so one user can't accidentally burn through the
      shared LLM budget. Tokens expire after an hour; just sign in again when
      that happens.
    </p>
  </div>

  <div class="invite-card" style="display:flex;flex-direction:column;align-items:center;gap:0.875rem">
    <div bind:this={gsiButtonRef} aria-label="Google sign-in button"></div>
    {#if gsiError}
      <p class="mono" style="font-size:0.7rem;color:var(--red);text-align:center">
        {gsiError}
      </p>
    {/if}
    <p class="mono" style="font-size:0.7rem;color:var(--muted);text-align:center">
      Trouble signing in? Make sure you're using your institutional Google
      account (not a personal Gmail).
    </p>
  </div>
</div>
{:else}

<!-- ═══════════════════════════════════════════════════════════════════════════
     STEP 1: Upload BioData
     ═══════════════════════════════════════════════════════════════════════ -->
{#if $step === 1}
<div class="fade-in step-container">

  <!-- Hero -->
  <div class="hero-section hero-upload">
    <div class="hero-badge">
      <div class="badge-dot"></div>
      <span>Step 1 of 3</span>
    </div>

    <h1 class="hero-title">
      Create your <span class="gradient-text">Resume</span><br/>
      leveraging our state-of-the-art Corpus
    </h1>

    <p class="hero-subtitle">
      Upload your <span class="biodata-term" tabindex="0">BioData.<span class="biodata-tooltip">A BioData is your complete project inventory: every project you've done, with all related facts, metrics, tools, and outcomes. As detailed as possible.</span></span>
      It should contain all our projects, along with all your bullets, metrics and impacts.
      Sit back and relax as  our 5-phase RAG pipeline
      extracts every project, metric, and tool, then crafts bullets that match any JD's tone.
    </p>

    <!-- Feature pills -->
    <div class="hero-features">
      <div class="feature-pill">
        <span class="pill-icon">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--accent-light)" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
        </span>
        <span class="pill-value">5-Phase RAG</span>
      </div>
      <div class="feature-pill">
        <span class="pill-icon">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--cyan)" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
        </span>
        <span class="pill-value">288 Exemplars</span>
      </div>
      <div class="feature-pill">
        <span class="pill-icon">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--green)" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
        </span>
        <span class="pill-value">Curated CVs</span>
      </div>
    </div>
  </div>

  <!-- Drop zone -->
  {#if $parseStatus === 'idle' || $parseStatus === 'error'}
  <div
    class="dropzone-wrapper" class:active={dragOver}
    role="region"
    aria-label="File upload dropzone"
    on:dragover|preventDefault={() => dragOver = true}
    on:dragleave={() => dragOver = false}
    on:drop={onDrop}
  >
    <div
      class="dropzone-inner"
      on:click={() => fileInput.click()}
      on:keydown={e => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();  // Space on a focused element scrolls otherwise
          fileInput.click();
        }
      }}
      role="button" tabindex="0"
    >
      <div class="drop-icon float-icon">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="url(#dropGrad)" stroke-width="1.5">
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
      <p class="drop-text">Drop your BioData here or <span class="gradient-text">browse files</span></p>
      <p class="drop-hint mono">.docx (recommended) or .pdf · max 10 MB</p>
      <input bind:this={fileInput} type="file" accept=".docx,.pdf" class="hidden" on:change={onFileChange} id="cv-upload" />
    </div>
  </div>

  <!-- Sample biodata download capsule (issue #29 / closes #7) -->
  <a
    class="sample-capsule"
    href="{base}/sample-biodata.docx"
    download="sample-biodata.docx"
    aria-label="Download a sample biodata to see the expected format"
  >
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="7 10 12 15 17 10"/>
      <line x1="12" y1="15" x2="12" y2="3"/>
    </svg>
    <span>Not sure what to upload? <strong>Download a sample biodata</strong></span>
  </a>

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
          <span class="gradient-text">{$parsedProjects.length}</span> project{$parsedProjects.length !== 1 ? 's' : ''} extracted
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

    <!-- Per-project parse warnings (F2): visible alongside successful projects -->
    {#if $parseWarnings.length > 0}
    <div class="warning-banner">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="flex-shrink:0;margin-top:2px">
        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
        <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
      </svg>
      <div style="flex:1">
        <p style="font-weight:600;font-size:0.8125rem;color:var(--amber);margin-bottom:0.25rem">
          {$parseWarnings.length} project{$parseWarnings.length !== 1 ? 's were' : ' was'} skipped during parsing
        </p>
        <ul style="margin:0;padding-left:1rem;font-size:0.75rem;color:var(--text-secondary);line-height:1.5">
          {#each $parseWarnings as w}
            <li>{w}</li>
          {/each}
        </ul>
      </div>
    </div>
    {/if}

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
        <p class="facts-hint">Optional: edit if the parser missed something.</p>
        {#each project.core_facts as fact, fi}
        <div class="fact-card">
          <span class="chip mono" style="font-size:0.6rem">{fact.fact_id}</span>
          <label for="fact-{project.project_id}-{fi}" class="sr-only">Fact text</label>
          <textarea id="fact-{project.project_id}-{fi}" rows="2" value={fact.text}
            on:change={e => updateFact(project.project_id, fi, 'text', e.currentTarget.value)}
            class="field mono" style="font-size:0.75rem;resize:none;min-height:3rem"></textarea>
          <div class="fact-fields">
            <div>
              <label for="tools-{project.project_id}-{fi}" class="field-label">Tools</label>
              <input id="tools-{project.project_id}-{fi}" class="field" style="font-size:0.75rem"
                value={fact.tools.join(', ')}
                on:change={e => updateFactArray(project.project_id, fi, 'tools', e.currentTarget.value)} />
            </div>
            <div>
              <label for="metrics-{project.project_id}-{fi}" class="field-label">Metrics</label>
              <input id="metrics-{project.project_id}-{fi}" class="field" style="font-size:0.75rem;background:var(--amber-dim);border-color:var(--amber-border);color:var(--amber)"
                value={fact.metrics.join(', ')}
                on:change={e => updateFactArray(project.project_id, fi, 'metrics', e.currentTarget.value)} />
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
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>
        </svg>
      </button>
    </div>
  </div>
  {/if}

</div>

<!-- ═══════════════════════════════════════════════════════════════════════════
     STEP 2: JD + AI Recommendation + Generate
     ═══════════════════════════════════════════════════════════════════════ -->
{:else if $step === 2}
<div class="fade-in step-container">

  <div class="hero-section">
    <div class="hero-badge">
      <div class="badge-dot badge-dot--cyan"></div>
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
      maxlength="10000"
      placeholder={"Paste the full job description here…\n\nWe are looking for a Senior ML Engineer with expertise in Python, SARIMA forecasting, and production MLOps pipelines…"}
      class="field jd-textarea"
    ></textarea>
    <div style="display:flex;justify-content:flex-end;margin-top:0.375rem">
      <span class="mono" style="font-size:0.7rem;color:{$jdText.length < 50 ? 'var(--red)' : ($jdText.length > 9500 ? 'var(--amber)' : 'var(--muted)')}">
        {$jdText.length} / 10000 chars {$jdText.length < 50 ? '· min 50' : ''}
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
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
    </svg>
    Analyse JD: Find Best Projects
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
        <p class="section-subtitle">Toggle projects to include in your optimised bullets.</p>
      </div>
      <button class="btn-ghost" on:click={() => { abortInFlight('recommend'); resetRecommend(); }}>Re-analyse</button>
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
              {rec.reason}
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
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>
        </svg>
      </button>
    </div>
  </div>
  {/if}

  <div style="padding-top:0.5rem">
    <button class="btn-ghost" on:click={() => step.set(1)}>
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/>
      </svg>
      Back
    </button>
  </div>

</div>

<!-- ═══════════════════════════════════════════════════════════════════════════
     STEP 3: Results
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
      <button class="btn-ghost" on:click={() => { abortInFlight('optimize'); step.set(2); resetGeneration(); }}>← Projects</button>
      <button class="btn-ghost" on:click={() => { abortInFlight('optimize'); step.set(1); resetGeneration(); resetParse(); }}>Start over</button>
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

{/if}  <!-- close sign-in gate {:else} from the top of the wizard -->

<style>
  /* ── Invite chip + card ─────────────────────────────────────────────── */
  .invite-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.7rem;
    color: var(--muted);
    background: var(--accent-dim);
    border: 1px solid rgba(124,58,237,0.15);
    padding: 0.3rem 0.7rem;
    border-radius: 999px;
    margin-bottom: 1.25rem;
    width: fit-content;
  }
  .invite-chip strong {
    color: var(--accent-light);
    font-family: 'JetBrains Mono', monospace;
  }
  .invite-change {
    background: transparent;
    border: none;
    color: var(--muted);
    cursor: pointer;
    font-size: 0.65rem;
    text-decoration: underline;
    padding: 0;
    margin-left: 0.25rem;
  }
  .invite-change:hover { color: var(--accent-light); }

  .invite-card {
    max-width: 420px;
    margin: 0 auto;
    padding: 1.75rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
  }

  /* ── Layout ─────────────────────────────────────────────────────────── */
  .step-container {
    display: flex;
    flex-direction: column;
    gap: 1.75rem;
  }

  /* ── Hero ────────────────────────────────────────────────────────────── */
  .hero-section {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    padding-bottom: 0.25rem;
  }

  .hero-upload {
    text-align: center;
    align-items: center;
    padding-top: 1.5rem;
    padding-bottom: 1rem;
  }

  .hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.7rem;
    font-weight: 500;
    color: var(--accent-light);
    background: var(--accent-dim);
    border: 1px solid rgba(124,58,237,0.15);
    padding: 0.35rem 0.85rem;
    border-radius: 999px;
    width: fit-content;
    font-family: 'JetBrains Mono', monospace;
  }

  .badge-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent-light);
    box-shadow: 0 0 8px var(--accent-glow);
    animation: badgePulse 2s ease-in-out infinite;
  }
  .badge-dot--cyan {
    background: var(--cyan);
    box-shadow: 0 0 8px rgba(6, 182, 212, 0.4);
  }
  @keyframes badgePulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.5; transform: scale(0.8); }
  }

  .hero-title {
    font-size: 2.25rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1.15;
    color: var(--text);
  }

  @media (min-width: 640px) {
    .hero-title { font-size: 2.75rem; }
  }

  .hero-subtitle {
    font-size: 0.9375rem;
    color: var(--text-secondary);
    line-height: 1.7;
    max-width: 38rem;
  }

  /* ── BioData tooltip ───────────────────────────────────────────────── */
  .biodata-term {
    position: relative;
    color: var(--accent-light);
    font-weight: 600;
    cursor: help;
    border-bottom: 1px dashed rgba(167, 139, 250, 0.4);
  }

  .biodata-tooltip {
    position: absolute;
    bottom: calc(100% + 10px);
    left: 50%;
    transform: translateX(-50%);
    width: 300px;
    padding: 0.75rem 1rem;
    background: var(--surface-solid);
    border: 1px solid var(--border-hi);
    border-radius: var(--radius-sm);
    font-size: 0.75rem;
    font-weight: 400;
    color: var(--text-secondary);
    line-height: 1.6;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.2s, transform 0.2s;
    transform: translateX(-50%) translateY(4px);
    z-index: 20;
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
  }

  .biodata-tooltip::after {
    content: '';
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    border: 6px solid transparent;
    border-top-color: var(--border-hi);
  }

  .biodata-term:hover .biodata-tooltip,
  .biodata-term:focus .biodata-tooltip {
    opacity: 1;
    transform: translateX(-50%) translateY(0);
    pointer-events: auto;
  }

  .hero-features {
    display: flex;
    flex-wrap: wrap;
    gap: 0.625rem;
    justify-content: center;
    margin-top: 0.75rem;
  }

  /* ── Drop zone ──────────────────────────────────────────────────────── */
  .drop-icon {
    margin-bottom: 1.5rem;
    display: flex;
    justify-content: center;
  }
  .drop-text {
    font-size: 1rem;
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

  /* ── Sample biodata capsule (tertiary CTA below dropzone) ──────────── */
  .sample-capsule {
    display: inline-flex;
    align-items: center;
    gap: 0.7rem;
    margin: 1.1rem auto 0;
    padding: 0.7rem 1.4rem;
    font-size: 0.9rem;
    color: var(--muted);
    background: var(--accent-dim);
    border: 1px solid rgba(124,58,237,0.22);
    border-radius: 999px;
    text-decoration: none;
    width: fit-content;
    transition: color 160ms ease, border-color 160ms ease, transform 160ms ease;
  }
  .sample-capsule:hover,
  .sample-capsule:focus-visible {
    color: var(--accent-light);
    border-color: rgba(124,58,237,0.45);
    transform: translateY(-1px);
  }
  .sample-capsule strong {
    color: var(--accent-light);
    font-weight: 600;
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
    font-size: 1rem;
    font-weight: 700;
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
  .project-card { transition: border-color 0.3s, box-shadow 0.3s; }
  .project-card:hover {
    border-color: var(--border-hi);
    box-shadow: 0 4px 30px rgba(0,0,0,0.2);
  }
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
  .reco-card { transition: all 0.3s; }
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

  .bullet-card { transition: border-color 0.3s; }
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

  /* ── Warning banner (F2 — partial parse failures) ──────────────────── */
  .warning-banner {
    display: flex;
    align-items: flex-start;
    gap: 0.625rem;
    padding: 0.75rem 1rem;
    border-radius: var(--radius-sm);
    background: var(--amber-dim);
    border: 1px solid var(--amber-border);
    color: var(--amber);
  }
</style>
