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

  // ── Fact editing (inline, optional) ──────────────────────────────────────
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
      // Pre-select the AI-recommended projects
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
    // Group bullets by project for clean copy
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

  // ── Score bar width ───────────────────────────────────────────────────────
  function scoreBar(score) {
    return `${Math.round(score * 100)}%`;
  }

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

  // bullets grouped by project title for Step 3
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
</script>

<!-- ═════════════════════════════════════════════════════════════════════════
     STEP 1 — Upload CV
     ═════════════════════════════════════════════════════════════════════════ -->
{#if $step === 1}
<div class="fade-in space-y-7">

  <div>
    <h1 class="text-2xl font-bold" style="color:var(--text)">Upload your CV</h1>
    <p class="text-sm mt-1" style="color:var(--muted)">
      .docx or .pdf — all projects are extracted automatically. Numbers preserved exactly.
    </p>
  </div>

  <!-- Drop zone -->
  {#if $parseStatus === 'idle' || $parseStatus === 'error'}
  <div
    class="border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all duration-200"
    style="border-color:{dragOver ? 'var(--indigo)' : 'var(--border-hi)'}; background:{dragOver ? 'rgba(99,102,241,0.06)' : 'var(--surface)'}"
    on:dragover|preventDefault={() => dragOver = true}
    on:dragleave={() => dragOver = false}
    on:drop={onDrop}
    on:click={() => fileInput.click()}
    role="button" tabindex="0"
    on:keydown={e => e.key === 'Enter' && fileInput.click()}
  >
    <div class="text-4xl mb-3">📄</div>
    <p class="font-semibold" style="color:var(--text)">Drop your biodata here or click to browse</p>
    <p class="text-xs mt-1.5" style="color:var(--muted)">.docx (best) or .pdf · max 10 MB</p>
    <input bind:this={fileInput} type="file" accept=".docx,.pdf" class="hidden" on:change={onFileChange} />
  </div>

  {#if $parseStatus === 'error'}
    <div class="rounded-lg px-4 py-3 text-sm" style="background:var(--red-dim);border:1px solid var(--red);color:#fca5a5">
      ⚠ {$parseError}
    </div>
  {/if}
  {/if}

  <!-- Parsing progress -->
  {#if $parseStatus === 'uploading' || $parseStatus === 'streaming'}
  <div class="surface px-5 py-4 space-y-3">
    <div class="flex items-center gap-3">
      <div class="w-4 h-4 rounded-full border-2 animate-spin" style="border-color:var(--indigo);border-top-color:transparent"></div>
      <span class="text-sm mono" style="color:var(--indigo)">{$parseProgress}</span>
    </div>
    {#if $parsedProjects.length > 0}
      <div class="space-y-1.5 pt-1">
        {#each $parsedProjects as p}
          <div class="slide-in flex items-center gap-2 text-sm" style="color:var(--muted)">
            <span style="color:var(--green)">✓</span>
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
  <div class="space-y-3">
    <div class="flex items-center justify-between">
      <p class="text-sm font-semibold" style="color:var(--text)">
        {$parsedProjects.length} project{$parsedProjects.length !== 1 ? 's' : ''} found
      </p>
      <button class="btn-ghost text-xs" on:click={resetParse}>Upload different file</button>
    </div>

    {#each $parsedProjects as project (project.project_id)}
    {@const expanded = expandedProject === project.project_id}
    <div class="rounded-xl overflow-hidden fade-in"
         style="border:1px solid var(--border);background:var(--surface)">

      <div class="flex items-start gap-3 px-4 py-3">
        <div class="flex-1 min-w-0">
          <p class="font-semibold text-sm" style="color:var(--text)">{project.title}</p>
          <div class="flex flex-wrap gap-1.5 mt-1.5">
            <span class="chip">{project.core_facts.length} facts</span>
            {#each [...new Set(project.core_facts.flatMap(f => f.tools))].slice(0,4) as tool}
              <span class="chip chip-indigo">{tool}</span>
            {/each}
            {#each project.core_facts.flatMap(f => f.metrics).slice(0,2) as m}
              <span class="chip chip-amber">{m}</span>
            {/each}
          </div>
        </div>
        <button class="text-xs mono flex-shrink-0 transition-colors"
                style="color:var(--muted)"
                on:click={() => toggleExpand(project.project_id)}>
          {expanded ? 'collapse ↑' : 'edit facts ↓'}
        </button>
      </div>

      {#if expanded}
      <div class="border-t px-4 py-3 space-y-3" style="border-color:var(--border);background:rgba(0,0,0,0.2)">
        <p class="text-xs" style="color:var(--muted)">
          Optional — edit if the parser missed something.
          <span class="chip chip-amber ml-1">amber = metrics</span>
        </p>
        {#each project.core_facts as fact, fi}
        <div class="rounded-lg p-3 space-y-2" style="background:var(--surface);border:1px solid var(--border)">
          <span class="chip mono text-xs">{fact.fact_id}</span>
          <textarea rows="2" value={fact.text}
            on:change={e => updateFact(project.project_id, fi, 'text', e.target.value)}
            class="field text-xs mono resize-none w-full" style="min-height:3rem"></textarea>
          <div class="grid grid-cols-2 gap-2">
            <div>
              <label class="text-xs" style="color:var(--muted)">Tools</label>
              <input class="field text-xs mt-0.5" value={fact.tools.join(', ')}
                on:change={e => updateFactArray(project.project_id, fi, 'tools', e.target.value)} />
            </div>
            <div>
              <label class="text-xs" style="color:var(--muted)">Metrics</label>
              <input class="field text-xs mt-0.5"
                style="background:var(--amber-dim);border-color:#92520d;color:var(--amber)"
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

    <div class="flex justify-end pt-1">
      <button class="btn-primary" on:click={goToJD}>
        Paste Job Description →
      </button>
    </div>
  </div>
  {/if}

</div>

<!-- ═════════════════════════════════════════════════════════════════════════
     STEP 2 — JD + AI Recommendation + Generate
     ═════════════════════════════════════════════════════════════════════════ -->
{:else if $step === 2}
<div class="fade-in space-y-6">

  <div>
    <h1 class="text-2xl font-bold" style="color:var(--text)">Job Description</h1>
    <p class="text-sm mt-1" style="color:var(--muted)">
      Paste the JD — AI will recommend which of your projects to highlight.
    </p>
  </div>

  <!-- JD textarea -->
  <div>
    <label class="text-sm font-medium" style="color:var(--text)">Job Description</label>
    <textarea
      bind:value={$jdText}
      rows="9"
      placeholder="Paste the full job description here…&#10;&#10;We are looking for a Senior ML Engineer with expertise in Python, SARIMA forecasting, and production MLOps pipelines…"
      class="field mt-1.5 resize-none leading-relaxed"
      style="font-size:0.8125rem"
    ></textarea>
    <div class="flex justify-end mt-1">
      <span class="text-xs mono" style="color:{$jdText.length < 50 ? 'var(--red)' : 'var(--muted)'}">
        {$jdText.length} chars {$jdText.length < 50 ? '· need 50+' : ''}
      </span>
    </div>
  </div>

  <!-- Settings (compact row) -->
  <div class="surface px-4 py-3">
    <p class="text-xs font-semibold uppercase tracking-wider mb-2.5" style="color:var(--muted)">Settings</p>
    <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
      <div>
        <label class="text-xs" style="color:var(--muted)">Role type</label>
        <select bind:value={$roleType} class="field mt-1 text-xs">
          {#each ROLES as r}<option value={r.value}>{r.label}</option>{/each}
        </select>
      </div>
      <div>
        <label class="text-xs" style="color:var(--muted)">Target chars <span class="chip-amber">±2</span></label>
        <input type="number" bind:value={$charLimit} min="60" max="300" class="field mt-1 text-xs mono" />
      </div>
      <div>
        <label class="text-xs" style="color:var(--muted)">Bullets / project</label>
        <input type="number" bind:value={$maxBullets} min="1" max="8" class="field mt-1 text-xs mono" />
      </div>
      <div>
        <label class="text-xs" style="color:var(--muted)">Projects to recommend</label>
        <input type="number" bind:value={$topK} min="1" max="6" class="field mt-1 text-xs mono" />
      </div>
    </div>
  </div>

  <!-- Analyse JD button -->
  {#if $recommendStatus === 'idle' || $recommendStatus === 'error'}
  <button
    class="btn-primary w-full py-3"
    disabled={$jdText.length < 50}
    on:click={analyseJD}
  >
    Analyse JD — recommend best projects
  </button>
  {#if $recommendStatus === 'error'}
    <div class="rounded-lg px-4 py-3 text-sm" style="background:var(--red-dim);border:1px solid var(--red);color:#fca5a5">
      ⚠ {$recommendError} — <button class="underline" on:click={analyseJD}>retry</button>
    </div>
  {/if}
  {/if}

  <!-- Loading -->
  {#if $recommendStatus === 'loading'}
  <div class="surface px-5 py-5 flex items-center gap-4">
    <div class="w-5 h-5 rounded-full border-2 animate-spin flex-shrink-0" style="border-color:var(--indigo);border-top-color:transparent"></div>
    <div>
      <p class="text-sm font-medium" style="color:var(--text)">Analysing job description…</p>
      <p class="text-xs mt-0.5" style="color:var(--muted)">Scoring all {$parsedProjects.length} projects against the JD</p>
    </div>
  </div>
  {/if}

  <!-- Recommendation results -->
  {#if $recommendStatus === 'done' && $recommendations.length > 0}
  <div class="space-y-3 fade-in">
    <div class="flex items-center justify-between">
      <p class="text-sm font-semibold" style="color:var(--text)">
        AI Recommendation
        <span class="font-normal" style="color:var(--muted)"> — toggle to override</span>
      </p>
      <button class="btn-ghost text-xs" on:click={() => { recommendStatus.set('idle'); }}>
        Re-analyse
      </button>
    </div>

    {#each $recommendations as rec}
    {@const selected = $selectedIds.has(rec.project_id)}
    {@const isRecommended = rec.recommended}
    <div class="rounded-xl overflow-hidden transition-all duration-200 fade-in"
         style="border:1px solid {selected ? (isRecommended ? 'var(--green)' : 'var(--indigo)') : 'var(--border)'};
                background:var(--surface);
                opacity:{!isRecommended && !selected ? '0.6' : '1'}">

      <div class="flex items-start gap-3 px-4 py-3">
        <!-- Toggle checkbox -->
        <button
          class="mt-0.5 w-5 h-5 rounded flex-shrink-0 flex items-center justify-center transition-colors"
          style="background:{selected ? (isRecommended ? 'var(--green)' : 'var(--indigo)') : 'var(--border)'};
                 border:1px solid {selected ? 'transparent' : 'var(--border-hi)'}"
          on:click={() => toggleProject(rec.project_id)}
          aria-label="Toggle {rec.title}"
        >
          {#if selected}<span class="text-white text-xs font-bold">✓</span>{/if}
        </button>

        <div class="flex-1 min-w-0">
          <!-- Title + rank badge -->
          <div class="flex items-center gap-2 flex-wrap">
            <p class="font-semibold text-sm leading-snug" style="color:var(--text)">{rec.title}</p>
            {#if isRecommended}
              <span class="chip chip-green text-xs">#{rec.rank} recommended</span>
            {:else}
              <span class="chip text-xs">#{rec.rank}</span>
            {/if}
          </div>

          <!-- Reason (only for recommended) -->
          {#if isRecommended && rec.reason}
            <p class="text-xs mt-1.5 leading-snug" style="color:var(--muted)">
              💡 {rec.reason}
            </p>
          {/if}

          <!-- Score bar -->
          <div class="mt-2 flex items-center gap-3">
            <div class="flex-1 h-1 rounded-full" style="background:var(--border)">
              <div class="h-1 rounded-full transition-all duration-500"
                   style="width:{scoreBar(rec.score)};background:{scoreColour(rec.score)}"></div>
            </div>
            <span class="text-xs mono flex-shrink-0" style="color:{scoreColour(rec.score)}">
              {Math.round(rec.score * 100)}% match
            </span>
          </div>

          <!-- Matched skills + metrics -->
          <div class="flex flex-wrap gap-1.5 mt-2">
            {#each rec.matched_skills.slice(0,4) as skill}
              <span class="chip chip-indigo">{skill}</span>
            {/each}
            {#each rec.top_metrics.slice(0,2) as m}
              <span class="chip chip-amber">{m}</span>
            {/each}
          </div>
        </div>
      </div>
    </div>
    {/each}

    <!-- Generate -->
    <div class="pt-2 space-y-2">
      {#if [...$selectedIds].filter(id => $parsedProjects.some(p => p.project_id === id)).length === 0}
        <p class="text-xs text-center" style="color:var(--red)">Select at least one project to continue.</p>
      {/if}
      <button
        class="btn-primary w-full py-3"
        disabled={[...$selectedIds].filter(id => $parsedProjects.some(p => p.project_id === id)).length === 0}
        on:click={generate}
      >
        {#if [...$selectedIds].filter(id => $parsedProjects.some(p => p.project_id === id)).length === 1}
          Generate bullets for 1 project →
        {:else}
          Generate bullets for {[...$selectedIds].filter(id => $parsedProjects.some(p => p.project_id === id)).length} projects →
        {/if}
      </button>
    </div>
  </div>
  {/if}

  <!-- Back -->
  <div class="flex justify-start pt-1">
    <button class="btn-ghost" on:click={() => step.set(1)}>← Back</button>
  </div>

</div>

<!-- ═════════════════════════════════════════════════════════════════════════
     STEP 3 — Results
     ═════════════════════════════════════════════════════════════════════════ -->
{:else if $step === 3}
<div class="fade-in space-y-6">

  <div class="flex items-start justify-between gap-4">
    <div>
      <h1 class="text-2xl font-bold" style="color:var(--text)">Your Bullets</h1>
      {#if $genStatus === 'streaming'}
        <p class="text-sm mt-1" style="color:var(--muted)">Generating<span class="cursor-blink"></span></p>
      {:else if $genStatus === 'done'}
        <p class="text-sm mt-1" style="color:var(--muted)">
          {$bullets.length} bullet{$bullets.length !== 1 ? 's' : ''} across {groupedBullets.length} project{groupedBullets.length !== 1 ? 's' : ''} · {$elapsed}s
        </p>
      {/if}
    </div>
    <div class="flex gap-2 flex-shrink-0 flex-wrap">
      {#if $bullets.length > 1}
        <button class="btn-ghost text-xs" on:click={copyAll}>
          {copiedIdx === -1 ? '✓ Copied all' : 'Copy all'}
        </button>
      {/if}
      <button class="btn-ghost text-xs" on:click={() => { step.set(2); resetGeneration(); }}>
        ← Change projects
      </button>
      <button class="btn-ghost text-xs" on:click={() => { step.set(1); resetGeneration(); resetParse(); }}>
        Start over
      </button>
    </div>
  </div>

  <!-- Live typewriter -->
  {#if $tokenBuffer}
    <div class="surface px-5 py-4 mono text-sm leading-relaxed" style="color:var(--indigo);border-color:rgba(99,102,241,0.3)">
      {$tokenBuffer}<span class="cursor-blink"></span>
    </div>
  {/if}

  <!-- Bullets grouped by project -->
  {#each groupedBullets as group}
  <div class="space-y-3 fade-in">
    <div class="flex items-center gap-3">
      <h2 class="text-sm font-semibold" style="color:var(--text)">{group.title}</h2>
      <div class="flex-1 h-px" style="background:var(--border)"></div>
      <span class="chip mono">{group.bullets.length} bullet{group.bullets.length !== 1 ? 's' : ''}</span>
    </div>

    {#each group.bullets as bullet, i}
    {@const globalIdx = $bullets.indexOf(bullet)}
    {@const withinTol = bullet.metadata?.within_tolerance}
    <div class="rounded-xl overflow-hidden fade-in"
         style="border:1px solid {withinTol ? 'var(--border)' : 'rgba(245,158,11,0.3)'};background:var(--surface)">

      <div class="px-5 pt-4 pb-3">
        <p class="mono text-sm leading-relaxed" style="color:var(--text)">{bullet.text}</p>
      </div>

      <div class="px-5 pb-3 flex items-center justify-between gap-3 flex-wrap"
           style="border-top:1px solid var(--border)">
        <div class="flex flex-wrap gap-1.5 pt-2">
          <span class="chip mono">{bullet.metadata?.char_count} / {bullet.metadata?.char_target} chars</span>
          <span class="chip {withinTol ? 'chip-green' : 'chip-amber'}">
            {withinTol ? '✓ within ±2' : '⚠ outside ±2'}
          </span>
          <span class="chip mono">{bullet.metadata?.iterations_taken} iter</span>
          {#if bullet.metadata?.jd_tone}
            <span class="chip">{bullet.metadata.jd_tone.replace(/_/g,' ')}</span>
          {/if}
        </div>
        <button
          class="flex-shrink-0 text-xs px-3 py-1.5 rounded-md transition-all duration-150 mono mt-2"
          style="background:{copiedIdx === globalIdx ? 'var(--green-dim)' : 'var(--border)'};
                 color:{copiedIdx === globalIdx ? 'var(--green)' : 'var(--muted)'};
                 border:1px solid {copiedIdx === globalIdx ? '#065f46' : 'transparent'}"
          on:click={() => copyBullet(bullet.text, globalIdx)}
        >
          {copiedIdx === globalIdx ? '✓ Copied' : 'Copy'}
        </button>
      </div>
    </div>
    {/each}
  </div>
  {/each}

  <!-- Empty state while streaming -->
  {#if $genStatus === 'streaming' && $bullets.length === 0 && !$tokenBuffer}
    <div class="surface px-5 py-8 text-center space-y-2">
      <div class="w-6 h-6 mx-auto rounded-full border-2 animate-spin"
           style="border-color:var(--indigo);border-top-color:transparent"></div>
      <p class="text-sm" style="color:var(--muted)">Scoring facts, retrieving style exemplars…</p>
    </div>
  {/if}

  <!-- Error state -->
  {#if $genStatus === 'error'}
    <div class="rounded-lg px-4 py-3 text-sm" style="background:var(--red-dim);border:1px solid var(--red);color:#fca5a5">
      ⚠ {$genError}
    </div>
  {/if}

</div>
{/if}
