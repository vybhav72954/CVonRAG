<script>
  import { parseCV, optimizeResume } from '$lib/api';
  import {
    step,
    parsedProjects, selectedIds, parseStatus, parseProgress, parseError, resetParse,
    jdText, roleType, charLimit, maxBullets,
    genStatus, tokenBuffer, bullets, genError, elapsed, resetGeneration,
    isUploading, isGenerating,
  } from '$lib/stores';

  // ── Step 1 state ─────────────────────────────────────────────────────────────
  let dragOver   = false;
  let fileInput;

  function onDrop(e) {
    e.preventDefault();
    dragOver = false;
    const file = e.dataTransfer?.files?.[0];
    if (file) uploadFile(file);
  }

  function onFileChange(e) {
    const file = e.target.files?.[0];
    if (file) uploadFile(file);
  }

  async function uploadFile(file) {
    resetParse();
    parseStatus.set('uploading');
    parseProgress.set(`Reading ${file.name}…`);

    await parseCV(file, {
      onProgress: ({ message }) => {
        parseStatus.set('streaming');
        parseProgress.set(message);
      },
      onProject: ({ project }) => {
        parsedProjects.update(ps => [...ps, project]);
        selectedIds.update(s => { s.add(project.project_id); return new Set(s); });
      },
      onDone: () => {
        parseStatus.set('done');
      },
      onError: (msg) => {
        parseStatus.set('error');
        parseError.set(msg);
      },
    });
  }

  function toggleProject(id) {
    selectedIds.update(s => {
      if (s.has(id)) s.delete(id); else s.add(id);
      return new Set(s);
    });
  }

  // Inline fact editing
  let expandedProject = null;

  function toggleExpand(id) {
    expandedProject = expandedProject === id ? null : id;
  }

  function updateFact(projectId, factIdx, field, value) {
    parsedProjects.update(ps => ps.map(p => {
      if (p.project_id !== projectId) return p;
      const facts = [...p.core_facts];
      facts[factIdx] = { ...facts[factIdx], [field]: value };
      return { ...p, core_facts: facts };
    }));
  }

  function updateFactArray(projectId, factIdx, field, rawValue) {
    const arr = rawValue.split(',').map(s => s.trim()).filter(Boolean);
    updateFact(projectId, factIdx, field, arr);
  }

  // ── Step 2 → 3 ───────────────────────────────────────────────────────────────
  function goToStep2() {
    if ($parsedProjects.filter(p => $selectedIds.has(p.project_id)).length > 0) {
      step.set(2);
    }
  }

  async function generate() {
    resetGeneration();
    genStatus.set('streaming');

    const selectedProjects = $parsedProjects.filter(p => $selectedIds.has(p.project_id));

    const payload = {
      job_description:  $jdText,
      target_role_type: $roleType,
      constraints: {
        target_char_limit:      Number($charLimit),
        tolerance:              2,
        bullet_prefix:          '•',
        max_bullets_per_project: Number($maxBullets),
      },
      projects: selectedProjects,
    };

    step.set(3);

    await optimizeResume(payload, {
      onToken:  t => tokenBuffer.update(b => b + t),
      onBullet: b => {
        bullets.update(arr => [...arr, b]);
        tokenBuffer.set('');
      },
      onDone:   d => { elapsed.set(d?.elapsed_seconds ?? 0); genStatus.set('done'); },
      onError:  e => { genError.set(e); genStatus.set('error'); },
    });
  }

  // ── Copy helpers ──────────────────────────────────────────────────────────────
  let copiedIdx = null;

  async function copyBullet(text, idx) {
    await navigator.clipboard.writeText(text);
    copiedIdx = idx;
    setTimeout(() => copiedIdx = null, 2000);
  }

  async function copyAll() {
    const all = $bullets.map(b => b.text).join('\n');
    await navigator.clipboard.writeText(all);
    copiedIdx = -1;
    setTimeout(() => copiedIdx = null, 2000);
  }

  const ROLES = [
    { value: 'ml_engineering',       label: 'ML Engineering' },
    { value: 'data_science',         label: 'Data Science' },
    { value: 'software_engineering', label: 'Software Engineering' },
    { value: 'quant_finance',        label: 'Quant / Finance' },
    { value: 'product_management',   label: 'Product Management' },
    { value: 'general',              label: 'General' },
  ];
</script>

<!-- ═══════════════════════════════════════════════════════════════════════════
     STEP 1 — Upload CV
     ═══════════════════════════════════════════════════════════════════════════ -->
{#if $step === 1}
<div class="fade-in space-y-8">

  <div>
    <h1 class="text-2xl font-bold" style="color:var(--text)">Upload your CV</h1>
    <p class="text-sm mt-1" style="color:var(--muted)">
      .docx or .pdf — projects are extracted automatically, numbers preserved exactly.
    </p>
  </div>

  <!-- Drop zone -->
  {#if $parseStatus === 'idle' || $parseStatus === 'error'}
  <div
    class="border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all duration-200
           {dragOver ? 'border-indigo-500 bg-indigo-950/20' : ''}"
    style="border-color:{dragOver ? 'var(--indigo)' : 'var(--border-hi)'}; background:{dragOver ? 'rgba(99,102,241,0.05)' : 'var(--surface)'}"
    on:dragover|preventDefault={() => dragOver = true}
    on:dragleave={() => dragOver = false}
    on:drop={onDrop}
    on:click={() => fileInput.click()}
    role="button"
    tabindex="0"
    on:keydown={e => e.key === 'Enter' && fileInput.click()}
  >
    <div class="text-4xl mb-3">📄</div>
    <p class="font-medium" style="color:var(--text)">Drop your CV here or click to browse</p>
    <p class="text-xs mt-1" style="color:var(--muted)">.docx (recommended) or .pdf · max 10 MB</p>
    <input bind:this={fileInput} type="file" accept=".docx,.pdf" class="hidden" on:change={onFileChange} />
  </div>

  {#if $parseStatus === 'error'}
    <div class="rounded-lg px-4 py-3 text-sm" style="background:var(--red-dim);border:1px solid var(--red);color:#fca5a5">
      ⚠ {$parseError}
    </div>
  {/if}
  {/if}

  <!-- Progress -->
  {#if $parseStatus === 'uploading' || $parseStatus === 'streaming'}
  <div class="surface px-5 py-4 space-y-3">
    <div class="flex items-center gap-3">
      <div class="w-4 h-4 rounded-full border-2 border-indigo-500 border-t-transparent animate-spin"></div>
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

  <!-- Project list after parse -->
  {#if $parseStatus === 'done' && $parsedProjects.length > 0}
  <div class="space-y-3">
    <div class="flex items-center justify-between">
      <p class="text-sm font-semibold" style="color:var(--text)">
        {$parsedProjects.length} project{$parsedProjects.length !== 1 ? 's' : ''} found
        <span class="font-normal" style="color:var(--muted)"> — select which to use</span>
      </p>
      <button class="btn-ghost text-xs" on:click={resetParse}>Upload different file</button>
    </div>

    {#each $parsedProjects as project (project.project_id)}
    {@const selected = $selectedIds.has(project.project_id)}
    {@const expanded = expandedProject === project.project_id}
    <div class="rounded-xl overflow-hidden transition-all duration-200 fade-in"
         style="border:1px solid {selected ? 'var(--indigo)' : 'var(--border)'}; background:var(--surface)">

      <!-- Project header row -->
      <div class="flex items-start gap-3 px-4 py-3">
        <!-- Checkbox -->
        <button
          class="mt-0.5 w-5 h-5 rounded flex-shrink-0 flex items-center justify-center transition-colors"
          style="background:{selected ? 'var(--indigo)' : 'var(--border)'}; border:1px solid {selected ? 'var(--indigo)' : 'var(--border-hi)'}"
          on:click={() => toggleProject(project.project_id)}
          aria-label="Toggle {project.title}"
        >
          {#if selected}<span class="text-white text-xs">✓</span>{/if}
        </button>

        <!-- Title + key facts preview -->
        <div class="flex-1 min-w-0">
          <p class="font-semibold text-sm leading-snug" style="color:var(--text)">{project.title}</p>
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

        <!-- Expand toggle -->
        <button class="text-xs transition-colors mono flex-shrink-0"
                style="color:var(--muted)"
                on:click={() => toggleExpand(project.project_id)}>
          {expanded ? 'collapse ↑' : 'edit facts ↓'}
        </button>
      </div>

      <!-- Expanded facts editor -->
      {#if expanded}
      <div class="border-t px-4 py-3 space-y-3" style="border-color:var(--border); background:rgba(0,0,0,0.2)">
        <p class="text-xs" style="color:var(--muted)">
          Edit facts before generating.
          <span class="chip chip-amber">amber = metrics</span> — numbers are preserved exactly.
        </p>
        {#each project.core_facts as fact, fi}
        <div class="rounded-lg p-3 space-y-2" style="background:var(--surface);border:1px solid var(--border)">
          <div class="flex items-center gap-2 mb-1">
            <span class="chip mono text-xs">{fact.fact_id}</span>
          </div>
          <textarea
            rows="2"
            value={fact.text}
            on:change={e => updateFact(project.project_id, fi, 'text', e.target.value)}
            class="field text-xs mono resize-none"
            style="min-height:3rem"
          ></textarea>
          <div class="grid grid-cols-2 gap-2">
            <div>
              <label class="text-xs" style="color:var(--muted)">Tools (comma-sep)</label>
              <input
                class="field text-xs mt-0.5"
                value={fact.tools.join(', ')}
                on:change={e => updateFactArray(project.project_id, fi, 'tools', e.target.value)}
              />
            </div>
            <div>
              <label class="text-xs" style="color:var(--muted)">Metrics (comma-sep)</label>
              <input
                class="field text-xs mt-0.5 chip-amber"
                style="background:var(--amber-dim);border-color:#92520d;color:var(--amber)"
                value={fact.metrics.join(', ')}
                on:change={e => updateFactArray(project.project_id, fi, 'metrics', e.target.value)}
              />
            </div>
          </div>
        </div>
        {/each}
      </div>
      {/if}
    </div>
    {/each}

    <!-- Continue -->
    <div class="flex justify-end pt-2">
      <button
        class="btn-primary"
        disabled={[...$selectedIds].filter(id => $parsedProjects.some(p => p.project_id === id)).length === 0}
        on:click={goToStep2}
      >
        Continue with {[...$selectedIds].filter(id => $parsedProjects.some(p => p.project_id === id)).length} project{[...$selectedIds].filter(id => $parsedProjects.some(p => p.project_id === id)).length !== 1 ? 's' : ''} →
      </button>
    </div>
  </div>
  {/if}

</div>

<!-- ═══════════════════════════════════════════════════════════════════════════
     STEP 2 — Job Description
     ═══════════════════════════════════════════════════════════════════════════ -->
{:else if $step === 2}
<div class="fade-in space-y-6">

  <div>
    <h1 class="text-2xl font-bold" style="color:var(--text)">Job Description</h1>
    <p class="text-sm mt-1" style="color:var(--muted)">
      Paste the full JD. The system extracts required skills, tone, and keywords automatically.
    </p>
  </div>

  <!-- JD textarea -->
  <div>
    <label class="text-sm font-medium" style="color:var(--text)">Job Description</label>
    <textarea
      bind:value={$jdText}
      rows="10"
      placeholder="Paste the full job description here…&#10;&#10;We are looking for a Senior ML Engineer with expertise in Python, SARIMA forecasting, and production MLOps pipelines…"
      class="field mt-1.5 resize-none leading-relaxed"
      style="font-size:0.8125rem"
    ></textarea>
    <div class="flex justify-end mt-1">
      <span class="text-xs mono" style="color: {$jdText.length < 50 ? 'var(--red)' : 'var(--muted)'}">
        {$jdText.length} chars {$jdText.length < 50 ? '(need at least 50)' : ''}
      </span>
    </div>
  </div>

  <!-- Settings row -->
  <div class="surface px-5 py-4">
    <p class="text-xs font-semibold uppercase tracking-wider mb-3" style="color:var(--muted)">Generation settings</p>
    <div class="grid grid-cols-3 gap-4">
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
        <label class="text-xs" style="color:var(--muted)">Max bullets / project</label>
        <input type="number" bind:value={$maxBullets} min="1" max="8" class="field mt-1 text-xs mono" />
      </div>
    </div>
  </div>

  <!-- Projects summary -->
  <div class="surface px-5 py-3">
    <p class="text-xs font-semibold uppercase tracking-wider mb-2" style="color:var(--muted)">Selected projects</p>
    <div class="flex flex-wrap gap-2">
      {#each $parsedProjects.filter(p => $selectedIds.has(p.project_id)) as p}
        <span class="chip chip-indigo">{p.title}</span>
      {/each}
    </div>
  </div>

  <!-- Nav -->
  <div class="flex justify-between pt-1">
    <button class="btn-ghost" on:click={() => step.set(1)}>← Back</button>
    <button
      class="btn-primary"
      disabled={$jdText.length < 50}
      on:click={generate}
    >
      Generate Bullets →
    </button>
  </div>

</div>

<!-- ═══════════════════════════════════════════════════════════════════════════
     STEP 3 — Results
     ═══════════════════════════════════════════════════════════════════════════ -->
{:else if $step === 3}
<div class="fade-in space-y-6">

  <div class="flex items-start justify-between gap-4">
    <div>
      <h1 class="text-2xl font-bold" style="color:var(--text)">Your Bullets</h1>
      {#if $genStatus === 'streaming'}
        <p class="text-sm mt-1" style="color:var(--muted)">Generating<span class="cursor-blink"></span></p>
      {:else if $genStatus === 'done'}
        <p class="text-sm mt-1" style="color:var(--muted)">
          {$bullets.length} bullet{$bullets.length !== 1 ? 's' : ''} · {$elapsed}s
        </p>
      {/if}
    </div>
    <div class="flex gap-2 flex-shrink-0">
      {#if $bullets.length > 1}
        <button class="btn-ghost text-xs" on:click={copyAll}>
          {copiedIdx === -1 ? '✓ Copied!' : 'Copy all'}
        </button>
      {/if}
      <button class="btn-ghost text-xs" on:click={() => { step.set(2); resetGeneration(); }}>
        ← Edit JD
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

  <!-- Completed bullets -->
  {#each $bullets as bullet, i}
  {@const withinTol = bullet.metadata?.within_tolerance}
  <div class="rounded-xl overflow-hidden fade-in" style="border:1px solid var(--border);background:var(--surface)">

    <!-- Bullet text -->
    <div class="px-5 pt-4 pb-3">
      <p class="mono text-sm leading-relaxed" style="color:var(--text)">{bullet.text}</p>
    </div>

    <!-- Metadata + actions -->
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
        style="background:{copiedIdx === i ? 'var(--green-dim)' : 'var(--border)'};
               color:{copiedIdx === i ? 'var(--green)' : 'var(--muted)'};
               border:1px solid {copiedIdx === i ? '#065f46' : 'transparent'}"
        on:click={() => copyBullet(bullet.text, i)}
      >
        {copiedIdx === i ? '✓ Copied' : 'Copy'}
      </button>
    </div>
  </div>
  {/each}

  <!-- Error state -->
  {#if $genStatus === 'error'}
    <div class="rounded-lg px-4 py-3 text-sm" style="background:var(--red-dim);border:1px solid var(--red);color:#fca5a5">
      ⚠ {$genError}
    </div>
  {/if}

  <!-- Empty state while streaming -->
  {#if $genStatus === 'streaming' && $bullets.length === 0 && !$tokenBuffer}
    <div class="surface px-5 py-8 text-center space-y-2">
      <div class="w-6 h-6 mx-auto rounded-full border-2 animate-spin"
           style="border-color:var(--indigo);border-top-color:transparent"></div>
      <p class="text-sm" style="color:var(--muted)">Analysing job description and scoring your facts…</p>
    </div>
  {/if}

</div>
{/if}
