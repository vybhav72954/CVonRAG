<script>
  import '../app.css';
  import { step } from '$lib/stores';

  const STEPS = [
    { n: 1, label: 'Upload CV'  },
    { n: 2, label: 'Job Description' },
    { n: 3, label: 'Results'    },
  ];
</script>

<div class="min-h-screen" style="background:var(--bg)">

  <!-- Header -->
  <header style="border-bottom:1px solid var(--border); background:var(--surface)">
    <div class="max-w-3xl mx-auto px-6 py-4 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <span class="text-lg font-bold tracking-tight" style="color:var(--text)">
          CVon<span style="color:var(--indigo)">RAG</span>
        </span>
        <span class="chip mono">v1.1</span>
      </div>

      <!-- Step indicator -->
      <div class="flex items-center gap-2">
        {#each STEPS as s, i}
          <div class="flex items-center gap-2">
            <div class="step-dot {$step === s.n ? 'step-active' : $step > s.n ? 'step-done' : 'step-pending'}">
              {#if $step > s.n}✓{:else}{s.n}{/if}
            </div>
            <span class="text-xs hidden sm:block transition-colors duration-200
                         {$step === s.n ? '' : ''}"
                  style="color: {$step === s.n ? 'var(--text)' : 'var(--muted)'}">
              {s.label}
            </span>
          </div>
          {#if i < STEPS.length - 1}
            <div class="w-6 h-px mx-1" style="background:var(--border-hi)"></div>
          {/if}
        {/each}
      </div>
    </div>
  </header>

  <main class="max-w-3xl mx-auto px-4 py-10">
    <slot />
  </main>
</div>
