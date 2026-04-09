<script>
  import '../app.css';
  import { step } from '$lib/stores';

  const STEPS = [
    { n: 1, label: 'Upload CV',       icon: '↑' },
    { n: 2, label: 'Job Description', icon: '≡' },
    { n: 3, label: 'Results',         icon: '◆' },
  ];
</script>

<div class="app-shell">

  <!-- ── Ambient background effects ───────────────────────────────────── -->
  <div class="ambient-bg" aria-hidden="true">
    <div class="ambient-orb orb-1"></div>
    <div class="ambient-orb orb-2"></div>
    <div class="bg-dots-layer"></div>
  </div>

  <!-- ── Header ───────────────────────────────────────────────────────── -->
  <header class="app-header">
    <div class="header-inner">
      <!-- Logo -->
      <div class="logo-group">
        <div class="logo-mark">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
            <path d="M12 2L2 7l10 5 10-5-10-5z" fill="url(#lg1)" opacity="0.9"/>
            <path d="M2 17l10 5 10-5M2 12l10 5 10-5" stroke="url(#lg2)" stroke-width="1.5" fill="none" stroke-linecap="round"/>
            <defs>
              <linearGradient id="lg1" x1="2" y1="2" x2="22" y2="12">
                <stop offset="0%" stop-color="#7c3aed"/>
                <stop offset="100%" stop-color="#06b6d4"/>
              </linearGradient>
              <linearGradient id="lg2" x1="2" y1="12" x2="22" y2="22">
                <stop offset="0%" stop-color="#7c3aed"/>
                <stop offset="100%" stop-color="#06b6d4"/>
              </linearGradient>
            </defs>
          </svg>
        </div>
        <span class="logo-text">
          CVon<span class="gradient-text">RAG</span>
        </span>
        <span class="version-chip">v1.1</span>
      </div>

      <!-- Step indicator -->
      <nav class="step-nav" aria-label="Progress">
        {#each STEPS as s, i}
          <div class="step-item" class:step-item--active={$step === s.n} class:step-item--done={$step > s.n}>
            <div class="step-dot {$step === s.n ? 'step-active' : $step > s.n ? 'step-done' : 'step-pending'}">
              {#if $step > s.n}
                <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
                  <path d="M3 8.5l3.5 3.5 6.5-8" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
              {:else}
                {s.n}
              {/if}
            </div>
            <span class="step-label">{s.label}</span>
          </div>
          {#if i < STEPS.length - 1}
            <div class="step-connector" class:step-connector--done={$step > s.n}></div>
          {/if}
        {/each}
      </nav>
    </div>
  </header>

  <!-- ── Main content ─────────────────────────────────────────────────── -->
  <main class="app-main">
    <slot />
  </main>

  <!-- ── Footer ───────────────────────────────────────────────────────── -->
  <footer class="app-footer">
    <span>Powered by Hopes, Dreams and Prayers</span>
    <span class="footer-dot">·</span>
    <span>Built with Coffee, Regret and No Sleep</span>
  </footer>
</div>

<style>
  .app-shell {
    min-height: 100vh;
    position: relative;
    display: flex;
    flex-direction: column;
  }

  /* ── Ambient background ─────────────────────────────────────────────── */
  .ambient-bg {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
  }

  .ambient-orb {
    position: absolute;
    border-radius: 50%;
    filter: blur(100px);
    opacity: 0.35;
  }

  .orb-1 {
    width: 600px; height: 600px;
    top: -200px; left: -100px;
    background: radial-gradient(circle, rgba(124, 58, 237, 0.15) 0%, transparent 70%);
    animation: orbDrift1 20s ease-in-out infinite;
  }

  .orb-2 {
    width: 500px; height: 500px;
    bottom: -150px; right: -100px;
    background: radial-gradient(circle, rgba(6, 182, 212, 0.1) 0%, transparent 70%);
    animation: orbDrift2 25s ease-in-out infinite;
  }

  @keyframes orbDrift1 {
    0%, 100% { transform: translate(0, 0); }
    50%      { transform: translate(60px, 40px); }
  }
  @keyframes orbDrift2 {
    0%, 100% { transform: translate(0, 0); }
    50%      { transform: translate(-40px, -30px); }
  }

  .bg-dots-layer {
    position: absolute;
    inset: 0;
    background-image: radial-gradient(rgba(255,255,255,0.025) 1px, transparent 1px);
    background-size: 28px 28px;
  }

  /* ── Header ─────────────────────────────────────────────────────────── */
  .app-header {
    position: sticky;
    top: 0;
    z-index: 50;
    background: rgba(6, 6, 10, 0.7);
    backdrop-filter: blur(20px) saturate(1.2);
    -webkit-backdrop-filter: blur(20px) saturate(1.2);
    border-bottom: 1px solid rgba(255,255,255,0.04);
  }

  .header-inner {
    max-width: 56rem;
    margin: 0 auto;
    padding: 0.875rem 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
  }

  /* ── Logo ────────────────────────────────────────────────────────────── */
  .logo-group {
    display: flex;
    align-items: center;
    gap: 0.625rem;
  }

  .logo-mark {
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .logo-text {
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text);
  }

  .version-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    font-weight: 500;
    padding: 0.15rem 0.4rem;
    border-radius: 4px;
    background: rgba(255,255,255,0.04);
    color: var(--muted);
    border: 1px solid rgba(255,255,255,0.06);
  }

  /* ── Step nav ───────────────────────────────────────────────────────── */
  .step-nav {
    display: flex;
    align-items: center;
    gap: 0;
  }

  .step-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0 0.25rem;
  }

  .step-label {
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--muted);
    transition: color 0.3s;
    display: none;
  }

  @media (min-width: 640px) {
    .step-label { display: block; }
  }

  .step-item--active .step-label { color: var(--text); }
  .step-item--done .step-label   { color: var(--text-secondary); }

  .step-connector {
    width: 2rem;
    height: 1px;
    background: rgba(255,255,255,0.06);
    margin: 0 0.25rem;
    transition: background 0.5s;
  }

  .step-connector--done {
    background: var(--green);
    box-shadow: 0 0 6px rgba(16, 185, 129, 0.3);
  }

  /* ── Main ────────────────────────────────────────────────────────────── */
  .app-main {
    flex: 1;
    position: relative;
    z-index: 1;
    max-width: 56rem;
    width: 100%;
    margin: 0 auto;
    padding: 2.5rem 1.5rem 4rem;
  }

  /* ── Footer ─────────────────────────────────────────────────────────── */
  .app-footer {
    position: relative;
    z-index: 1;
    text-align: center;
    padding: 1.5rem;
    font-size: 0.7rem;
    color: var(--muted);
    border-top: 1px solid rgba(255,255,255,0.03);
    display: flex;
    justify-content: center;
    gap: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
  }

  .footer-dot { opacity: 0.3; }
</style>
