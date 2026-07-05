---
title: ""
permalink: /blog/
author_profile: false
---

<section class="bl-hero">
  <p class="bl-eyebrow">Writing</p>
  <h1 class="bl-title">Notes, essays,<br>half-formed ideas.</h1>
  <p class="bl-sub">Mechanistic interpretability, world models, and the papers I keep coming back to. Written when the thought is ready, published in the order it arrived.</p>
  <div class="bl-meta">
    <span data-bl-count>{{ site.posts.size }} entries</span>
    <span class="bl-meta__sep">·</span>
    <span>{{ site.posts.last.date | date: "%Y" }}–{{ site.posts.first.date | date: "%Y" }}</span>
  </div>
</section>

<nav class="bl-filters" data-bl-filters>
  <button class="bl-filter is-active" data-bl-filter="all">All</button>
  <button class="bl-filter" data-bl-filter="essay">Essays</button>
  <button class="bl-filter" data-bl-filter="paper">Paper notes</button>
</nav>

{% assign posts = site.posts | sort: "date" | reverse %}
{% assign total = posts.size %}

<ol class="bl-list" data-bl-list>
  {% for post in posts %}
    {% assign is_paper = false %}
    {% if post.url contains "notes-on-" %}{% assign is_paper = true %}{% endif %}
    {% assign kind = "essay" %}
    {% assign kind_label = "Essay" %}
    {% if is_paper %}{% assign kind = "paper" %}{% assign kind_label = "Paper note" %}{% endif %}
    {% assign index_num = forloop.index | prepend: "000" %}
    {% assign index_num = index_num | slice: -3, 3 %}
    <li class="bl-entry" data-bl-kind="{{ kind }}">
      <a class="bl-row" href="{{ post.url | relative_url }}">
        <div class="bl-row__meta">
          <span class="bl-row__idx">№ {{ index_num }}</span>
          <time class="bl-row__date" datetime="{{ post.date | date_to_xmlschema }}">
            {{ post.date | date: "%b %-d, %Y" }}
          </time>
          {% if post.read_time_label %}
          <span class="bl-row__read">{{ post.read_time_label }}</span>
          {% endif %}
        </div>
        <div class="bl-row__body">
          <span class="bl-row__kind bl-row__kind--{{ kind }}">{{ kind_label }}</span>
          <h2 class="bl-row__title">{{ post.title }}</h2>
          <p class="bl-row__excerpt">{{ post.excerpt | strip_html | truncatewords: 42 }}</p>
          <span class="bl-row__cta">Read <span aria-hidden="true">→</span></span>
        </div>
      </a>
    </li>
  {% endfor %}
</ol>

<div class="bl-empty" data-bl-empty hidden><p>Nothing here yet in that view.</p></div>

<style>
  /* -------- Distill-style blog index (overrides earlier .bl-* rules) -------- */

  .bl-hero {
    max-width: 780px !important;
    margin: 0.5rem 0 3.2rem !important;
    padding: 0 !important;
  }
  .bl-eyebrow {
    font-family: var(--nn-mono) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.32em !important;
    text-transform: uppercase !important;
    color: var(--nn-muted) !important;
    margin: 0 0 1.4rem !important;
  }
  .bl-title {
    font-family: var(--nn-serif) !important;
    font-size: 3rem !important;
    line-height: 1.02 !important;
    letter-spacing: -0.025em !important;
    color: var(--nn-ink) !important;
    font-weight: 500 !important;
    margin: 0 0 1.2rem !important;
  }
  .bl-title br { display: block; }
  .bl-sub {
    font-family: var(--nn-serif) !important;
    font-size: 1.15rem !important;
    color: var(--nn-body) !important;
    line-height: 1.55 !important;
    font-style: italic !important;
    max-width: 620px !important;
    margin: 0 0 1.6rem !important;
  }
  .bl-meta {
    font-family: var(--nn-mono);
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--nn-soft);
    display: flex;
    gap: 0.7rem;
    align-items: center;
  }
  .bl-meta__sep { color: var(--nn-line); }

  .bl-filters {
    display: flex;
    gap: 0.4rem;
    padding: 0 0 1rem;
    margin: 0 0 0.5rem;
    border-bottom: 1px solid var(--nn-line);
  }
  .bl-filter {
    appearance: none;
    background: transparent;
    border: none;
    padding: 0.5rem 0.9rem;
    font-family: var(--nn-mono);
    font-size: 0.74rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--nn-soft);
    cursor: pointer;
    border-radius: 2px;
    transition: color 150ms, background 150ms;
  }
  .bl-filter:hover { color: var(--nn-ink); }
  .bl-filter.is-active {
    color: var(--nn-accent-dark);
    background: var(--nn-accent-soft);
  }

  .bl-list {
    list-style: none !important;
    padding: 0 !important;
    margin: 0 !important;
    max-width: 960px !important;
    display: block !important;
  }
  .bl-entry {
    list-style: none !important;
    margin: 0 !important;
    padding: 0 !important;
    border-bottom: 1px solid var(--nn-line);
  }
  .bl-entry:first-child { border-top: 1px solid var(--nn-line); }
  .bl-entry[hidden] { display: none !important; }

  .bl-row {
    display: grid !important;
    grid-template-columns: 160px 1fr !important;
    gap: 2.2rem !important;
    padding: 2.2rem 0 2.2rem !important;
    text-decoration: none !important;
    color: inherit !important;
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    transition: background 180ms ease;
  }
  .bl-row:hover {
    background: linear-gradient(to right, var(--nn-accent-soft) 0%, rgba(255,255,255,0) 60%) !important;
    transform: none !important;
    box-shadow: none !important;
  }

  .bl-row__meta {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
    padding-top: 0.35rem;
  }
  .bl-row__idx {
    font-family: var(--nn-mono);
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    color: var(--nn-soft);
  }
  .bl-row__date {
    font-family: var(--nn-mono);
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    color: var(--nn-muted);
    text-transform: uppercase;
  }
  .bl-row__read {
    font-family: var(--nn-mono);
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    color: var(--nn-soft);
  }

  .bl-row__body { min-width: 0; }
  .bl-row__kind {
    display: inline-block;
    font-family: var(--nn-mono);
    font-size: 0.66rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    padding: 0.18rem 0.55rem;
    border-radius: 2px;
    margin-bottom: 0.7rem;
    background: transparent;
    border: 1px solid var(--nn-line);
    color: var(--nn-muted);
  }
  .bl-row__kind--paper {
    color: var(--nn-accent-dark);
    border-color: var(--nn-accent);
    background: var(--nn-accent-soft);
  }
  .bl-row__kind--essay {
    color: #2a9e8e;
    border-color: #2a9e8e;
    background: rgba(42, 158, 142, 0.08);
  }
  .bl-row__title {
    font-family: var(--nn-serif) !important;
    font-size: 1.65rem !important;
    line-height: 1.18 !important;
    letter-spacing: -0.012em !important;
    color: var(--nn-ink) !important;
    font-weight: 500 !important;
    margin: 0 0 0.7rem !important;
    transition: color 150ms;
  }
  .bl-row:hover .bl-row__title { color: var(--nn-accent-dark) !important; }
  .bl-row__excerpt {
    font-family: var(--nn-serif) !important;
    font-size: 1.05rem !important;
    color: var(--nn-body) !important;
    line-height: 1.6 !important;
    margin: 0 0 1rem !important;
    max-width: 62ch;
  }
  .bl-row__cta {
    font-family: var(--nn-mono);
    font-size: 0.75rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--nn-accent-dark);
    opacity: 0.55;
    transition: opacity 150ms, letter-spacing 200ms;
  }
  .bl-row:hover .bl-row__cta {
    opacity: 1;
    letter-spacing: 0.22em;
  }

  .bl-empty {
    padding: 3rem 1rem !important;
    text-align: center !important;
    color: var(--nn-muted) !important;
    font-family: var(--nn-serif);
    font-style: italic;
  }

  @media (max-width: 720px) {
    .bl-title { font-size: 2.2rem !important; }
    .bl-title br { display: none; }
    .bl-sub { font-size: 1.02rem !important; }
    .bl-row {
      grid-template-columns: 1fr !important;
      gap: 0.7rem !important;
      padding: 1.8rem 0 !important;
    }
    .bl-row__meta {
      flex-direction: row;
      flex-wrap: wrap;
      gap: 0.8rem;
      padding-top: 0;
    }
    .bl-row__title { font-size: 1.35rem !important; }
    .bl-row__excerpt { font-size: 1rem !important; }
  }
</style>

<script>
(function() {
  const filters = document.querySelectorAll('[data-bl-filter]');
  const entries = document.querySelectorAll('.bl-entry');
  const empty = document.querySelector('[data-bl-empty]');
  const countEl = document.querySelector('[data-bl-count]');
  const totalCount = entries.length;
  if (!filters.length) return;

  function apply(kind) {
    let visible = 0;
    entries.forEach(e => {
      const show = kind === 'all' || e.dataset.blKind === kind;
      e.hidden = !show;
      if (show) visible++;
    });
    if (countEl) countEl.textContent = visible + (visible === 1 ? ' entry' : ' entries');
    if (empty) empty.hidden = visible !== 0;
  }

  filters.forEach(f => {
    f.addEventListener('click', () => {
      filters.forEach(x => x.classList.toggle('is-active', x === f));
      apply(f.dataset.blFilter);
    });
  });

  if (countEl) countEl.textContent = totalCount + (totalCount === 1 ? ' entry' : ' entries');
})();
</script>
