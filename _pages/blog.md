---
title: ""
permalink: /blog/
author_profile: false
---

<section class="blidx-masthead">
  <h1 class="blidx-title">Writing</h1>
  <p class="blidx-dek">Notes on world models, interpretability, and the papers I keep returning to. In roughly the order they were written.</p>
</section>

{% assign posts = site.posts | sort: "date" | reverse %}
{% assign current_year = "" %}

<div class="blidx-list">
  {% for post in posts %}
    {% assign year = post.date | date: "%Y" %}
    {% if year != current_year %}
      {% if current_year != "" %}</section>{% endif %}
      <section class="blidx-year">
        <h2 class="blidx-year__label">{{ year }}</h2>
      {% assign current_year = year %}
    {% endif %}

    {% assign is_paper = false %}
    {% if post.url contains "notes-on-" %}{% assign is_paper = true %}{% endif %}

    <article class="blidx-item">
      <a class="blidx-item__link" href="{{ post.url | relative_url }}">
        <div class="blidx-item__col-l">
          <time class="blidx-item__date" datetime="{{ post.date | date_to_xmlschema }}">
            {{ post.date | date: "%b %-d" }}
          </time>
          {% if is_paper %}
          <span class="blidx-item__tag">Paper note</span>
          {% endif %}
        </div>
        <div class="blidx-item__col-r">
          <h3 class="blidx-item__title">{{ post.title }}</h3>
          <p class="blidx-item__dek">{{ post.excerpt | strip_html | truncatewords: 30 }}</p>
        </div>
      </a>
    </article>
  {% endfor %}
  </section>
</div>

<style>
  /* ================================================================
     Blog index — restrained editorial layout
     Reference points: Distill.pub, Anthropic research blog
     One column of thought per entry, hairline separators, quiet
     typography. No cards, no badges, no CTAs.
     ================================================================ */

  .blidx-masthead {
    max-width: 720px;
    margin: 0.75rem 0 4.5rem;
    padding: 0;
  }
  .blidx-title {
    font-family: var(--nn-serif) !important;
    font-size: 3.6rem !important;
    line-height: 0.98 !important;
    letter-spacing: -0.03em !important;
    color: var(--nn-ink) !important;
    font-weight: 500 !important;
    margin: 0 0 1.4rem !important;
  }
  .blidx-dek {
    font-family: var(--nn-serif);
    font-size: 1.15rem;
    line-height: 1.55;
    color: var(--nn-body);
    font-style: italic;
    max-width: 560px;
    margin: 0 !important;
  }

  .blidx-list { max-width: 920px; }

  .blidx-year {
    position: relative;
    padding-top: 1rem;
    margin-top: 2.4rem;
  }
  .blidx-year:first-of-type { margin-top: 0; }
  .blidx-year__label {
    font-family: var(--nn-mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.28em !important;
    text-transform: uppercase !important;
    color: var(--nn-soft) !important;
    font-weight: 400 !important;
    margin: 0 0 0.4rem !important;
    padding-bottom: 0.9rem;
    border-bottom: 1px solid var(--nn-ink);
  }

  .blidx-item {
    border-bottom: 1px solid var(--nn-line);
    margin: 0;
  }
  .blidx-item:last-child { border-bottom: none; }
  .blidx-item__link {
    display: grid;
    grid-template-columns: 108px 1fr;
    gap: 2.4rem;
    padding: 1.7rem 0;
    text-decoration: none !important;
    color: inherit !important;
    transition: none;
  }
  .blidx-item__link:hover .blidx-item__title {
    color: var(--nn-accent-dark) !important;
  }
  .blidx-item__link:hover .blidx-item__title::after {
    width: 100%;
  }

  .blidx-item__col-l {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    padding-top: 0.25rem;
  }
  .blidx-item__date {
    font-family: var(--nn-mono);
    font-size: 0.78rem;
    color: var(--nn-muted);
    letter-spacing: 0.02em;
  }
  .blidx-item__tag {
    font-family: var(--nn-mono);
    font-size: 0.66rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--nn-soft);
  }

  .blidx-item__col-r { min-width: 0; }
  .blidx-item__title {
    font-family: var(--nn-serif) !important;
    font-size: 1.5rem !important;
    line-height: 1.2 !important;
    letter-spacing: -0.01em !important;
    color: var(--nn-ink) !important;
    font-weight: 500 !important;
    margin: 0 0 0.55rem !important;
    position: relative;
    display: inline;
    background-image: linear-gradient(to right, var(--nn-accent-dark), var(--nn-accent-dark));
    background-repeat: no-repeat;
    background-position: 0 100%;
    background-size: 0% 1px;
    transition: color 220ms ease, background-size 320ms ease;
  }
  .blidx-item__link:hover .blidx-item__title {
    background-size: 100% 1px;
  }
  .blidx-item__dek {
    font-family: var(--nn-serif);
    font-size: 1.02rem;
    line-height: 1.55;
    color: var(--nn-muted);
    margin: 0.5rem 0 0 !important;
    max-width: 60ch;
    font-style: italic;
  }

  @media (max-width: 720px) {
    .blidx-title { font-size: 2.4rem !important; }
    .blidx-dek { font-size: 1rem; }
    .blidx-item__link {
      grid-template-columns: 1fr;
      gap: 0.5rem;
      padding: 1.4rem 0;
    }
    .blidx-item__col-l {
      flex-direction: row;
      align-items: center;
      gap: 0.9rem;
      padding-top: 0;
    }
    .blidx-item__title { font-size: 1.25rem !important; }
    .blidx-item__dek { font-size: 0.98rem; }
  }
</style>
