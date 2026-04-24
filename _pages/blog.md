---
title: ""
permalink: /blog/
author_profile: false
---

{% assign posts = site.posts | sort: "series_index" %}
{% assign standalone = site.posts | where_exp: "p", "p.series_index == nil" | sort: "date" | reverse %}

{% assign total = 31 %}
{% assign published = 0 %}
{% for p in posts %}{% if p.series_index %}{% assign published = published | plus: 1 %}{% endif %}{% endfor %}
{% assign pct = published | times: 100 | divided_by: total %}

{% assign phases_all = "Phase 1 · Foundations|Why MI matters, what it actually is, and who's doing the work.;Phase 2 · Building Blocks|Neurons, weights, layers, and the training process that makes any of this work.;Phase 3 · Transformers|Self-attention, residual streams, the architecture that took over AI.;Phase 4 · Features in Superposition|Why single neurons lie, and how sparse autoencoders surface real features.;Phase 5 · Circuits|Reverse-engineering specific behaviours: IOI, induction heads, grokking.;Phase 6 · Scaling Up|From toy models to frontier-scale interpretability.;Phase 7 · Reading the Model|Logit lens, attribution patching, activation steering.;Phase 8 · Open Problems|Deceptive alignment, model organisms, the research frontier." | split: ";" %}

{% assign phases_live = posts | group_by: "series" %}

<section class="bx-hero">
  <p class="bx-eyebrow">The Series</p>
  <h1 class="bx-title">Mechanistic interpretability, from first principles.</h1>
  <p class="bx-sub">A 31-post curriculum. Beginner to advanced. Interactive where it helps, rigorous where it has to be. Read in order, or jump in wherever.</p>
  <div class="bx-progress">
    <div class="bx-progress__meta">
      <span><b>{{ published }}</b> of {{ total }} posts published</span>
      <span>{{ pct }}% complete</span>
    </div>
    <div class="bx-progress__bar"><span class="bx-progress__fill" style="width: {{ pct }}%"></span></div>
  </div>
  {% if posts.size > 0 %}
    {% assign first = posts.first %}
    <a class="bx-cta" href="{{ first.url | relative_url }}">Start with Post 1 · {{ first.title }} →</a>
  {% endif %}
</section>

<!-- The map: an at-a-glance strip of all 8 phases -->
<section class="bx-map" aria-label="Phase roadmap">
  <div class="bx-map__label">Roadmap</div>
  <ol class="bx-map__list">
    {% for entry in phases_all %}
      {% assign parts = entry | split: "|" %}
      {% assign phase_name = parts[0] %}
      {% assign phase_desc = parts[1] %}
      {% assign phase_num = forloop.index %}
      {% assign phase_count = 0 %}
      {% for grp in phases_live %}{% if grp.name == phase_name %}{% assign phase_count = grp.items.size %}{% endif %}{% endfor %}
      {% if phase_count > 0 %}
        <li class="bx-map__item is-live">
          <a href="#phase-{{ phase_num }}">
            <span class="bx-map__num">{{ phase_num }}</span>
            <span class="bx-map__name">{{ phase_name | split: "·" | last | strip }}</span>
            <span class="bx-map__status">{{ phase_count }} post{% if phase_count != 1 %}s{% endif %} · ready</span>
          </a>
        </li>
      {% else %}
        <li class="bx-map__item is-coming">
          <span class="bx-map__num">{{ phase_num }}</span>
          <span class="bx-map__name">{{ phase_name | split: "·" | last | strip }}</span>
          <span class="bx-map__status">coming soon</span>
        </li>
      {% endif %}
    {% endfor %}
  </ol>
</section>

<!-- Live phases as chapters with a connected trail of posts -->
{% for phase in phases_live %}
  {% assign phase_posts = phase.items | sort: "series_index" %}
  {% assign first_idx = phase_posts.first.series_index %}
  {% assign last_idx = phase_posts.last.series_index %}
  {% assign desc = "" %}
  {% assign phase_num_display = 0 %}
  {% for entry in phases_all %}
    {% assign parts = entry | split: "|" %}
    {% if parts[0] == phase.name %}
      {% assign desc = parts[1] %}
      {% assign phase_num_display = forloop.index %}
    {% endif %}
  {% endfor %}
  {% assign phase_short = phase.name | split: "·" | last | strip %}

<section class="bx-chapter" id="phase-{{ phase_num_display }}">
  <header class="bx-chapter__head">
    <div class="bx-chapter__num">{{ phase_num_display }}</div>
    <div class="bx-chapter__meta">
      <div class="bx-chapter__kicker">Phase {{ phase_num_display }}</div>
      <h2 class="bx-chapter__name">{{ phase_short }}</h2>
      <p class="bx-chapter__desc">{{ desc }}</p>
      <div class="bx-chapter__facts">
        <span>{{ phase_posts.size }} post{% if phase_posts.size != 1 %}s{% endif %}</span>
        <span>·</span>
        <span>Posts {{ first_idx }}{% if first_idx != last_idx %}–{{ last_idx }}{% endif %}</span>
      </div>
    </div>
  </header>

  <ol class="bx-trail">
    {% for post in phase_posts %}
    <li class="bx-trail__item">
      <span class="bx-trail__node">{{ post.series_index }}</span>
      <a class="bx-trail__card" href="{{ post.url | relative_url }}">
        <h3 class="bx-trail__title">{{ post.title }}</h3>
        <p class="bx-trail__excerpt">{{ post.excerpt | strip_html | truncatewords: 34 }}</p>
        <div class="bx-trail__meta">
          <time>{{ post.date | date: "%b %-d, %Y" }}</time>
          {% if post.read_time_label %}<span>·</span><span>{{ post.read_time_label }}</span>{% endif %}
          <span class="bx-trail__arrow" aria-hidden="true">→</span>
        </div>
      </a>
    </li>
    {% endfor %}
  </ol>
</section>
{% endfor %}

<!-- Upcoming phases shown as locked chapters -->
<section class="bx-chapter bx-chapter--coming">
  <header class="bx-chapter__head">
    <div class="bx-chapter__num bx-chapter__num--locked">•</div>
    <div class="bx-chapter__meta">
      <div class="bx-chapter__kicker">What's next</div>
      <h2 class="bx-chapter__name">Phases 3 through 8</h2>
      <p class="bx-chapter__desc">The rest of the curriculum. In order. New posts drop when they're ready.</p>
    </div>
  </header>

  <ol class="bx-upcoming">
    {% for entry in phases_all %}
      {% assign parts = entry | split: "|" %}
      {% assign phase_name = parts[0] %}
      {% assign phase_desc = parts[1] %}
      {% assign phase_num = forloop.index %}
      {% assign live = false %}
      {% for grp in phases_live %}{% if grp.name == phase_name %}{% assign live = true %}{% endif %}{% endfor %}
      {% unless live %}
        <li class="bx-upcoming__item">
          <span class="bx-upcoming__num">{{ phase_num }}</span>
          <div class="bx-upcoming__body">
            <div class="bx-upcoming__name">{{ phase_name | split: "·" | last | strip }}</div>
            <div class="bx-upcoming__desc">{{ phase_desc }}</div>
          </div>
          <span class="bx-upcoming__status">soon</span>
        </li>
      {% endunless %}
    {% endfor %}
  </ol>
</section>

{% if standalone.size > 0 %}
<section class="bx-chapter">
  <header class="bx-chapter__head">
    <div class="bx-chapter__num bx-chapter__num--alt">◆</div>
    <div class="bx-chapter__meta">
      <div class="bx-chapter__kicker">Other</div>
      <h2 class="bx-chapter__name">Standalone posts</h2>
      <p class="bx-chapter__desc">One-offs outside the main series.</p>
    </div>
  </header>
  <ol class="bx-trail">
    {% for post in standalone %}
    <li class="bx-trail__item bx-trail__item--alt">
      <span class="bx-trail__node">·</span>
      <a class="bx-trail__card" href="{{ post.url | relative_url }}">
        <h3 class="bx-trail__title">{{ post.title }}</h3>
        <p class="bx-trail__excerpt">{{ post.excerpt | strip_html | truncatewords: 30 }}</p>
        <div class="bx-trail__meta">
          <time>{{ post.date | date: "%b %-d, %Y" }}</time>
          <span class="bx-trail__arrow" aria-hidden="true">→</span>
        </div>
      </a>
    </li>
    {% endfor %}
  </ol>
</section>
{% endif %}

{% if posts.size == 0 and standalone.size == 0 %}
<div class="nn-blog-empty"><p>No posts yet. Check back soon.</p></div>
{% endif %}
