---
title: ""
permalink: /blog/
author_profile: false
---

<section class="bl-hero">
  <p class="bl-eyebrow">The Blog</p>
  <h1 class="bl-title">Notes, essays, and half-formed ideas.</h1>
  <p class="bl-sub">Mechanistic interpretability, research intuition, and whatever else I'm thinking about. Written in order of whenever it happens.</p>
</section>

{% assign posts = site.posts | sort: "date" | reverse %}

{% if posts.size > 0 %}
<ol class="bl-list">
  {% for post in posts %}
  <li class="bl-item">
    <a class="bl-card" href="{{ post.url | relative_url }}">
      <div class="bl-card__meta">
        <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%b %-d, %Y" }}</time>
        {% if post.read_time_label %}<span aria-hidden="true">·</span><span>{{ post.read_time_label }}</span>{% endif %}
      </div>
      <h2 class="bl-card__title">{{ post.title }}</h2>
      <p class="bl-card__excerpt">{{ post.excerpt | strip_html | truncatewords: 36 }}</p>
      <span class="bl-card__arrow" aria-hidden="true">Read →</span>
    </a>
  </li>
  {% endfor %}
</ol>
{% else %}
<div class="bl-empty"><p>No posts yet. Check back soon.</p></div>
{% endif %}
