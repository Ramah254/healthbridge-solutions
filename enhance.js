/* HealthBridge Solutions — scroll-triggered reveal animations
   Applies a subtle fade-and-rise to every section, card, and headline
   as the user scrolls. Respects prefers-reduced-motion. */

(function () {
  'use strict';

  if (typeof window === 'undefined') return;

  // Bail out on browsers without IntersectionObserver
  if (!('IntersectionObserver' in window)) return;

  // Respect users who've asked for reduced motion
  if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  document.addEventListener('DOMContentLoaded', function () {
    // Targets that get the reveal treatment
    var selectors = [
      'section > .container > h2',
      'section > .container > .text-center',
      '.product-card',
      '.job-card',
      '.metric-card',
      '.commitment-card',
      '.usp-card',
      '.step-card',
      '.problem-item',
      '.change-card',
      '.journey-step',
      '.founder-card',
      '.results-promise',
      '.faq-item',
      '.trust-strip',
      '.testimonial-card',
      '.email-capture'
    ];

    var targets = document.querySelectorAll(selectors.join(','));

    targets.forEach(function (el) {
      el.classList.add('reveal');
    });

    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          observer.unobserve(entry.target);
        }
      });
    }, {
      threshold: 0.12,
      rootMargin: '0px 0px -40px 0px'
    });

    targets.forEach(function (el) {
      observer.observe(el);
    });
  });
})();
