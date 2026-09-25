/* HealthBridge Solutions: lead capture.
   Every form marked data-lead="<type>" is sent to the Make webhook (instant reply,
   Airtable pipeline, founder alert) and, when it has a Formspree action, to Formspree
   as a backup. window.hbLead(data) is exposed for forms with their own handlers. */
(function () {
  'use strict';
  var HOOK = 'https://hook.eu2.make.com/vy7cwiq7ss0kbvvv32dxuny9nyj5kaad';

  function send(data) {
    try {
      data.page = data.page || window.location.pathname;
      data.referrer = data.referrer || document.referrer || '';
      data.utm = data.utm || window.location.search || '';
      return fetch(HOOK, { method: 'POST', mode: 'no-cors', body: new URLSearchParams(data) });
    } catch (e) {
      return Promise.reject(e);
    }
  }
  window.hbLead = send;

  function val(fd, k) { var v = fd.get(k); return v ? String(v).trim() : ''; }

  document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('form[data-lead]').forEach(function (form) {
      form.addEventListener('submit', async function (e) {
        e.preventDefault();
        var btn = form.querySelector('button[type="submit"]');
        var label = btn ? btn.innerHTML : '';
        if (btn) { btn.disabled = true; btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending...'; }
        var fd = new FormData(form);
        if (val(fd, '_gotcha')) return; // honeypot
        var ok = false;
        try {
          await send({
            form: form.getAttribute('data-lead'),
            name: val(fd, 'name'), email: val(fd, 'email'), phone: val(fd, 'phone'),
            facility: val(fd, 'facility'), role: val(fd, 'role')
          });
          ok = true;
        } catch (err) { /* fall through to Formspree */ }
        try {
          if (form.action && form.action.indexOf('formspree.io') > -1) {
            var r = await fetch(form.action, { method: 'POST', body: fd, headers: { Accept: 'application/json' } });
            ok = ok || r.ok;
          }
        } catch (err) { /* ignore */ }
        if (ok) {
          var fields = form.querySelector('[data-lead-fields]');
          var success = form.querySelector('[data-lead-success]');
          if (fields) fields.style.display = 'none';
          if (success) success.classList.add('show');
          if (window.va) window.va('event', { name: 'lead_' + form.getAttribute('data-lead') });
        } else {
          if (btn) { btn.disabled = false; btn.innerHTML = label; }
          window.alert('Something went wrong. Please try again, or message us on WhatsApp at +254 795 822 310.');
        }
      });
    });
  });
})();
