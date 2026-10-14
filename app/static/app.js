'use strict';
document.querySelectorAll('form').forEach(form => {
  form.addEventListener('submit', event => {
    if (form.dataset.confirm && !window.confirm(form.dataset.confirm)) {
      event.preventDefault(); return;
    }
    const button = form.querySelector('button[data-loading]');
    if (button) {
      button.disabled = true;
      button.textContent = button.dataset.loading;
      form.setAttribute('aria-busy', 'true');
    }
  });
});
