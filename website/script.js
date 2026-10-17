/**
 * ATELIER FORM — Web Engineering & Digital Product Studio
 * Production Script: Platform Tabs, Scope Calculator & Contact Flow
 * Standards: Native Vanilla JS, Zero Third-Party Bloat, Full Accessibility
 */

document.addEventListener('DOMContentLoaded', () => {
  initPlatformTabs();
  initScopeCalculator();
  initContactForm();
  initNavScroll();
  initMobileNav();
});

/* ==========================================================================
   1. PLATFORM TABS CONTROLLER
   ========================================================================== */
function initPlatformTabs() {
  const tabButtons = document.querySelectorAll('.platform-tab-btn');
  const panels = document.querySelectorAll('.platform-content-panel');

  if (!tabButtons.length || !panels.length) return;

  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetPlatform = btn.getAttribute('data-platform');

      // Update button states
      tabButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      // Update panel visibility
      panels.forEach(panel => {
        if (panel.id === `platform-${targetPlatform}`) {
          panel.classList.add('active');
        } else {
          panel.classList.remove('active');
        }
      });
    });
  });
}

/* ==========================================================================
   2. INTERACTIVE SCOPE & SPRINT TIMELINE CALCULATOR
   ========================================================================== */
function initScopeCalculator() {
  const scaleButtons = document.querySelectorAll('[data-calc-scale]');
  const platformButtons = document.querySelectorAll('[data-calc-platform]');
  const addonCheckboxes = document.querySelectorAll('.calc-addon');
  
  const timelineDisplay = document.getElementById('calc-timeline-output');
  const deliverablesList = document.getElementById('calc-deliverables-list');
  const applyBtn = document.getElementById('calc-apply-btn');

  if (!timelineDisplay || !deliverablesList) return;

  let currentScale = 'launch'; // 'launch', 'growth', 'flagship'
  let currentPlatform = 'framer'; // 'framer', 'webflow', 'wordpress', 'shopify'

  // Scale Button Click
  scaleButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      scaleButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentScale = btn.getAttribute('data-calc-scale');
      recomputeScope();
    });
  });

  // Platform Button Click
  platformButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      platformButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentPlatform = btn.getAttribute('data-calc-platform');
      recomputeScope();
    });
  });

  // Addon Checkboxes
  addonCheckboxes.forEach(cb => {
    cb.addEventListener('change', () => {
      recomputeScope();
    });
  });

  function recomputeScope() {
    let baseWeeks = 3;
    let baseDeliverables = [
      'Complete high-fidelity Figma file (autolayout + responsive frames)',
      'Fully responsive production build with zero code warnings',
      'Unified design token system (type, colors, spacing)'
    ];

    // Scale calculation
    if (currentScale === 'launch') {
      baseWeeks = 3;
      baseDeliverables.push('1–5 core pages (Home, Product, Pricing, About, Contact)');
    } else if (currentScale === 'growth') {
      baseWeeks = 4;
      baseDeliverables.push('6–12 custom page templates & conversion-optimized flows');
    } else if (currentScale === 'flagship') {
      baseWeeks = 6;
      baseDeliverables.push('12+ custom page templates + modular component library');
    }

    // Platform specific
    const platformNames = {
      framer: 'Framer (Interactive Canvas & Rapid Deploy)',
      webflow: 'Webflow (Client-First System & Clean Classes)',
      wordpress: 'WordPress (Custom Block Architecture & Gutenberg)',
      shopify: 'Shopify (Liquid 2.0 & High-Speed Checkout)'
    };
    baseDeliverables.push(`Native build on ${platformNames[currentPlatform] || 'Target Platform'}`);

    // Addons
    let extraDays = 0;
    addonCheckboxes.forEach(cb => {
      if (cb.checked) {
        if (cb.id === 'addon-cms') {
          baseDeliverables.push('Content-ready CMS collections with relational schemas');
          extraDays += 3;
        } else if (cb.id === 'addon-interactions') {
          baseDeliverables.push('Custom micro-interactions & tactile scroll effects');
          extraDays += 3;
        } else if (cb.id === 'addon-seo') {
          baseDeliverables.push('Technical SEO audit, 301 redirect map & schema markup');
          extraDays += 2;
        } else if (cb.id === 'addon-analytics') {
          baseDeliverables.push('Privacy-friendly analytics & custom conversion tracking');
          extraDays += 1;
        }
      }
    });

    const totalWeeks = Math.ceil(baseWeeks + (extraDays / 5));
    timelineDisplay.textContent = `${totalWeeks} Weeks`;

    // Render deliverables with clean SVG checks (no emojis)
    deliverablesList.innerHTML = baseDeliverables.map(item => `
      <div class="calc-included-item">
        <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M5 13l4 4L19 7" />
        </svg>
        <span>${escapeHtml(item)}</span>
      </div>
    `).join('');
  }

  // Auto-transfer to inquiry form
  if (applyBtn) {
    applyBtn.addEventListener('click', () => {
      const platformSelect = document.getElementById('form-platform');
      const messageField = document.getElementById('form-message');

      if (platformSelect) {
        platformSelect.value = currentPlatform;
      }

      if (messageField) {
        const checkedAddons = Array.from(addonCheckboxes)
          .filter(cb => cb.checked)
          .map(cb => cb.nextElementSibling ? cb.nextElementSibling.textContent.trim() : cb.id);

        const summaryText = `Project Scope: ${currentScale.toUpperCase()} tier on ${currentPlatform.toUpperCase()}. Estimated timeline: ${timelineDisplay.textContent}. Addons: ${checkedAddons.length ? checkedAddons.join(', ') : 'Standard package'}.`;
        
        if (!messageField.value.includes('Project Scope:')) {
          messageField.value = `${summaryText}\n\nOur team goals: `;
        }
      }

      const contactSection = document.getElementById('contact');
      if (contactSection) {
        contactSection.scrollIntoView({ behavior: 'smooth' });
      }

      showToast('Configured scope transferred to project inquiry form.');
    });
  }

  // Initial calculation
  recomputeScope();
}

/* ==========================================================================
   3. INQUIRY FORM CONTROLLER
   ========================================================================== */
function initContactForm() {
  const form = document.getElementById('project-inquiry-form');
  if (!form) return;

  form.addEventListener('submit', (e) => {
    e.preventDefault();

    const nameInput = document.getElementById('form-name');
    const emailInput = document.getElementById('form-email');
    const messageInput = document.getElementById('form-message');

    if (!nameInput.value.trim() || !emailInput.value.trim() || !messageInput.value.trim()) {
      showToast('Please fill in your name, work email, and project overview.');
      return;
    }

    const submitBtn = form.querySelector('button[type="submit"]');
    const originalText = submitBtn.textContent;
    submitBtn.disabled = true;
    submitBtn.textContent = 'Submitting...';

    // Simulate clean human agency intake confirmation
    setTimeout(() => {
      submitBtn.disabled = false;
      submitBtn.textContent = originalText;
      form.reset();
      showToast('Project brief received. We will review your specs within 24 business hours.');
    }, 800);
  });
}

/* ==========================================================================
   4. NAVIGATION SMOOTH SCROLL & ACTIVE TRACKING
   ========================================================================== */
function initNavScroll() {
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.nav-link');

  if (!sections.length || !navLinks.length) return;

  window.addEventListener('scroll', () => {
    let currentId = '';
    const scrollPos = window.scrollY + 120;

    sections.forEach(sec => {
      const top = sec.offsetTop;
      const height = sec.offsetHeight;
      if (scrollPos >= top && scrollPos < top + height) {
        currentId = sec.getAttribute('id');
      }
    });

    navLinks.forEach(link => {
      link.classList.remove('active');
      if (link.getAttribute('href') === `#${currentId}`) {
        link.classList.add('active');
      }
    });
  }, { passive: true });
}

/* ==========================================================================
   5. MOBILE NAVIGATION TOGGLE
   ========================================================================== */
function initMobileNav() {
  const toggleBtn = document.querySelector('.mobile-nav-toggle');
  const navLinks = document.querySelector('.nav-links');

  if (!toggleBtn || !navLinks) return;

  toggleBtn.addEventListener('click', () => {
    const isExpanded = navLinks.style.display === 'flex';
    if (isExpanded) {
      navLinks.style.display = '';
    } else {
      navLinks.style.display = 'flex';
      navLinks.style.flexDirection = 'column';
      navLinks.style.position = 'absolute';
      navLinks.style.top = 'var(--header-height)';
      navLinks.style.left = '0';
      navLinks.style.width = '100%';
      navLinks.style.background = '#111317';
      navLinks.style.borderBottom = '1px solid var(--border-medium)';
      navLinks.style.padding = '24px';
      navLinks.style.gap = '20px';
    }
  });

  // Close mobile nav when clicking a link
  navLinks.querySelectorAll('a').forEach(link => {
    link.addEventListener('click', () => {
      if (window.innerWidth <= 768) {
        navLinks.style.display = '';
      }
    });
  });
}

/* ==========================================================================
   6. UTILITY: TOAST NOTICE
   ========================================================================== */
function showToast(message) {
  let toast = document.getElementById('global-toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'global-toast';
    toast.className = 'toast-notice';
    toast.innerHTML = `
      <svg class="toast-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M5 13l4 4L19 7" />
      </svg>
      <span class="toast-text"></span>
    `;
    document.body.appendChild(toast);
  }

  toast.querySelector('.toast-text').textContent = message;
  toast.classList.add('show');

  setTimeout(() => {
    toast.classList.remove('show');
  }, 4200);
}

function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
