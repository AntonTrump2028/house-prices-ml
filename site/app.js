(function () {
  'use strict';

  var COOKIE_NAME = 'hp_country';
  var COOKIE_DAYS = 365;
  var SEARCH_DEBOUNCE_MS = 400;
  var CURRENT_YEAR = new Date().getFullYear();
  var BUY_YEAR_MIN = 1990;
  var BUY_YEAR_MAX = CURRENT_YEAR + 1;

  var DEFAULT_RATES = {
    Poland: 7.5,
    Germany: 3.8,
    USA: 6.8,
    Ukraine: 18,
    Other: 8
  };

  var CURRENCY_BY_COUNTRY = {
    Poland: 'PLN',
    Germany: 'EUR',
    USA: 'USD',
    Ukraine: 'UAH',
    Other: 'у.е.'
  };

  var COUNTRY_CODE_MAP = {
    pl: 'Poland',
    de: 'Germany',
    us: 'USA',
    ua: 'Ukraine'
  };

  var form = document.getElementById('estimate-form');
  var mortgageEl = document.getElementById('mortgage');
  var mortgageFields = document.getElementById('mortgage-fields');
  var countryRateWrap = document.getElementById('country-rate-wrap');
  var customRateWrap = document.getElementById('custom-rate-wrap');
  var countryRateDisplay = document.getElementById('country_rate_display');
  var interestRateEl = document.getElementById('interest_rate');
  var submitBtn = document.getElementById('submit-btn');
  var resultEl = document.getElementById('result');
  var resultPrice = document.getElementById('result-price');
  var resultMortgage = document.getElementById('result-mortgage');
  var resultDown = document.getElementById('result-down');
  var resultMonthly = document.getElementById('result-monthly');
  var apiError = document.getElementById('api-error');

  var addressQuery = document.getElementById('address-query');
  var searchBtn = document.getElementById('search-btn');
  var searchResults = document.getElementById('search-results');
  var searchStatus = document.getElementById('search-status');
  var addressDisplay = document.getElementById('address-display');
  var mapHint = document.getElementById('map-hint');
  var buyYearEl = document.getElementById('buy_year');

  var rates = Object.assign({}, DEFAULT_RATES);
  var searchTimer = null;
  var searchAbort = null;
  var lastResults = [];
  var activeIndex = -1;
  var map = null;
  var marker = null;
  var reduceMotion = false;

  try {
    reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  } catch (e) {
    reduceMotion = false;
  }

  function getCookie(name) {
    var match = document.cookie.match(
      new RegExp('(?:^|; )' + name.replace(/([.$?*|{}()[\]\\/+^])/g, '\\$1') + '=([^;]*)')
    );
    return match ? decodeURIComponent(match[1]) : null;
  }

  function setCookie(name, value, days) {
    var expires = '';
    if (typeof days === 'number') {
      var date = new Date();
      date.setTime(date.getTime() + days * 24 * 60 * 60 * 1000);
      expires = '; expires=' + date.toUTCString();
    }
    document.cookie = name + '=' + encodeURIComponent(value) + expires + '; path=/; SameSite=Lax';
  }

  function formatNumber(value) {
    var n = Number(value);
    if (!isFinite(n)) return '—';
    return Math.round(n).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
  }

  function formatMoney(value, currency) {
    return formatNumber(value) + ' ' + (currency || '');
  }

  function countryFromCode(code) {
    if (!code) return 'Other';
    var key = String(code).toLowerCase();
    return COUNTRY_CODE_MAP[key] || 'Other';
  }

  function pickCity(addr) {
    if (!addr || typeof addr !== 'object') return '';
    return (
      addr.city ||
      addr.town ||
      addr.village ||
      addr.municipality ||
      addr.county ||
      addr.state ||
      ''
    );
  }

  function getSelectedCountry() {
    return form.country.value || 'Other';
  }

  function getRateMode() {
    var checked = form.querySelector('input[name="rate_mode"]:checked');
    return checked ? checked.value : 'country';
  }

  function updateCountryRateDisplay() {
    var country = getSelectedCountry();
    var rate = rates[country] != null ? rates[country] : DEFAULT_RATES.Other;
    countryRateDisplay.value = String(rate);
  }

  function updateRateModeUI() {
    var isCustom = getRateMode() === 'custom';
    countryRateWrap.hidden = isCustom;
    customRateWrap.hidden = !isCustom;
  }

  function updateMortgageUI() {
    var on = mortgageEl.checked;
    mortgageFields.hidden = !on;
    if (on) {
      updateRateModeUI();
      updateCountryRateDisplay();
    }
  }

  function clearFieldErrors() {
    form.querySelectorAll('.field__error').forEach(function (el) {
      el.hidden = true;
      el.textContent = '';
    });
    apiError.hidden = true;
    apiError.textContent = '';
  }

  function showFieldError(id, message) {
    var el = document.getElementById(id + '-error');
    if (!el) return;
    el.textContent = message;
    el.hidden = false;
  }

  function setSearchStatus(message, isError) {
    if (!message) {
      searchStatus.hidden = true;
      searchStatus.textContent = '';
      searchStatus.classList.remove('field__status--error');
      return;
    }
    searchStatus.hidden = false;
    searchStatus.textContent = message;
    searchStatus.classList.toggle('field__status--error', !!isError);
  }

  function closeResults() {
    searchResults.hidden = true;
    searchResults.innerHTML = '';
    addressQuery.setAttribute('aria-expanded', 'false');
    activeIndex = -1;
    lastResults = [];
  }

  function renderResults(items) {
    lastResults = items || [];
    searchResults.innerHTML = '';
    activeIndex = -1;

    if (!lastResults.length) {
      closeResults();
      return;
    }

    lastResults.forEach(function (item, index) {
      var li = document.createElement('li');
      li.setAttribute('role', 'option');
      li.id = 'search-option-' + index;

      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'search__item';
      btn.setAttribute('aria-selected', 'false');
      btn.textContent = item.display_name || '';
      btn.addEventListener('click', function () {
        selectResult(item);
      });

      li.appendChild(btn);
      searchResults.appendChild(li);
    });

    searchResults.hidden = false;
    addressQuery.setAttribute('aria-expanded', 'true');
  }

  function highlightActive() {
    var buttons = searchResults.querySelectorAll('.search__item');
    buttons.forEach(function (btn, i) {
      var selected = i === activeIndex;
      btn.setAttribute('aria-selected', selected ? 'true' : 'false');
      if (selected) btn.scrollIntoView({ block: 'nearest' });
    });
  }

  function setMarker(lat, lon, fly) {
    if (!map || !isFinite(lat) || !isFinite(lon)) return;

    if (marker) {
      marker.setLatLng([lat, lon]);
    } else {
      marker = L.marker([lat, lon]).addTo(map);
    }

    mapHint.hidden = true;

    if (fly && !reduceMotion && typeof map.flyTo === 'function') {
      map.flyTo([lat, lon], 15, { duration: 0.8 });
    } else {
      map.setView([lat, lon], 15);
    }
  }

  function selectResult(item) {
    if (!item) return;

    var lat = Number(item.lat);
    var lon = Number(item.lon);
    var addr = item.address || {};
    var countryCode = addr.country_code || item.country_code || '';
    var country = countryFromCode(countryCode);
    var city = pickCity(addr) || item.city || '';
    var display = item.display_name || '';

    addressDisplay.value = display;
    form.lat.value = isFinite(lat) ? String(lat) : '';
    form.lon.value = isFinite(lon) ? String(lon) : '';
    form.city.value = city;
    form.country.value = country;
    form.country_code.value = countryCode ? String(countryCode).toLowerCase() : '';

    if (countryCode) {
      setCookie(COOKIE_NAME, country, COOKIE_DAYS);
    }

    updateCountryRateDisplay();
    setMarker(lat, lon, true);
    closeResults();
    setSearchStatus('');
    addressQuery.value = display;
  }

  async function searchNominatimDirect(q) {
    var url =
      'https://nominatim.openstreetmap.org/search?format=json&addressdetails=1&limit=8&q=' +
      encodeURIComponent(q);
    var res = await fetch(url, {
      headers: { Accept: 'application/json' },
      signal: searchAbort ? searchAbort.signal : undefined
    });
    if (!res.ok) throw new Error('nominatim ' + res.status);
    return res.json();
  }

  async function runSearch(q) {
    q = (q || '').trim();
    if (q.length < 2) {
      closeResults();
      setSearchStatus(q ? 'Введите не менее 2 символов' : '');
      return;
    }

    if (searchAbort) {
      try {
        searchAbort.abort();
      } catch (e) {}
    }
    searchAbort = typeof AbortController !== 'undefined' ? new AbortController() : null;

    setSearchStatus('Ищем адрес…');
    searchBtn.disabled = true;

    try {
      var items = null;

      try {
        var res = await fetch('/api/search?q=' + encodeURIComponent(q), {
          headers: { Accept: 'application/json' },
          signal: searchAbort ? searchAbort.signal : undefined
        });

        if (res.status === 404) {
          items = await searchNominatimDirect(q);
        } else if (!res.ok) {
          throw new Error('search ' + res.status);
        } else {
          items = await res.json();
        }
      } catch (apiErr) {
        if (apiErr && apiErr.name === 'AbortError') return;
        items = await searchNominatimDirect(q);
      }

      if (!Array.isArray(items)) {
        if (items && Array.isArray(items.results)) items = items.results;
        else items = [];
      }

      if (!items.length) {
        closeResults();
        setSearchStatus('Ничего не найдено. Уточните запрос.', false);
        return;
      }

      renderResults(items);
      setSearchStatus('Найдено: ' + items.length);
    } catch (err) {
      if (err && err.name === 'AbortError') return;
      closeResults();
      setSearchStatus('Не удалось выполнить поиск адреса. Попробуйте снова.', true);
    } finally {
      searchBtn.disabled = false;
    }
  }

  function scheduleSearch() {
    if (searchTimer) clearTimeout(searchTimer);
    searchTimer = setTimeout(function () {
      runSearch(addressQuery.value);
    }, SEARCH_DEBOUNCE_MS);
  }

  function validate() {
    clearFieldErrors();
    var ok = true;

    if (!addressDisplay.value.trim() || !form.lat.value || !form.lon.value) {
      showFieldError('address', 'Выберите адрес из результатов поиска');
      ok = false;
    }

    var year = Number(buyYearEl.value);
    if (
      !buyYearEl.value ||
      !Number.isInteger(year) ||
      year < BUY_YEAR_MIN ||
      year > BUY_YEAR_MAX
    ) {
      showFieldError(
        'buy_year',
        'Укажите год от ' + BUY_YEAR_MIN + ' до ' + BUY_YEAR_MAX
      );
      ok = false;
    }

    var area = Number(form.living_area.value);
    if (!form.living_area.value || !isFinite(area) || area <= 0) {
      showFieldError('living_area', 'Укажите площадь больше 0');
      ok = false;
    }

    var rooms = Number(form.bedrooms.value);
    if (!form.bedrooms.value || !Number.isInteger(rooms) || rooms < 1) {
      showFieldError('bedrooms', 'Укажите целое число комнат ≥ 1');
      ok = false;
    }

    if (mortgageEl.checked) {
      var years = Number(form.loan_years.value);
      if (!form.loan_years.value || !Number.isInteger(years) || years < 1) {
        showFieldError('loan_years', 'Укажите срок в годах');
        ok = false;
      }

      if (getRateMode() === 'custom') {
        var rate = Number(interestRateEl.value);
        if (interestRateEl.value === '' || !isFinite(rate) || rate < 0) {
          showFieldError('interest_rate', 'Укажите ставку');
          ok = false;
        }
      }
    }

    return ok;
  }

  function buildPayload() {
    var mortgage = mortgageEl.checked;
    var rateMode = getRateMode();
    var interestRate = null;

    if (mortgage) {
      if (rateMode === 'custom') {
        interestRate = Number(interestRateEl.value);
      } else {
        interestRate = Number(countryRateDisplay.value);
      }
    }

    var downPct = Number(form.down_payment_pct.value);
    if (!isFinite(downPct)) downPct = 20;

    return {
      country: getSelectedCountry(),
      city: form.city.value.trim(),
      address: addressDisplay.value.trim(),
      buy_year: Number(buyYearEl.value),
      living_area: Number(form.living_area.value),
      bedrooms: Number(form.bedrooms.value),
      lat: Number(form.lat.value),
      lon: Number(form.lon.value),
      mortgage: mortgage,
      loan_years: mortgage ? Number(form.loan_years.value) : null,
      interest_rate: interestRate,
      down_payment_pct: mortgage ? downPct : null,
      rate_mode: mortgage ? rateMode : null
    };
  }

  function pickNumber(obj, keys) {
    for (var i = 0; i < keys.length; i++) {
      var v = obj[keys[i]];
      if (v != null && v !== '' && isFinite(Number(v))) return Number(v);
    }
    return null;
  }

  function renderResult(data) {
    var estimate = pickNumber(data, [
      'estimate',
      'price',
      'predicted_price',
      'prediction'
    ]);
    if (estimate == null) {
      throw new Error('В ответе нет оценки цены');
    }

    var currency =
      data.currency || CURRENCY_BY_COUNTRY[getSelectedCountry()] || 'у.е.';

    resultPrice.textContent = formatMoney(estimate, currency);

    var down = pickNumber(data, ['down_payment', 'downPayment']);
    var monthly = pickNumber(data, ['monthly_payment', 'monthlyPayment']);

    if (mortgageEl.checked && (down != null || monthly != null)) {
      resultMortgage.hidden = false;
      resultDown.textContent = down != null ? formatMoney(down, currency) : '—';
      resultMonthly.textContent =
        monthly != null ? formatMoney(monthly, currency) : '—';
    } else {
      resultMortgage.hidden = true;
      resultDown.textContent = '';
      resultMonthly.textContent = '';
    }

    resultEl.hidden = false;
  }

  function setLoading(loading) {
    submitBtn.disabled = loading;
    submitBtn.textContent = loading ? 'Считаем…' : 'Рассчитать';
  }

  async function loadMeta() {
    try {
      var res = await fetch('/api/meta', { headers: { Accept: 'application/json' } });
      if (!res.ok) return;
      var data = await res.json();
      if (data && data.rates && typeof data.rates === 'object') {
        Object.keys(data.rates).forEach(function (key) {
          var n = Number(data.rates[key]);
          if (isFinite(n)) rates[key] = n;
        });
      }
    } catch (e) {
      // local defaults
    }
  }

  function initMap() {
    if (typeof L === 'undefined') {
      mapHint.textContent = 'Карта недоступна (Leaflet не загрузился).';
      return;
    }

    map = L.map('map', {
      scrollWheelZoom: true,
      zoomControl: true
    }).setView([50, 10], 4);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
    }).addTo(map);

    // Optional: click on map does nothing by default (search-only).
    // Keep map interactive for pan/zoom only.

    setTimeout(function () {
      try {
        map.invalidateSize();
      } catch (e) {}
    }, 100);
  }

  function initBuyYearBounds() {
    buyYearEl.min = String(BUY_YEAR_MIN);
    buyYearEl.max = String(BUY_YEAR_MAX);
    if (!buyYearEl.value) {
      buyYearEl.value = String(CURRENT_YEAR);
    }
  }

  function initCountryFromCookie() {
    var saved = getCookie(COOKIE_NAME);
    if (saved && rates[saved] != null) {
      form.country.value = saved;
    } else if (!form.country.value) {
      form.country.value = 'Other';
    }
  }

  // Events
  searchBtn.addEventListener('click', function () {
    if (searchTimer) clearTimeout(searchTimer);
    runSearch(addressQuery.value);
  });

  addressQuery.addEventListener('input', function () {
    scheduleSearch();
  });

  addressQuery.addEventListener('keydown', function (event) {
    var open = !searchResults.hidden && lastResults.length;

    if (event.key === 'Enter') {
      event.preventDefault();
      if (open && activeIndex >= 0 && lastResults[activeIndex]) {
        selectResult(lastResults[activeIndex]);
      } else {
        if (searchTimer) clearTimeout(searchTimer);
        runSearch(addressQuery.value);
      }
      return;
    }

    if (!open) return;

    if (event.key === 'ArrowDown') {
      event.preventDefault();
      activeIndex = (activeIndex + 1) % lastResults.length;
      highlightActive();
    } else if (event.key === 'ArrowUp') {
      event.preventDefault();
      activeIndex = activeIndex <= 0 ? lastResults.length - 1 : activeIndex - 1;
      highlightActive();
    } else if (event.key === 'Escape') {
      closeResults();
    }
  });

  document.addEventListener('click', function (event) {
    if (!event.target.closest('.search')) {
      closeResults();
    }
  });

  mortgageEl.addEventListener('change', updateMortgageUI);

  form.querySelectorAll('input[name="rate_mode"]').forEach(function (radio) {
    radio.addEventListener('change', updateRateModeUI);
  });

  form.addEventListener('submit', async function (event) {
    event.preventDefault();
    resultEl.hidden = true;
    apiError.hidden = true;

    if (!validate()) return;

    setLoading(true);

    try {
      var res = await fetch('/api/estimate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json'
        },
        body: JSON.stringify(buildPayload())
      });

      var data = null;
      try {
        data = await res.json();
      } catch (parseErr) {
        data = null;
      }

      if (!res.ok) {
        var msg =
          (data && (data.error || data.message)) ||
          'Не удалось получить оценку (код ' + res.status + '). Попробуйте позже.';
        throw new Error(msg);
      }

      if (!data || typeof data !== 'object') {
        throw new Error('Сервер вернул пустой ответ.');
      }

      renderResult(data);
    } catch (err) {
      var text =
        err && err.message
          ? err.message
          : 'API недоступен. Проверьте соединение и попробуйте снова.';
      if (/Failed to fetch|NetworkError|fetch/i.test(String(err && err.message))) {
        text = 'API недоступен. Проверьте соединение и попробуйте снова.';
      }
      apiError.textContent = text;
      apiError.hidden = false;
      resultEl.hidden = true;
    } finally {
      setLoading(false);
    }
  });

  initBuyYearBounds();
  initCountryFromCookie();
  initMap();
  updateMortgageUI();
  updateCountryRateDisplay();
  loadMeta().then(function () {
    updateCountryRateDisplay();
  });
})();
