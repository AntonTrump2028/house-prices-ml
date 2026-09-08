(function () {
  'use strict';

  var COOKIE_NAME = 'hp_country';
  var COOKIE_DAYS = 365;

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

  var form = document.getElementById('estimate-form');
  var countryEl = document.getElementById('country');
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

  var rates = Object.assign({}, DEFAULT_RATES);

  function getCookie(name) {
    var match = document.cookie.match(new RegExp('(?:^|; )' + name.replace(/([.$?*|{}()[\]\\/+^])/g, '\\$1') + '=([^;]*)'));
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

  function clearErrors() {
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

  function getRateMode() {
    var checked = form.querySelector('input[name="rate_mode"]:checked');
    return checked ? checked.value : 'country';
  }

  function updateCountryRateDisplay() {
    var country = countryEl.value;
    var rate = rates[country] != null ? rates[country] : DEFAULT_RATES.Other;
    countryRateDisplay.value = String(rate);
  }

  function updateRateModeUI() {
    var mode = getRateMode();
    var isCustom = mode === 'custom';
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

  function validate() {
    clearErrors();
    var ok = true;

    if (!countryEl.value) {
      showFieldError('country', 'Выберите страну');
      ok = false;
    }

    var city = form.city.value.trim();
    if (!city) {
      showFieldError('city', 'Укажите город');
      ok = false;
    }

    var address = form.address.value.trim();
    if (!address) {
      showFieldError('address', 'Укажите адрес');
      ok = false;
    }

    if (!form.buy_date.value) {
      showFieldError('buy_date', 'Укажите дату покупки');
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

    var year = Number(form.year_built.value);
    if (!form.year_built.value || !Number.isInteger(year) || year < 1800 || year > 2100) {
      showFieldError('year_built', 'Укажите год в формате YYYY');
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

    return {
      country: countryEl.value,
      city: form.city.value.trim(),
      address: form.address.value.trim(),
      buy_date: form.buy_date.value,
      living_area: Number(form.living_area.value),
      bedrooms: Number(form.bedrooms.value),
      year_built: Number(form.year_built.value),
      mortgage: mortgage,
      loan_years: mortgage ? Number(form.loan_years.value) : null,
      rate_mode: mortgage ? rateMode : null,
      interest_rate: interestRate
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
    var estimate = pickNumber(data, ['estimate', 'price', 'predicted_price', 'prediction']);
    if (estimate == null) {
      throw new Error('В ответе нет оценки цены');
    }

    var currency =
      data.currency ||
      CURRENCY_BY_COUNTRY[countryEl.value] ||
      'у.е.';

    resultPrice.textContent = formatMoney(estimate, currency);

    var down = pickNumber(data, ['down_payment', 'downPayment']);
    var monthly = pickNumber(data, ['monthly_payment', 'monthlyPayment']);

    if (mortgageEl.checked && (down != null || monthly != null)) {
      resultMortgage.hidden = false;
      resultDown.textContent = down != null ? formatMoney(down, currency) : '—';
      resultMonthly.textContent = monthly != null ? formatMoney(monthly, currency) : '—';
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
      if (Array.isArray(data.countries) && data.countries.length) {
        // keep existing options; only ensure cookie/default still valid
      }
    } catch (e) {
      // local defaults
    }
  }

  function initCountryFromCookie() {
    var saved = getCookie(COOKIE_NAME);
    if (saved) {
      var option = Array.prototype.find.call(countryEl.options, function (opt) {
        return opt.value === saved;
      });
      if (option) countryEl.value = saved;
    }
    setCookie(COOKIE_NAME, countryEl.value, COOKIE_DAYS);
  }

  countryEl.addEventListener('change', function () {
    setCookie(COOKIE_NAME, countryEl.value, COOKIE_DAYS);
    updateCountryRateDisplay();
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
      apiError.textContent =
        err && err.message
          ? err.message
          : 'API недоступен. Проверьте соединение и попробуйте снова.';
      if (/Failed to fetch|NetworkError|fetch/i.test(String(err && err.message))) {
        apiError.textContent = 'API недоступен. Проверьте соединение и попробуйте снова.';
      }
      apiError.hidden = false;
      resultEl.hidden = true;
    } finally {
      setLoading(false);
    }
  });

  initCountryFromCookie();
  updateMortgageUI();
  updateCountryRateDisplay();
  loadMeta().then(function () {
    updateCountryRateDisplay();
  });
})();
