# House Prices Prediction (Ames Housing)

Учебный ML-проект: от сырых данных Kaggle до Random Forest с честной проверкой.

## Цель
Предсказать `SalePrice` по признакам дома (Kaggle House Prices — Advanced Regression Techniques).

## Структура
```text
Proj/
  data/                 # CSV (не в git — скачай с Kaggle)
  models/               # .joblib (не в git — собери скриптами)
  plots/                # графики EDA
  01_load_data.py ... 09_check_model.py
  data_description.txt
  requirements.txt
  README.md
```

## Быстрый старт
Нужен **Python 3.11**.

1. Скачай с Kaggle в `data/`: `train.csv`, `test.csv`
2. Установи зависимости:

```bash
pip install -r requirements.txt
```

3. Прогон пайплайна:

```bash
python 01_load_data.py
python 02_missing.py
python 03_clean.py
python 04_eda.py
python 05_linear.py
python 06_forest.py
python 07_save_model.py
python 08_tune_forest.py
python 09_check_model.py
```

## Результаты
Базовое сравнение (hold-out 20%, `random_state=42`, числовые признаки):

| Модель | MAE | RMSE | R2 |
|--------|-----|------|----|
| Linear Regression | 19868 | 26501 | 0.870 |
| Random Forest (100 trees) | 17929 | 24674 | 0.887 |

После тюнинга (`08`): отбор по **valid**, отчёт по **test** (60/20/20):

| Модель | MAE | RMSE | R2 |
|--------|-----|------|----|
| Random Forest (best, 300 trees) | 17436 | 24518 | 0.889 |

Артефакт: `models/rf_house_prices_best.joblib` (после `08`). Проверка: `09_check_model.py` → `CHECK PASSED`.

## Выводы
- Сильнее всего с ценой связаны `OverallQual` и `GrLivArea`.
- Random Forest точнее линейной модели.
- Нельзя оценивать full-data модель на holdout из тех же строк (завышенный R2).
- Маркер категорий в очистке: `Missing` (не `None`) — иначе pandas снова сделает NaN при чтении CSV.
- Текстовые признаки пока не в модели — запас для улучшения.

## Заметки
- `LotFrontage` заполняется медианой по всему train до split (мягкая утечка импутации).
- Если редактор выбрал Python 3.13 без пакетов — запускай через `python3.11`.

## Портфолио-сайт
Открой site/index.html в браузере (белый + голубой, этапы как стройка дома).

