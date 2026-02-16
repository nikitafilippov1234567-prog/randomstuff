import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LassoCV, ElasticNetCV
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 100)
print("АНАЛИЗ ФАКТОРОВ ВЛИЯНИЯ НА ЦЕНЫ НЕДВИЖИМОСТИ")
print("С учетом малого объема данных (2022-01 до 2025-09)")
print("=" * 100)

# ЗАГРУЗКА ДАННЫХ

print("\n[1/6] Загрузка объединенных данных...")

data_file = r"G:\downloads\housingdata.csv" #Путь
df = pd.read_csv(data_file, sep=";", parse_dates=['date'])

print(f"  Загружено: {len(df)} строк, {len(df.columns)} столбцов")
print(f"  Период: {df['date'].min()} - {df['date'].max()}")

# Фильтруем: с 2022-01 (когда появились Предложения)
df = df[df['date'] >= '2022-01-01'].copy()
df = df[df['date'] <= '2025-09-01'].copy()  # До сентября 2025

print(f"  После фильтрации: {len(df)} месяцев (2022-01 до 2025-09)")

df.set_index('date', inplace=True)


# ФЕДЕРАЛЬНЫЕ ОКРУГА КОТОРЫЕ ЕСТЬ В ДАННЫХ

FEDERAL_DISTRICTS = {
    'Центральный ФО': [
        'Белгородская область', 'Брянская область', 'Владимирская область',
        'Воронежская область', 'Ивановская область', 'Калужская область',
        'Костромская область', 'Курская область', 'Липецкая область',
        'Московская область', 'Орловская область', 'Рязанская область',
        'Смоленская область', 'Тамбовская область', 'Тверская область',
        'Тульская область', 'Ярославская область', 'Москва'
    ],
    'Северо-Западный ФО': [
        'Республика Карелия', 'Республика Коми', 'Архангельская область',
        'Вологодская область', 'Калининградская область', 'Ленинградская область',
        'Мурманская область', 'Новгородская область', 'Псковская область',
        'Санкт-Петербург', 'Ненецкий автономный округ'
    ],
    'Южный ФО': [
        'Республика Адыгея', 'Республика Калмыкия', 'Республика Крым',
        'Краснодарский край', 'Астраханская область', 'Волгоградская область',
        'Ростовская область', 'Севастополь'
    ],
    'Северо-Кавказский ФО': [
        'Республика Дагестан', 'Республика Ингушетия', 'Кабардино-Балкарская Республика',
        'Карачаево-Черкесская Республика', 'Республика Северная Осетия - Алания',
        'Чеченская Республика', 'Ставропольский край'
    ],
    'Приволжский ФО': [
        'Республика Башкортостан', 'Республика Марий Эл', 'Республика Мордовия',
        'Республика Татарстан', 'Удмуртская Республика', 'Чувашская Республика',
        'Пермский край', 'Кировская область', 'Нижегородская область',
        'Оренбургская область', 'Пензенская область', 'Самарская область',
        'Саратовская область', 'Ульяновская область'
    ],
    'Уральский ФО': [
        'Курганская область', 'Свердловская область', 'Тюменская область',
        'Челябинская область', 'Ханты-Мансийский автономный округ',
        'Ямало-Ненецкий автономный округ'
    ],
    'Сибирский ФО': [
        'Республика Алтай', 'Республика Тыва', 'Республика Хакасия',
        'Алтайский край', 'Красноярский край', 'Иркутская область',
        'Кемеровская область', 'Новосибирская область', 'Омская область',
        'Томская область'
    ],
    'Дальневосточный ФО': [
        'Республика Бурятия', 'Республика Саха (Якутия)', 'Забайкальский край',
        'Камчатский край', 'Приморский край', 'Хабаровский край',
        'Амурская область', 'Магаданская область', 'Сахалинская область',
        'Еврейская автономная область', 'Чукотский автономный округ'
    ]
}

def get_federal_district(region):
    """Определение федерального округа по региону"""
    for fo, regions in FEDERAL_DISTRICTS.items():
        if region in regions:
            return fo
    return None

# ПОДГОТОВКА ПАНЕЛЬНЫХ ДАННЫХ

print("\n[2/6] Подготовка панельных данных...")

# Идентифицируем регионы с полными данными
price_cols = [col for col in df.columns if col.startswith('real_estate_deals_primary_market-')]
regions_available = []

print(f" Проверка доступности данных по регионам...")

for col in price_cols:
    region = col.replace('real_estate_deals_primary_market-', '')
    
    # Формируем названия колонок с учетом возможных пробелов
    # Пробуем оба варианта: с подчеркиванием и с пробелом
    housing_variants = [f'housing_completed_{region}', f'housing_completed {region}']
    loans_variants = [f'housing_loans_{region}', f'housing_loans {region}']
    
    # Находим существующие колонки
    housing_col = next((col for col in housing_variants if col in df.columns), None)
    loans_col = next((col for col in loans_variants if col in df.columns), None)
    
    required_cols = {
        'price_primary': f'real_estate_deals_primary_market-{region}',
        'price_secondary': f'real_estate_deals_secondary_market-{region}',
        'housing': housing_col,
        'loans': loans_col,
        'offers_primary': f'predlozheniya-novostroek-{region}',
        'offers_secondary': f'predlozheniya-vtorichnoi-nedvizhimosti-{region}'
    }
    
    # Проверяем наличие колонок
    available = {}
    for key, col_name in required_cols.items():
        available[key] = col_name is not None and col_name in df.columns
    
    # Нужна хотя бы одна цена (первичка или вторичка), жилье и кредиты
    has_price = available['price_primary'] or available['price_secondary']
    
    if has_price and available['housing'] and available['loans']:
        # Проверяем есть ли хоть какие-то предложения
        has_any_offers = available['offers_primary'] or available['offers_secondary']
        
        # Формируем список для проверки пропусков (только реальные названия колонок)
        check_cols = []
        
        # Добавляем housing и loans
        if required_cols['housing']:
            check_cols.append(required_cols['housing'])
        if required_cols['loans']:
            check_cols.append(required_cols['loans'])
        
        # Добавляем цену (приоритет - первичка)
        if available['price_primary']:
            check_cols.append(required_cols['price_primary'])
            price_type = 'primary'
        else:
            check_cols.append(required_cols['price_secondary'])
            price_type = 'secondary'
        
        if available['offers_primary']:
            check_cols.append(required_cols['offers_primary'])
        elif available['offers_secondary']:
            check_cols.append(required_cols['offers_secondary'])
        
        na_count = df[check_cols].isna().sum().sum()
        total_cells = len(df) * len(check_cols)
        completeness = (1 - na_count / total_cells) * 100
        
        # Берем регионы с полнотой > 80%
        if completeness > 80:
            regions_available.append({
                'region': region,
                'price_type': price_type,
                'has_offers': has_any_offers,
                'offers_type': 'primary' if available['offers_primary'] else ('secondary' if available['offers_secondary'] else None),
                'completeness': completeness,
                'na_count': na_count
            })

print(f"  Найдено регионов с полными данными: {len(regions_available)}")

if len(regions_available) == 0:
    print("\n ВНИМАНИЕ: Не найдено регионов с полными данными!")
    print("   Проверим какие колонки есть в датасете:")
    
    # Показываем примеры колонок
    print(f"\n  Примеры колонок цен:")
    for col in price_cols[:5]:
        print(f"    • {col}")
    
    print(f"\n  Примеры колонок housing_completed:")
    housing_cols = [col for col in df.columns if col.startswith('housing_completed_')]
    for col in housing_cols[:5]:
        print(f"    • {col}")
    
    print(f"\n  Примеры колонок housing_loans:")
    loans_cols = [col for col in df.columns if col.startswith('housing_loans_')]
    for col in loans_cols[:5]:
        print(f"    • {col}")
    
    print(f"\n  Примеры колонок predlozheniya:")
    offers_cols = [col for col in df.columns if col.startswith('predlozheniya')]
    for col in offers_cols[:5]:
        print(f"    • {col}")
    
    raise ValueError("Не удалось найти регионы с полными данными. Проверьте названия колонок.")
# Чекаем
print(f"  Топ-5 регионов по полноте данных:")
regions_available_sorted = sorted(regions_available, key=lambda x: x['completeness'], reverse=True)
for r in regions_available_sorted[:5]:
    price_info = "первичка" if r['price_type'] == 'primary' else "вторичка"
    offers_info = f"{r['offers_type']}" if r['has_offers'] else "нет"
    print(f"    • {r['region']:<35s} цена: {price_info}, предложения: {offers_info}, полнота: {r['completeness']:.1f}%")

# Создаем панельный датасет
panel_data = []

for region_info in regions_available:
    region = region_info['region']
    price_type = region_info['price_type']
    fo = get_federal_district(region)
    
    # Определяем колонку с ценой
    if price_type == 'primary':
        price_col = f'real_estate_deals_primary_market-{region}'
    else:
        price_col = f'real_estate_deals_secondary_market-{region}'
    
    # Определяем колонку с предложениями
    if region_info['offers_type'] == 'primary':
        offers_col = f'predlozheniya-novostroek-{region}'
    elif region_info['offers_type'] == 'secondary':
        offers_col = f'predlozheniya-vtorichnoi-nedvizhimosti-{region}'
    else:
        offers_col = None
    
    for date in df.index:
        row = {
            'date': date,
            'region': region,
            'federal_district': fo,
            'market_type': price_type,
            # Зависимая переменная
            'price': df.loc[date, price_col],
            # Независимые
            'rate': df.loc[date, 'Ключевая ставка, %'],
            'inflation': df.loc[date, 'Базовая инфляция по трем месяцам, %'],
            'housing_completed': df.loc[date, f'housing_completed_{region}'],
            'housing_loans': df.loc[date, f'housing_loans_{region}'],
        }
        
        # Добавляем предложения если есть
        if offers_col:
            row['offers'] = df.loc[date, offers_col]
        else:
            row['offers'] = np.nan
        
        panel_data.append(row)

df_panel = pd.DataFrame(panel_data)

print(f"\n  Создан панельный датасет:")
print(f"    Всего строк: {len(df_panel)}")
print(f"    Пропусков в offers: {df_panel['offers'].isna().sum()}")

# Удаляем строки с критичными пропусками (кроме offers)
critical_cols = ['price', 'rate', 'inflation', 'housing_completed', 'housing_loans']
df_panel = df_panel.dropna(subset=critical_cols)

print(f"  После удаления пропусков: {len(df_panel)} строк")

print(f"  Панельный датасет: {len(df_panel)} наблюдений")
print(f"  Регионов: {df_panel['region'].nunique()}")
print(f"  Месяцев: {df_panel['date'].nunique()}")

# СТРАТЕГИЯ 1: RAW DATA (абсолютные значения)

print("\n[3/6] Стратегия 1: Анализ на абсолютных значениях (RAW)...")

# Создаем копию для RAW стратегии
df_raw = df_panel.copy()

# Нормализация (StandardScaler для каждого региона отдельно)
print("  Нормализация данных...")

numeric_cols = ['price', 'housing_completed', 'housing_loans']

df_raw_normalized = []
for region in df_raw['region'].unique():
    region_data = df_raw[df_raw['region'] == region].copy()
    
    scaler = RobustScaler()  # Устойчив к выбросам
    region_data[numeric_cols] = scaler.fit_transform(region_data[numeric_cols])
    
    # Нормализуем offers отдельно если есть
    if not region_data['offers'].isna().all():
        offers_scaler = RobustScaler()
        region_data[['offers']] = offers_scaler.fit_transform(region_data[['offers']].fillna(0))
    
    df_raw_normalized.append(region_data)

df_raw_norm = pd.concat(df_raw_normalized, ignore_index=True)

print(f"     Нормализовано {len(df_raw_norm)} наблюдений")

# СТРАТЕГИЯ 2: RATIO DATA (относительные величины)

print("\n[4/6] Стратегия 2: Анализ на относительных величинах (RATIO)...")

df_ratio = df_panel.copy()

# Вычисляем относительные изменения (% от начального уровня)
print("  📊 Расчет относительных величин...")

for region in df_ratio['region'].unique():
    mask = df_ratio['region'] == region
    
    # Берем первое значение как базу
    for col in numeric_cols:
        base_value = df_ratio.loc[mask, col].iloc[0]
        if base_value > 0:
            df_ratio.loc[mask, f'{col}_ratio'] = (df_ratio.loc[mask, col] / base_value - 1) * 100
        else:
            df_ratio.loc[mask, f'{col}_ratio'] = 0

print(f"     ✓ Рассчитаны relative changes для {len(df_ratio)} наблюдений")

# [5/6] МЕТОДЫ АНАЛИЗА

print("\n[5/6] Анализ влияния факторов на цены...")

output_folder = r"G:\downloads\price_factors_results"
os.makedirs(output_folder, exist_ok=True)

# МЕТОД 1: КОРРЕЛЯЦИОННЫЙ АНАЛИЗ (Spearman)

print("\n  Корреляционный анализ (Spearman)...")

# Подготовка данных для корреляции
corr_cols = ['price', 'rate', 'inflation', 'housing_completed', 'housing_loans']
if df_panel['offers'].notna().sum() > 100:  # Если есть достаточно данных
    corr_cols.append('offers')

corr_data = df_panel[corr_cols].copy()

# Spearman корреляция (устойчива к выбросам и нелинейности)
corr_matrix = corr_data.corr(method='spearman')

print(f"\n     Корреляция с ценой (Spearman):")
price_corr = corr_matrix['price'].drop('price').sort_values(ascending=False)
for factor, corr in price_corr.items():
    direction = "↑" if corr > 0 else "↓"
    if abs(corr) > 0.5:
        significance = "***"
    elif abs(corr) > 0.3:
        significance = "**"
    elif abs(corr) > 0.1:
        significance = "*"
    else:
        significance = ""
    print(f"       {factor:<25s} {direction} {corr:>7.4f} {significance}")

# Сохраняем
corr_matrix.to_csv(f"{output_folder}/correlation_matrix.csv", sep=";")
print(f"\n    Сохранено: correlation_matrix.csv")

# ============================================================================
# МЕТОД 2: ПРОВЕРКА МУЛЬТИКОЛЛИНЕАРНОСТИ (корреляции для панели)
# ============================================================================
print("\n  Проверка мультиколлинеарности (корреляционная матрица)...")

# ВАЖНО: VIF некорректен для панельных данных!
# Используем корреляции между независимыми переменными

# Подготовка данных (без пропусков)
X_cols = ['rate', 'inflation', 'housing_loans', 'housing_completed']
if 'offers' in corr_cols:
    X_cols.append('offers')

corr_X = df_panel[X_cols].dropna()

# Корреляционная матрица между независимыми переменными
corr_matrix_X = corr_X.corr(method='pearson')

print(f"\n     Корреляции между независимыми переменными:")
print(f"     (Порог для беспокойства: |r| > 0.8)")
print()

# Форматированный вывод
col_width = 15
header = "Variable".ljust(col_width)
for col in corr_matrix_X.columns:
    header += col[:12].ljust(col_width)
print(f"     {header}")
print(f"     {'-' * len(header)}")

for idx, row_name in enumerate(corr_matrix_X.index):
    row_str = row_name[:12].ljust(col_width)
    for col_idx, val in enumerate(corr_matrix_X.iloc[idx]):
        if idx == col_idx:
            row_str += "1.00".ljust(col_width)
        elif idx > col_idx:
            row_str += f"{val:.3f}".ljust(col_width)
        else:
            row_str += "".ljust(col_width)
    print(f"     {row_str}")

# Находим максимальные корреляции (кроме диагонали)
max_corrs = []
for i in range(len(corr_matrix_X.columns)):
    for j in range(i+1, len(corr_matrix_X.columns)):
        corr_val = corr_matrix_X.iloc[i, j]
        max_corrs.append({
            'Var1': corr_matrix_X.columns[i],
            'Var2': corr_matrix_X.columns[j],
            'Correlation': corr_val,
            'Abs_Corr': abs(corr_val)
        })

max_corrs_df = pd.DataFrame(max_corrs).sort_values('Abs_Corr', ascending=False)

print(f"\n     Топ-3 самые сильные корреляции:")
for _, row in max_corrs_df.head(3).iterrows():
    status = "ВЫСОКАЯ" if row['Abs_Corr'] > 0.8 else ("❗ УМЕРЕННАЯ" if row['Abs_Corr'] > 0.6 else "Приемлемая")
    print(f"       {status}: {row['Var1']:<20s} ↔ {row['Var2']:<20s}  r = {row['Correlation']:>6.3f}")

# Сохраняем
corr_matrix_X.to_csv(f"{output_folder}/correlation_matrix_X.csv", sep=";")
print(f"\n     Сохранено: correlation_matrix_X.csv")

# Выводы
high_corr = max_corrs_df[max_corrs_df['Abs_Corr'] > 0.8]
if len(high_corr) > 0:
    print(f"\n     ВНИМАНИЕ: {len(high_corr)} пар с корреляцией |r| > 0.8")
    print(f"       Рекомендация: рассмотреть удаление одной из переменных или использовать регуляризацию")
else:
    print(f"\n     Мультиколлинеарность приемлемая (все |r| < 0.8)")
    print(f"       Панельная структура + кластеризованные SE дополнительно компенсируют")

# Сохраняем топ корреляций
max_corrs_df.to_csv(f"{output_folder}/pairwise_correlations.csv", sep=";", index=False)

# МЕТОД 3: LASSO/ELASTIC NET для отбора факторов

print("\n  LASSO/Elastic Net для отбора значимых факторов")

from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler

# Подготовка данных
model_data = df_panel[corr_cols].dropna()
X = model_data.drop('price', axis=1).values
y = model_data['price'].values

# Нормализация
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# LASSO с CV
print(f"     Обучение LASSO (5-fold CV)...")
lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso.fit(X_scaled, y_scaled)

print(f"       • Оптимальная alpha: {lasso.alpha_:.6f}")
print(f"       • R² (CV): {lasso.score(X_scaled, y_scaled):.4f}")

# Elastic Net
print(f"     Обучение Elastic Net (5-fold CV)...")
elastic = ElasticNetCV(cv=5, random_state=42, max_iter=10000, l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99])
elastic.fit(X_scaled, y_scaled)

print(f"       Оптимальная alpha: {elastic.alpha_:.6f}")
print(f"       Оптимальная l1_ratio: {elastic.l1_ratio_:.2f}")
print(f"       R² (CV): {elastic.score(X_scaled, y_scaled):.4f}")

# Сравнение коэффициентов
feature_names = X_cols = model_data.drop('price', axis=1).columns

results_comparison = pd.DataFrame({
    'Factor': feature_names,
    'LASSO_coef': lasso.coef_,
    'ElasticNet_coef': elastic.coef_,
    'LASSO_selected': lasso.coef_ != 0,
    'ElasticNet_selected': elastic.coef_ != 0
})

print(f"\n     Отобранные факторы:")
selected = results_comparison[(results_comparison['LASSO_selected']) | (results_comparison['ElasticNet_selected'])]

for _, row in selected.iterrows():
    lasso_mark = "✓" if row['LASSO_selected'] else "✗"
    elastic_mark = "✓" if row['ElasticNet_selected'] else "✗"
    print(f"       {row['Factor']:<25s} LASSO:{lasso_mark}  Elastic:{elastic_mark}  "
          f"(βL={row['LASSO_coef']:>7.4f}, βE={row['ElasticNet_coef']:>7.4f})")

results_comparison.to_csv(f"{output_folder}/lasso_elasticnet_results.csv", sep=";", index=False)
print(f"\n  Сохранено: lasso_elasticnet_results.csv")

# МЕТОД 4: ПАНЕЛЬНАЯ РЕГРЕССИЯ (FE)

print("\n  Панельная регрессия с фиксированными эффектами")

try:
    from linearmodels.panel import PanelOLS
    
    # Подготовка данных
    panel_reg_data = df_panel[corr_cols + ['region', 'date']].dropna()
    panel_reg_data = panel_reg_data.set_index(['region', 'date'])
    
    # Нормализация для сравнимости коэффициентов
    for col in corr_cols:
        panel_reg_data[f'{col}_norm'] = (panel_reg_data[col] - panel_reg_data[col].mean()) / panel_reg_data[col].std()
    
    # Формула (используем только те факторы, где корреляции не критичны)
    # Проверяем есть ли переменные с очень высокой корреляцией
    high_corr_pairs = max_corrs_df[max_corrs_df['Abs_Corr'] > 0.85]
    
    if len(high_corr_pairs) > 0:
        print(f"     Обнаружены пары с очень высокой корреляцией (|r| > 0.85)")
        print(f"     Исключаем одну переменную из каждой пары для устойчивости оценок")
        
        # Исключаем переменные с высокой корреляцией (берем первую из пары)
        exclude_vars = set()
        for _, row in high_corr_pairs.iterrows():
            exclude_vars.add(row['Var2'])  # Исключаем вторую переменную
        
        ok_factors = [col for col in X_cols if col not in exclude_vars]
    else:
        ok_factors = X_cols
    
    if len(ok_factors) > 0:
        formula_parts = [f"{f}_norm" for f in ok_factors]
        formula = f"price_norm ~ {' + '.join(formula_parts)} + EntityEffects"
        
        print(f"     Формула: {formula}")
        
        # Оценка модели
        model = PanelOLS.from_formula(formula, data=panel_reg_data)
        results = model.fit(cov_type='clustered', cluster_entity=True)
        
        print(f"\n{results.summary}")
        
        # Сохраняем результаты
        with open(f"{output_folder}/panel_regression_summary.txt", 'w', encoding='utf-8') as f:
            f.write(str(results.summary))
        
        # Извлекаем коэффициенты
        panel_coefs = pd.DataFrame({
            'Factor': results.params.index,
            'Coefficient': results.params.values,
            'Std_Error': results.std_errors.values,
            'T_stat': results.tstats.values,
            'P_value': results.pvalues.values
        })
        
        panel_coefs.to_csv(f"{output_folder}/panel_regression_coefficients.csv", sep=";", index=False)
        print(f"\n   Сохранено: panel_regression_summary.txt и panel_regression_coefficients.csv")
    else:
        print(f"     Все факторы имеют высокую взаимную корреляцию, используем регуляризацию")
        print(f"     См. результаты LASSO/Elastic Net для отбора факторов")
        
except Exception as e:
    print(f"     Ошибка при оценке панельной регрессии: {e}")

# ============================================================================
# МЕТОД 5: ROLLING REGRESSION (динамика влияния)
# ============================================================================
print("\n  5️⃣ Rolling regression (изменение влияния во времени)...")

from sklearn.linear_model import LinearRegression

# Параметры
window = 12  # Окно 12 месяцев

# Подготовка
rolling_data = df_panel[corr_cols + ['date']].dropna().sort_values('date')

# Группируем по дате и берем средние (агрегируем по регионам)
rolling_monthly = rolling_data.groupby('date').mean()

if len(rolling_monthly) >= window + 12:  # Минимум данных для rolling
    
    rolling_results = []
    
    for i in range(window, len(rolling_monthly)):
        window_data = rolling_monthly.iloc[i-window:i]
        
        X_window = window_data.drop('price', axis=1).values
        y_window = window_data['price'].values
        
        # Нормализация
        X_scaled = (X_window - X_window.mean(axis=0)) / (X_window.std(axis=0) + 1e-8)
        y_scaled = (y_window - y_window.mean()) / (y_window.std() + 1e-8)
        
        # Регрессия
        model = LinearRegression()
        model.fit(X_scaled, y_scaled)
        
        result = {
            'date': rolling_monthly.index[i],
            'r2': model.score(X_scaled, y_scaled)
        }
        
        for j, col in enumerate(window_data.drop('price', axis=1).columns):
            result[f'coef_{col}'] = model.coef_[j]
        
        rolling_results.append(result)
    
    rolling_df = pd.DataFrame(rolling_results)
    
    print(f"     Выполнено {len(rolling_df)} rolling регрессий (окно {window} мес)")
    print(f"     Период: {rolling_df['date'].min()} - {rolling_df['date'].max()}")
    print(f"     Средний R²: {rolling_df['r2'].mean():.4f}")
    
    rolling_df.to_csv(f"{output_folder}/rolling_regression_results.csv", sep=";", index=False)
    print(f"\n     Сохранено: rolling_regression_results.csv")
    
    # Визуализация динамики коэффициентов
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # График 1: R²
    axes[0].plot(rolling_df['date'], rolling_df['r2'], linewidth=2, color='darkblue')
    axes[0].set_title('Динамика качества модели (Rolling R²)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('R²')
    axes[0].grid(alpha=0.3)
    axes[0].axhline(y=rolling_df['r2'].mean(), color='red', linestyle='--', label=f'Среднее: {rolling_df["r2"].mean():.3f}')
    axes[0].legend()
    
    # График 2: Коэффициенты
    coef_cols = [col for col in rolling_df.columns if col.startswith('coef_')]
    for col in coef_cols:
        factor_name = col.replace('coef_', '')
        axes[1].plot(rolling_df['date'], rolling_df[col], label=factor_name, linewidth=2)
    
    axes[1].set_title('Динамика влияния факторов (Rolling коэффициенты)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Нормализованный коэффициент')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].grid(alpha=0.3)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{output_folder}/rolling_regression_dynamics.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"     Сохранено: rolling_regression_dynamics.png")
else:
    print(f"     Недостаточно данных для rolling regression")
    print(f"     Нужно минимум {window + 12} месяцев, есть {len(rolling_monthly)}")

# ============================================================================
# [6/6] ИТОГОВЫЕ ВЫВОДЫ
# ============================================================================
print("\n[6/6] Формирование итоговых выводов...")

# Собираем все результаты в одну таблицу
summary_data = []

# Из корреляций
for factor in price_corr.index:
    summary_data.append({
        'Factor': factor,
        'Spearman_Correlation': price_corr[factor],
        'Correlation_Significance': '***' if abs(price_corr[factor]) > 0.5 else ('**' if abs(price_corr[factor]) > 0.3 else '*')
    })

summary_df = pd.DataFrame(summary_data)

# Добавляем корреляции (максимальная абсолютная корреляция для каждого фактора)
max_corr_per_factor = []
for factor in summary_df['Factor']:
    if factor in corr_matrix_X.columns:
        # Находим максимальную корреляцию с другими факторами
        corrs_with_others = corr_matrix_X[factor].drop(factor)
        max_corr = corrs_with_others.abs().max()
        max_corr_per_factor.append(max_corr)
    else:
        max_corr_per_factor.append(np.nan)

summary_df['Max_Correlation'] = max_corr_per_factor

# Добавляем LASSO/Elastic Net
summary_df = summary_df.merge(
    results_comparison[['Factor', 'LASSO_coef', 'ElasticNet_coef', 'LASSO_selected', 'ElasticNet_selected']], 
    on='Factor', how='left'
)

# Сортируем по абсолютной корреляции
summary_df['abs_corr'] = summary_df['Spearman_Correlation'].abs()
summary_df = summary_df.sort_values('abs_corr', ascending=False).drop('abs_corr', axis=1)

print(f"\n ИТОГОВАЯ ТАБЛИЦА ФАКТОРОВ:\n")
print(summary_df.to_string(index=False))

summary_df.to_csv(f"{output_folder}/FINAL_SUMMARY.csv", sep=";", index=False)

print(f"\n\n{'='*100}")
print("АНАЛИЗ ЗАВЕРШЕН")
print(f"{'='*100}")
print(f"\n Все результаты сохранены в: {output_folder}")
print(f"\n Созданные файлы:")
print(f"   1. correlation_matrix.csv - матрица корреляций")
print(f"   2. vif_results.csv - проверка мультиколлинеарности")
print(f"   3. lasso_elasticnet_results.csv - отбор факторов")
print(f"   4. panel_regression_summary.txt - результаты панельной регрессии")
print(f"   5. panel_regression_coefficients.csv - коэффициенты панели")
print(f"   6. rolling_regression_results.csv - динамика влияния")
print(f"   7. rolling_regression_dynamics.png - график динамики")
print(f"   8. FINAL_SUMMARY.csv - ИТОГОВАЯ СВОДНАЯ ТАБЛИЦА")

print(f"\n  КЛЮЧЕВЫЕ ВЫВОДЫ:")
print(f"\n   Самые сильные факторы (по корреляции):")
for i, row in summary_df.head(3).iterrows():
    direction = "положительно" if row['Spearman_Correlation'] > 0 else "отрицательно"
    print(f"     {i+1}. {row['Factor']:<25s} влияет {direction:>15s} (ρ={row['Spearman_Correlation']:>7.4f})")

print(f"\n   Отобранные факторы (LASSO+Elastic Net):")
selected_factors = summary_df[(summary_df['LASSO_selected']) | (summary_df['ElasticNet_selected'])]
if len(selected_factors) > 0:
    for _, row in selected_factors.iterrows():
        print(f"     • {row['Factor']}")
else:
    print(f"     ⚠ Все факторы отброшены (слишком сильная регуляризация или коллинеарность)")

print(f"\n   Мультиколлинеарность:")
high_corr_pairs = max_corrs_df[max_corrs_df['Abs_Corr'] > 0.8]
if len(high_corr_pairs) > 0:
    print(f"     ⚠ {len(high_corr_pairs)} пар с высокой корреляцией (|r| > 0.8):")
    for _, row in high_corr_pairs.iterrows():
        print(f"       • {row['Var1']} ↔ {row['Var2']} (r={row['Correlation']:.2f})")
else:
    print(f"     ✓ Мультиколлинеарность приемлемая (все корреляции |r| < 0.8)")

print(f"\n{'='*100}\n")

# ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ: ВЛИЯНИЕ СТАВКИ НА КРЕДИТЫ И ДОСТУПНОСТЬ

print("\n" + "="*100)
print("ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ: СТАВКА → КРЕДИТЫ → ДОСТУПНОСТЬ")
print("="*100)

# 1. ВЛИЯНИЕ СТАВКИ НА ОБЪЕМ КРЕДИТОВ

print("\n  Анализ 1: Влияние ключевой ставки на объем жилищных кредитов")

# Вычисляем доступность (кредиты/цена)
df_panel['affordability'] = df_panel['housing_loans'] / df_panel['price']

# Корреляции
rate_loans_corr = df_panel[['rate', 'housing_loans']].corr(method='spearman').loc['rate', 'housing_loans']
rate_afford_corr = df_panel[['rate', 'affordability']].corr(method='spearman').loc['rate', 'affordability']

print(f"\n     Корреляции (Spearman):")
print(f"       • Ставка → Объем кредитов:  ρ = {rate_loans_corr:>7.4f}")
print(f"       • Ставка → Доступность:     ρ = {rate_afford_corr:>7.4f}")

# Регрессия: loans = f(rate)
from sklearn.linear_model import LinearRegression

# Подготовка данных
analysis_data = df_panel[['rate', 'housing_loans', 'affordability', 'price']].dropna()

# Модель 1: Кредиты от ставки
X_rate = analysis_data[['rate']].values
y_loans = analysis_data['housing_loans'].values

# Нормализация
X_rate_scaled = (X_rate - X_rate.mean()) / X_rate.std()
y_loans_scaled = (y_loans - y_loans.mean()) / y_loans.std()

model_loans = LinearRegression()
model_loans.fit(X_rate_scaled, y_loans_scaled)

r2_loans = model_loans.score(X_rate_scaled, y_loans_scaled)

print(f"\n     Регрессия: housing_loans = β₀ + β₁ × rate")
print(f"       • Коэффициент β₁:  {model_loans.coef_[0]:>7.4f}")
print(f"       • R²:              {r2_loans:>7.4f}")
print(f"       • Интерпретация:   Повышение ставки на 1 п.п. → изменение объема кредитов на {model_loans.coef_[0]:.2f} σ")

# Модель 2: Доступность от ставки
y_afford = analysis_data['affordability'].values
y_afford_scaled = (y_afford - y_afford.mean()) / y_afford.std()

model_afford = LinearRegression()
model_afford.fit(X_rate_scaled, y_afford_scaled)

r2_afford = model_afford.score(X_rate_scaled, y_afford_scaled)

print(f"\n     Регрессия: affordability = β₀ + β₁ × rate")
print(f"       • Коэффициент β₁:  {model_afford.coef_[0]:>7.4f}")
print(f"       • R²:              {r2_afford:>7.4f}")
print(f"       • Интерпретация:   Повышение ставки на 1 п.п. → изменение доступности на {model_afford.coef_[0]:.2f} σ")

# 2. ПАНЕЛЬНАЯ РЕГРЕССИЯ ДЛЯ КРЕДИТОВ И ДОСТУПНОСТИ

print("\n  Анализ 2: Панельная регрессия с фиксированными эффектами")

try:
    from linearmodels.panel import PanelOLS
    
    # Подготовка панельных данных
    panel_credit_data = df_panel[['region', 'date', 'rate', 'housing_loans', 'affordability', 'price']].dropna()
    panel_credit_data = panel_credit_data.set_index(['region', 'date'])
    
    # Нормализация
    for col in ['rate', 'housing_loans', 'affordability', 'price']:
        panel_credit_data[f'{col}_norm'] = (panel_credit_data[col] - panel_credit_data[col].mean()) / panel_credit_data[col].std()
    
    # Модель 1: Кредиты от ставки + рег эффекты
    print(f"\n     Модель 1: housing_loans ~ rate + EntityEffects")
    
    model_panel_loans = PanelOLS.from_formula(
        'housing_loans_norm ~ rate_norm + EntityEffects',
        data=panel_credit_data
    )
    results_panel_loans = model_panel_loans.fit(cov_type='clustered', cluster_entity=True)
    
    beta_rate_loans = results_panel_loans.params['rate_norm']
    pval_rate_loans = results_panel_loans.pvalues['rate_norm']
    r2_panel_loans = results_panel_loans.rsquared
    
    print(f"       • β(rate):   {beta_rate_loans:>7.4f}  (p={pval_rate_loans:.4f})")
    print(f"       • R²:        {r2_panel_loans:>7.4f}")
    
    # Модель 2: Доступность от ставки + рег эффекты
    print(f"\n     Модель 2: affordability ~ rate + EntityEffects")
    
    model_panel_afford = PanelOLS.from_formula(
        'affordability_norm ~ rate_norm + EntityEffects',
        data=panel_credit_data
    )
    results_panel_afford = model_panel_afford.fit(cov_type='clustered', cluster_entity=True)
    
    beta_rate_afford = results_panel_afford.params['rate_norm']
    pval_rate_afford = results_panel_afford.pvalues['rate_norm']
    r2_panel_afford = results_panel_afford.rsquared
    
    print(f"       • β(rate):   {beta_rate_afford:>7.4f}  (p={pval_rate_afford:.4f})")
    print(f"       • R²:        {r2_panel_afford:>7.4f}")
    
    # Сохраняем результаты
    panel_credit_results = pd.DataFrame({
        'Model': ['Loans ~ Rate', 'Affordability ~ Rate'],
        'Beta_rate': [beta_rate_loans, beta_rate_afford],
        'P_value': [pval_rate_loans, pval_rate_afford],
        'R_squared': [r2_panel_loans, r2_panel_afford]
    })
    
    panel_credit_results.to_csv(f"{output_folder}/rate_credit_affordability_analysis.csv", sep=";", index=False)
    
    # Детальные результаты
    with open(f"{output_folder}/rate_credit_affordability_detailed.txt", 'w', encoding='utf-8') as f:
        f.write("МОДЕЛЬ 1: КРЕДИТЫ ОТ СТАВКИ\n")
        f.write("="*80 + "\n")
        f.write(str(results_panel_loans.summary))
        f.write("\n\n")
        f.write("МОДЕЛЬ 2: ДОСТУПНОСТЬ ОТ СТАВКИ\n")
        f.write("="*80 + "\n")
        f.write(str(results_panel_afford.summary))
    
    print(f"\n     ✅ Сохранено:")
    print(f"        • rate_credit_affordability_analysis.csv")
    print(f"        • rate_credit_affordability_detailed.txt")
    
except Exception as e:
    print(f"     ⚠ Ошибка: {e}")

# ============================================================================
# 3. ДИНАМИКА ПО ВРЕМЕНИ (Rolling)
# ============================================================================
print("\n  Анализ 3: Динамика влияния ставки во времени (Rolling)")

# Группируем по месяцам
monthly_agg = df_panel.groupby('date').agg({
    'rate': 'mean',
    'housing_loans': 'mean',
    'affordability': 'mean',
    'price': 'mean'
}).reset_index()

if len(monthly_agg) >= 18:
    window = 12
    rolling_credit_results = []
    
    for i in range(window, len(monthly_agg)):
        window_data = monthly_agg.iloc[i-window:i]
        
        X = window_data[['rate']].values
        y_loans = window_data['housing_loans'].values
        y_afford = window_data['affordability'].values
        
        # Нормализация
        X_scaled = (X - X.mean()) / (X.std() + 1e-8)
        y_loans_scaled = (y_loans - y_loans.mean()) / (y_loans.std() + 1e-8)
        y_afford_scaled = (y_afford - y_afford.mean()) / (y_afford.std() + 1e-8)
        
        # Регрессии
        model_l = LinearRegression().fit(X_scaled, y_loans_scaled)
        model_a = LinearRegression().fit(X_scaled, y_afford_scaled)
        
        rolling_credit_results.append({
            'date': monthly_agg.iloc[i]['date'],
            'beta_loans': model_l.coef_[0],
            'r2_loans': model_l.score(X_scaled, y_loans_scaled),
            'beta_affordability': model_a.coef_[0],
            'r2_affordability': model_a.score(X_scaled, y_afford_scaled)
        })
    
    rolling_credit_df = pd.DataFrame(rolling_credit_results)
    
    print(f"     Выполнено {len(rolling_credit_df)} rolling регрессий")
    print(f"       Средний β(loans):        {rolling_credit_df['beta_loans'].mean():>7.4f}")
    print(f"       Средний β(affordability): {rolling_credit_df['beta_affordability'].mean():>7.4f}")
    
    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # График 1: Динамика ставки и кредитов
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    
    ax1.plot(monthly_agg['date'], monthly_agg['rate'], 'b-', linewidth=2, label='Ключевая ставка')
    ax1_twin.plot(monthly_agg['date'], monthly_agg['housing_loans'], 'r-', linewidth=2, label='Объем кредитов')
    
    ax1.set_ylabel('Ключевая ставка, %', color='b')
    ax1_twin.set_ylabel('Объем кредитов, млн руб', color='r')
    ax1.set_title('Динамика ставки и объема кредитов', fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    
    # График 2: Динамика коэффициента влияния на кредиты
    axes[0, 1].plot(rolling_credit_df['date'], rolling_credit_df['beta_loans'], 'g-', linewidth=2)
    axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[0, 1].set_ylabel('β (нормализованный)')
    axes[0, 1].set_title('Динамика влияния ставки на кредиты (Rolling)', fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # График 3: Динамика ставки и доступности
    ax3 = axes[1, 0]
    ax3_twin = ax3.twinx()
    
    ax3.plot(monthly_agg['date'], monthly_agg['rate'], 'b-', linewidth=2, label='Ключевая ставка')
    ax3_twin.plot(monthly_agg['date'], monthly_agg['affordability'], 'purple', linewidth=2, label='Доступность')
    
    ax3.set_ylabel('Ключевая ставка, %', color='b')
    ax3_twin.set_ylabel('Доступность (кредиты/цена)', color='purple')
    ax3.set_title('Динамика ставки и доступности', fontweight='bold')
    ax3.grid(alpha=0.3)
    ax3.tick_params(axis='y', labelcolor='b')
    ax3_twin.tick_params(axis='y', labelcolor='purple')
    
    # График 4: Динамика коэффициента влияния на доступность
    axes[1, 1].plot(rolling_credit_df['date'], rolling_credit_df['beta_affordability'], 'orange', linewidth=2)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1, 1].set_ylabel('β (нормализованный)')
    axes[1, 1].set_title('Динамика влияния ставки на доступность (Rolling)', fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_folder}/rate_credit_affordability_dynamics.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Сохраняем данные
    rolling_credit_df.to_csv(f"{output_folder}/rate_credit_affordability_rolling.csv", sep=";", index=False)
    
    print(f"\n   Сохранено:")
    print(f"        • rate_credit_affordability_rolling.csv")
    print(f"        • rate_credit_affordability_dynamics.png")
else:
    print(f"     Недостаточно данных для rolling анализа")

# ИТОГИ ДОПОЛНИТЕЛЬНОГО АНАЛИЗА

print("\n" + "="*100)
print("ВЫВОДЫ ПО ВЛИЯНИЮ СТАВКИ НА КРЕДИТЫ И ДОСТУПНОСТЬ")
print("="*100)

print(f"\n Корреляционный анализ:")
print(f"   • Ставка → Кредиты:     ρ = {rate_loans_corr:.4f}")
if rate_loans_corr < -0.3:
    print(f"     → Сильная ОТРИЦАТЕЛЬНАЯ связь: рост ставки → снижение кредитов")
elif rate_loans_corr < 0:
    print(f"     → Слабая отрицательная связь")
else:
    print(f"     → Положительная связь (неожиданно!)")

print(f"\n   • Ставка → Доступность: ρ = {rate_afford_corr:.4f}")
if rate_afford_corr < -0.3:
    print(f"     → Сильная ОТРИЦАТЕЛЬНАЯ связь: рост ставки → снижение доступности")
elif rate_afford_corr < 0:
    print(f"     → Слабая отрицательная связь")
else:
    print(f"     → Положительная связь")

print(f"\n Панельная регрессия (с учетом региональных эффектов):")
try:
    print(f"   • β(ставка → кредиты):     {beta_rate_loans:.4f}  {'***' if pval_rate_loans < 0.01 else '**' if pval_rate_loans < 0.05 else '*' if pval_rate_loans < 0.1 else ''}")
    print(f"   • β(ставка → доступность): {beta_rate_afford:.4f}  {'***' if pval_rate_afford < 0.01 else '**' if pval_rate_afford < 0.05 else '*' if pval_rate_afford < 0.1 else ''}")
except:
    pass

print(f"\n" + "="*100 + "\n")
