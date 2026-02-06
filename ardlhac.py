import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

CSV_PATH = r"G:\merged_data.csv" # замените на путь к вашему файлу
df = pd.read_csv(CSV_PATH)
df["date"] = df["date"].astype(str).str[:7]
df = df[(df["date"] >= "2021-01") & (df["date"] <= "2025-10")].copy()
df = df.sort_values("date").reset_index(drop=True)

print(f"\n{'═'*70}")
print(f" ДАННЫЕ")
print(f"{'═'*70}")
print(f"  Период: {df['date'].min()} — {df['date'].max()}")
print(f"  Наблюдений: {len(df)} месяцев")

def col(name):
    """Безопасно берём столбец как float."""
    return pd.to_numeric(df[name], errors="coerce")

REGIONS = [
    "Москва", "Санкт-Петербург", "Московская область",
    "Краснодарский край", "Республика Татарстан", "Свердловская область",
    "Новосибирская область", "Нижегородская область", "Республика Башкортостан",
    "Ростовская область", "Приморский край", "Пермский край",
    "Хабаровский край", "Иркутская область", "Калининградская область",
    "Пензенская область", "Рязанская область", "Удмуртская Республика",
    "Россия",
]


oboroty_all = col("oboroty-biznesa-Все отрасли")

age_active = (
    col("potrebitelskaya-aktivnost-po-kategoriyam-tovarov-v-razreze-vozrastov-Категории товаров,25 - 34 лет") +
    col("potrebitelskaya-aktivnost-po-kategoriyam-tovarov-v-razreze-vozrastov-Категории товаров,35 - 64 лет")
) / 2

age_inactive = (
    col("potrebitelskaya-aktivnost-po-kategoriyam-tovarov-v-razreze-vozrastov-Категории товаров,15 - 24 лет") +
    col("potrebitelskaya-aktivnost-po-kategoriyam-tovarov-v-razreze-vozrastov-Категории товаров,65+ лет")
) / 2

spend_essential = (
    col("consumer-spending-Продовольственные товары") +
    col("oboroty-biznesa-Услуги ЖКХ")
) / 2

spend_discretionary = (
    col("consumer-spending-Непродовольственные товары") +
    col("consumer-spending-Общественное питание") +
    col("oboroty-biznesa-Спорт и досуг")
) / 3

key_rate = col("real-key-interest-rate-Ключевая ставка, %")

national_factors = {
    "Обороты бизнеса (все отрасли)": oboroty_all,
    "Возраст: активные (25–64)": age_active,
    "Возраст: неактивные (15–24, 65+)": age_inactive,
    "Расходы обязательные": spend_essential,
    "Расходы необязательные": spend_discretionary,
    "Ставка ЦБ": key_rate,
}

# ─── Региональные supply/deals (динамически) ───────────────────
SUPPLY_DEALS_PREFIXES = [
    ("Предложение бизнес-класс", "predlozheniya-novostroek-biznes-klassa-"),
    ("Предложение эконом/комфорт", "predlozheniya-novostroek-ekonomkomfort-klassa-"),
    ("Предложение первичный", "predlozheniya-novostroek-"),
    ("Предложение вторичный", "predlozheniya-vtorichnoi-nedvizhimosti-"),
    ("Сделки первичный", "real_estate_deals_primary_market-"),
    ("Сделки вторичный", "real_estate_deals_secondary_market-"),
]

factors_per_region = {}
regional_available = {}

for reg in REGIONS:
    reg_factors = national_factors.copy()
    has_regional = False
    
    for name, prefix in SUPPLY_DEALS_PREFIXES:
        colname = prefix + reg
        if colname in df.columns:
            reg_factors[name] = col(colname)
            has_regional = True
        else:
            # fallback на Россию, если регионального нет
            fallback_col = prefix + "Россия"
            if fallback_col in df.columns:
                reg_factors[name] = col(fallback_col)
            else:
                print(f"  ⚠ {reg}: нет данных даже по России для {name}")
    
    # Вычисляем РЕГИОНАЛЬНЫЕ доли (если базовые supply/deals доступны)
    eps = 1e-10
    if "Предложение первичный" in reg_factors and reg_factors["Предложение первичный"] is not None:
        if "Предложение бизнес-класс" in reg_factors:
            reg_factors["Доля бизнес в первичном"] = reg_factors["Предложение бизнес-класс"] / (reg_factors["Предложение первичный"] + eps)
        if "Предложение эконом/комфорт" in reg_factors:
            reg_factors["Доля эконом в первичном"] = reg_factors["Предложение эконом/комфорт"] / (reg_factors["Предложение первичный"] + eps)
        
        secondary = reg_factors.get("Предложение вторичный", 0)
        if secondary is not None:
            reg_factors["Доля первичного в общем"] = reg_factors["Предложение первичный"] / (reg_factors["Предложение первичный"] + secondary + eps)
        
        if "Сделки первичный" in reg_factors and "Сделки вторичный" in reg_factors:
            reg_factors["Отношение сделок P/S"] = reg_factors["Сделки первичный"] / (reg_factors["Сделки вторичный"] + eps)
    
    # Преобразуем в DataFrame
    factors_per_region[reg] = pd.DataFrame({"date": df["date"], **reg_factors})
    regional_available[reg] = has_regional
    
    if not has_regional:
        print(f"  ⚠ {reg}: используются только национальные факторы (нет региональных supply/deals)")

print(f"\n  Регионов с собственными supply/deals: {sum(regional_available.values())} из {len(REGIONS)}")


REGIONS = [
    "Москва", "Санкт-Петербург", "Московская область",
    "Краснодарский край", "Республика Татарстан", "Свердловская область",
    "Новосибирская область", "Нижегородская область", "Республика Башкортостан",
    "Ростовская область", "Приморский край", "Пермский край",
    "Хабаровский край", "Иркутская область", "Калининградская область",
    "Пензенская область", "Рязанская область", "Удмуртская Республика",
    "Россия"
]

SEGMENTS = [
    ("Бизнес-класс первичный", "biznes-klass"),
    ("Эконом/комфорт первичный", "ekonomkomfort"),
    ("Первичный общий", ""),
    ("Вторичный рынок", "vtorichnii-rinok"),
]

prices = pd.DataFrame({"date": df["date"]})

for reg in REGIONS:
    for seg_name, suffix in SEGMENTS:
        if suffix == "vtorichnii-rinok":
            colname = f"dinamika-tsen-obyavlenii-vtorichnii-rinok-{reg}"
        elif suffix == "":
            colname = f"dinamika-tsen-obyavlenii-pervichnii-rinok-{reg}"
        else:
            colname = f"dinamika-tsen-obyavlenii-pervichnii-rinok-{suffix}-{reg}"
        
        if colname in df.columns:
            prices[f"{reg} — {seg_name}"] = col(colname)
            print(f"  Добавлен: {reg} — {seg_name} ({colname})")  # для дебага
        else:
            print(f"  Нет данных для {reg} — {seg_name} ({colname})")

AVAILABLE_REGIONS = []
for col in prices.columns:
    if col != "date" and " — " in col:
        reg_part = col.split(" — ")[0]
        if reg_part not in AVAILABLE_REGIONS:
            AVAILABLE_REGIONS.append(reg_part)

print(f"\n  Доступных регионов с хотя бы одним сегментом цен: {len(AVAILABLE_REGIONS)}")
print("  Список:", ", ".join(AVAILABLE_REGIONS))

AVAILABLE_PRICES = [c for c in prices.columns if c != "date"]
print(f"  Доступных комбинаций регион-сегмент: {len(AVAILABLE_PRICES)}")


# ИМПОРТ
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Настройки
VIF_THRESHOLD   = 10
MIN_OBS         = 12
HAC_LAGS        = 3
ALPHA           = 0.05
RIDGE_THRESHOLD = 0.45       # если adj-R² МНК меньщ этого, то Ridge

# ОПРЕДЕЛЕНИЕ ГРУПП ФАКТОРОВ
# Национальные — всегда входят в обе стратегии
NATIONAL = [
    "Обороты бизнеса (все отрасли)",
    "Возраст: активные (25–64)",
    "Возраст: неактивные (15–24, 65+)",
    "Расходы обязательные",
    "Расходы необязательные",
    "Ставка ЦБ",
]

# абс
RAW_COLS = [
    "Предложение бизнес-класс",
    "Предложение эконом/комфорт",
    "Предложение первичный",
    "Предложение вторичный",
    "Сделки первичный",
    "Сделки вторичный",
]

# доли
RATIO_COLS = [
    "Доля бизнес в первичном",
    "Доля эконом в первичном",
    "Доля первичного в общем",
    "Отношение сделок P/S",
]


def _build_strategies(feat_df: pd.DataFrame) -> dict:
    # квадратичный член ставки
    if "Ставка ЦБ" in feat_df.columns:
        feat_df = feat_df.copy()
        feat_df["Ставка ЦБ²"] = feat_df["Ставка ЦБ"] ** 2

    # абс
    raw_cols  = [c for c in NATIONAL + RAW_COLS + ["Ставка ЦБ²"] if c in feat_df.columns]
    # доли
    ratio_cols = [c for c in NATIONAL + RATIO_COLS + ["Ставка ЦБ²"] if c in feat_df.columns]

    return {
        "RAW":   feat_df[raw_cols],
        "RATIO": feat_df[ratio_cols],
    }


# VIF 
def _prune_vif(X: pd.DataFrame, threshold: float = VIF_THRESHOLD):
    dropped = {}
    X_work = X.copy()
    while X_work.shape[1] >= 2:
        vifs = pd.Series(
            [variance_inflation_factor(X_work.values, i) for i in range(X_work.shape[1])],
            index=X_work.columns
        )
        if vifs.max() < threshold:
            break
        worst = vifs.idxmax()
        dropped[worst] = round(float(vifs.max()), 2)
        X_work = X_work.drop(columns=[worst])
    return X_work, dropped


# OLS + HAC
def _fit_ols_hac(y, X, lags=HAC_LAGS):
    X_c = add_constant(X)
    return OLS(y, X_c).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


# RIDGE
def _fit_ridge(y, X):
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    alphas = np.logspace(-3, 3, 50)
    ridge = RidgeCV(alphas=alphas, cv=5).fit(X_s, y)

    y_pred = ridge.predict(X_s)
    ss_res  = np.sum((y.values - y_pred) ** 2)
    ss_tot  = np.sum((y.values - y.mean()) ** 2)
    r2      = 1 - ss_res / ss_tot
    n, k    = X.shape
    adj_r2  = 1 - (1 - r2) * (n - 1) / (n - k - 1)

    # Коэффициенты в ориг масштабе (для интерпретации)
    coefs_scaled = ridge.coef_
    coefs_orig   = coefs_scaled / scaler.scale_
    coef_series  = pd.Series(coefs_orig, index=X.columns)

    return y_pred, coef_series, ridge.alpha_, r2, adj_r2


# Текст от сонет
FACTOR_DESCR = {
    "const":                          "Базовый уровень цены (свободный член)",
    "Обороты бизнеса (все отрасли)":  "Общая деловая активность в регионе",
    "Возраст: активные (25–64)":      "Покупательная сила группы 25–64 лет",
    "Возраст: неактивные (15–24, 65+)":"Покупательная сила молодёжи и пожилых",
    "Расходы обязательные":           "Расходы на товары первой необходимости и ЖКХ",
    "Расходы необязательные":         "Дискреционные расходы (досуг, питание вне дома)",
    "Ставка ЦБ":                      "Ключевая ставка Банка России (линейный эффект)",
    "Ставка ЦБ²":                     "Ключевая ставка в квадрате (нелинейный эффект)",
    "Предложение бизнес-класс":       "Объём предложения бизнес-класс новостроек",
    "Предложение эконом/комфорт":     "Объём предложения эконом/комфорт новостроек",
    "Предложение первичный":          "Общий объём предложения на первичном рынке",
    "Предложение вторичный":          "Объём предложения на вторичном рынке",
    "Сделки первичный":               "Число сделок на первичном рынке",
    "Сделки вторичный":               "Число сделок на вторичном рынке",
    "Доля бизнес в первичном":        "Доля бизнес-класса в первичном предложении",
    "Доля эконом в первичном":        "Доля эконом/комфорт в первичном предложении",
    "Доля первичного в общем":        "Доля первичного рынка в общем объёме",
    "Отношение сделок P/S":           "Соотношение сделок первичный / вторичный",
    "lag_price":                      "Цена предыдущего месяца (лаг-1, ARDL)",
}


def _interpret_coef(name, coef, pval):
    sig   = "значимый" if pval < ALPHA else "НЕ значимый"
    direc = "положительное" if coef > 0 else "отрицательное"
    descr = FACTOR_DESCR.get(name, name)
    return f"  • {descr}: {direc} влияние (β = {coef:+.4f}), {sig} (p = {pval:.4f})."


# ОТЧЁТ
def _print_report(reg, seg, model, X_clean, dropped_vif, dw, shap_stat, shap_p,
                  strategy_name, idx, total, ridge_info=None):
    use_ridge = ridge_info is not None and ridge_info.get("used", False)

    print(f"\n{'─'*82}")
    print(f"  МОДЕЛЬ [{idx}/{total}]  ▸  {reg}  —  {seg}  "
          f"[стратегия: {strategy_name}]"
          f"{'  ⚡ Ridge' if use_ridge else ''}")
    print(f"{'─'*82}")

    # удалённые VIF
    if dropped_vif:
        print(f"\n  Удалены по VIF ≥ {VIF_THRESHOLD}:")
        for name, val in dropped_vif.items():
            print(f"      • {name:48s} VIF = {val}")
        print(f"     Эти переменные коллинеарны и убраны.")
    else:
        print(f"\n  Все VIF < {VIF_THRESHOLD}, мультиколлинеарность отсутствует.")

    # коэфы
    if not use_ridge:
        params  = model.params
        bse     = model.bse
        tvalues = model.tvalues
        pvalues = model.pvalues
        r2      = model.rsquared
        adj_r2  = model.rsquared_adj
        fval    = model.fvalue
        fp      = model.f_pvalue
    else:
        params  = ridge_info["coefs"]
        bse     = pd.Series(np.nan, index=params.index)
        tvalues = pd.Series(np.nan, index=params.index)
        pvalues = pd.Series(np.nan, index=params.index)
        r2      = ridge_info["r2"]
        adj_r2  = ridge_info["adj_r2"]
        fval    = np.nan
        fp      = np.nan

    print(f"\n  Коэффициенты (SE — Newey–West HAC, alpha = {ALPHA}):"
          if not use_ridge else
          f"\n  Коэффициенты (Ridge, α_reg = {ridge_info['alpha']:.4f}):")
    print(f"  {'Фактор':<50} {'β':>10} {'SE(HAC)':>10} {'t':>8} {'p':>8} {'sig':>5}")
    print(f"  {'─'*50} {'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*5}")

    for name in params.index:
        if use_ridge:
            print(f"  {name:<50} {params[name]:>+10.4f} {'—':>10} {'—':>8} {'—':>8} {'—':>5}")
        else:
            p = pvalues[name]
            stars = "***" if p < 0.01 else (" **" if p < 0.05 else ("  *" if p < 0.10 else "   "))
            print(f"  {name:<50} {params[name]:>+10.4f} {bse[name]:>10.4f} "
                  f"{tvalues[name]:>8.3f} {pvalues[name]:>8.4f} {stars:>5}")

    print(f"  {'─'*92}")
    if not use_ridge:
        print(f"  Легенда: *** p<0.01  ** p<0.05  * p<0.10")

    # качество моделей
    print(f"\n  Качество модели:")
    print(f"      R²      = {r2:.4f}   (объясняет {r2*100:.1f}% дисперсии цен)")
    print(f"      adj-R²  = {adj_r2:.4f}")
    if not use_ridge:
        sig_f = "модель значима " if fp < 0.05 else "модель НЕ значима"
        print(f"      F       = {fval:.2f}   (p = {fp:.4f}) — {sig_f}")

    # DW
    if dw < 1.5:
        dw_c = "автокорреляция остатков есть (HAC компенсирует SE)"
    elif dw > 2.5:
        dw_c = "отрицательная автокорреляция"
    else:
        dw_c = "автокорреляция минимальна"
    print(f"\n  Диагностика:")
    print(f"      DW = {dw:.3f}  →  {dw_c}")
    if not np.isnan(shap_p):
        norm_c = "нормальность не отвергается" if shap_p > 0.05 else "остатки не нормальны"
        print(f"      Шапиро: W = {shap_stat:.4f}, p = {shap_p:.4f}  →  {norm_c}")

    # интерпретация
    print(f"\n  Интерпретация:")
    if not use_ridge:
        for name in params.index:
            print(_interpret_coef(name, params[name], pvalues[name]))
    else:
        for name in params.index:
            direc = "положительное" if params[name] > 0 else "отрицательное"
            descr = FACTOR_DESCR.get(name, name)
            print(f"  • {descr}: {direc} влияние (β = {params[name]:+.4f}).")

    # итог
    method = "Ridge" if use_ridge else "OLS+HAC"
    print(f"\n  Итог [{method}]: модель объясняет {adj_r2*100:.1f}% дисперсии цен "
          f"(adj-R²) в «{reg}», сегмент «{seg}».")


# графики
def _plot_model(reg, seg, y, y_pred_lag, y_pred_no_lag, residuals,
                r2_lag, r2_no_lag, method_label):

    n = len(y)
    t = np.arange(n)

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"{reg}  —  {seg}  [{method_label}]",
                 fontsize=16, fontweight="bold")
    fig.subplots_adjust(top=0.92, hspace=0.38, wspace=0.28)

    gs = gridspec.GridSpec(3, 2, figure=fig)

    # helper: линейный + scatter для одной версии модели
    def _draw_pair(ax_line, ax_scat, y_true, y_hat, r2_val,
                   title_suffix, color_line, color_dot):
        # линейный график
        ax_line.plot(t, y_true.values, color="#2196F3", lw=1.8, label="Факт")
        ax_line.plot(t, y_hat, color=color_line, lw=1.8, ls="--",
                     label=f"Модель  R²={r2_val:.3f}")
        ax_line.set_title(f"Факт vs Предсказание — {title_suffix}",
                          fontsize=11, fontweight="bold")
        ax_line.set_xlabel("Месяц (индекс)")
        ax_line.set_ylabel("Цена")
        ax_line.legend(fontsize=9, loc="best")
        ax_line.grid(alpha=0.25)

        # scatter
        ax_scat.scatter(y_hat, y_true.values, color=color_dot, s=22, alpha=0.75)
        lo = min(y_true.min(), y_hat.min())
        hi = max(y_true.max(), y_hat.max())
        ax_scat.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = ŷ")
        ax_scat.set_title(f"Scatter — {title_suffix}  R²={r2_val:.3f}",
                          fontsize=11, fontweight="bold")
        ax_scat.set_xlabel("Предсказание")
        ax_scat.set_ylabel("Факт")
        ax_scat.legend(fontsize=9)
        ax_scat.grid(alpha=0.25)

    # левый столбец: С лагом
    _draw_pair(
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[1, 0]),
        y, y_pred_lag, r2_lag,
        title_suffix="с лагом (ARDL)",
        color_line="#FF5722",
        color_dot="#673AB7",
    )

    # правый столбец: БЕЗ лага
    _draw_pair(
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 1]),
        y, y_pred_no_lag, r2_no_lag,
        title_suffix="без лага (чистые факторы)",
        color_line="#4CAF50",
        color_dot="#FF9800",
    )

    # row 2 left: остатки основной модели
    ax5 = fig.add_subplot(gs[2, 0])
    bar_colors = ["#EF5350" if r > 0 else "#42A5F5" for r in residuals]
    ax5.bar(t, residuals, color=bar_colors, alpha=0.75, width=0.8)
    ax5.axhline(0, color="black", lw=0.9)
    ax5.set_title("Остатки во времени (основная модель)",
                  fontsize=11, fontweight="bold")
    ax5.set_xlabel("Месяц (индекс)")
    ax5.set_ylabel("Остаток")
    ax5.grid(alpha=0.25)

    # row 2 right: QQ-plot
    ax6 = fig.add_subplot(gs[2, 1])
    sm.qqplot(pd.Series(residuals), line="s", ax=ax6, alpha=0.7)
    ax6.set_title("QQ-plot остатков (основная модель)",
                  fontsize=11, fontweight="bold")
    ax6.grid(alpha=0.25)

    plt.savefig(f"reg_{reg}_{seg}.png".replace(" ", "_").replace("/", "_"),
                dpi=150, bbox_inches="tight")
    plt.show()


# Мейн
def run_all_regressions(factors_per_region: dict,
                        prices: pd.DataFrame,
                        available_regions: list):
    
    SEGMENT_NAMES = [
        "Бизнес-класс первичный",
        "Эконом/комфорт первичный",
        "Первичный общий",
        "Вторичный рынок",
    ]

    # ── считаем общее число моделей для прогресса ──
    total = 0
    for reg in sorted(available_regions):
        if reg not in factors_per_region:
            continue
        for seg in SEGMENT_NAMES:
            if f"{reg} — {seg}" in prices.columns:
                total += 1

    print(f"\n{'═'*82}")
    print(f"  ЛИНЕЙНАЯ РЕГРЕССИЯ v2  ▸  моделей: {total}")
    print(f"  Стратегии факторов: RAW (объёмы) и RATIO (доли)")
    print(f"  Дополнения: lag(1) цены (ARDL), Ставка ЦБ² (нелинейность)")
    print(f"  Fallback: Ridge CV если OLS adj-R² < {RIDGE_THRESHOLD}")
    print(f"{'═'*82}\n")

    results_log = []
    idx = 0

    for reg in sorted(available_regions):
        if reg not in factors_per_region:
            continue

        feat_df = factors_per_region[reg].drop(columns=["date"], errors="ignore")

        for seg in SEGMENT_NAMES:
            price_col = f"{reg} — {seg}"
            if price_col not in prices.columns:
                continue
            idx += 1

            # маска: убираем NaN
            y_raw = prices[price_col]
            mask  = y_raw.notna() & feat_df.notna().all(axis=1)
            # сдвигаем на 1 для лага теряем первую строку
            mask  = mask & mask.shift(1).fillna(False).astype(bool)

            y     = y_raw[mask].reset_index(drop=True).astype(float)
            feat  = feat_df[mask].reset_index(drop=True).astype(float)

            if len(y) < MIN_OBS:
                print(f"  [{idx}/{total}] {reg} — {seg}: мало наблюдений ({len(y)}), пропуск.")
                continue

            # лаг(1) цены
            lag_price = y_raw[mask].shift(1).reset_index(drop=True).astype(float)
            # после shift первая строка NaN убирать
            valid     = lag_price.notna()
            y         = y[valid].reset_index(drop=True)
            feat      = feat[valid].reset_index(drop=True)
            lag_price = lag_price[valid].reset_index(drop=True)

            if len(y) < MIN_OBS:
                print(f"  ⚠ [{idx}/{total}] {reg} — {seg}: после лага мало наблюдений, пропуск.")
                continue

            # две страты
            strategies = _build_strategies(feat)

            best_model       = None
            best_adj_r2      = -np.inf
            best_strategy    = ""
            best_X_clean     = None
            best_dropped     = {}
            best_ridge_info  = None

            for strat_name, X_strat in strategies.items():
                # lag добавляем ПОСЛЕ VIF. VIF считается по X, без lag_price
                X_clean, dropped = _prune_vif(X_strat)

                if X_clean.shape[1] == 0:
                    continue

                # lag back
                X_clean = X_clean.copy()
                X_clean["lag_price"] = lag_price.values

                # OLS + HAC
                model = _fit_ols_hac(y, X_clean)

                if model.rsquared_adj > best_adj_r2:
                    best_adj_r2     = model.rsquared_adj
                    best_model      = model
                    best_strategy   = strat_name
                    best_X_clean    = X_clean
                    best_dropped    = dropped
                    best_ridge_info = None      # сбрасываем Ridge

            if best_model is None:
                print(f" [{idx}/{total}] {reg} — {seg}: обе стратегии пустые после VIF, скип.")
                continue

            # Ridge без VIF
            if best_adj_r2 < RIDGE_THRESHOLD:
                # восстанавливаем полный X лучшей страты + lag
                X_ridge = strategies[best_strategy].copy()
                X_ridge["lag_price"] = lag_price.values

                y_pred_r, coefs_r, alpha_r, r2_r, adj_r2_r = _fit_ridge(y, X_ridge)
                print(f"  [{idx}/{total}] {reg} — {seg}: Ridge vs OLS "
                      f"adj-R²: {adj_r2_r:.4f} vs {best_adj_r2:.4f} "
                      f"(α = {alpha_r:.4f}, факторов = {X_ridge.shape[1]})")
                if adj_r2_r > best_adj_r2:
                    best_ridge_info = {
                        "used":    True,
                        "alpha":   alpha_r,
                        "r2":      r2_r,
                        "adj_r2":  adj_r2_r,
                        "coefs":   coefs_r,
                        "y_pred":  y_pred_r,
                    }
                    print(f"  Ridge выбран (лучше OLS на {adj_r2_r - best_adj_r2:+.4f})")

            # остатки
            if best_ridge_info and best_ridge_info["used"]:
                residuals = y.values - best_ridge_info["y_pred"]
                y_pred    = best_ridge_info["y_pred"]
                method_label = f"Ridge α={best_ridge_info['alpha']:.3f}"
            else:
                residuals = best_model.resid.values
                y_pred    = best_model.fittedvalues.values
                method_label = "OLS+HAC"

            dw = durbin_watson(residuals)
            if len(residuals) <= 5000:
                shap_stat, shap_p = shapiro(residuals)
            else:
                shap_stat, shap_p = np.nan, np.nan

            # принт
            _print_report(
                reg, seg, best_model, best_X_clean, best_dropped,
                dw, shap_stat, shap_p, best_strategy, idx, total,
                ridge_info=best_ridge_info
            )

            # модель БЕЗ лага (для графика)
            X_no_lag = best_X_clean.drop(columns=["lag_price"], errors="ignore")
            if X_no_lag.shape[1] > 0:
                model_no_lag  = _fit_ols_hac(y, X_no_lag)
                y_pred_no_lag = model_no_lag.fittedvalues.values
                r2_no_lag     = model_no_lag.rsquared
            else:
                # если после убрать лаг ничего не осталось — просто среднее
                y_pred_no_lag = np.full(len(y), y.mean())
                r2_no_lag     = 0.0

            # R² основной модели (с лагом)
            r2_lag = (best_ridge_info["r2"]
                      if best_ridge_info and best_ridge_info["used"]
                      else best_model.rsquared)

            # график
            _plot_model(reg, seg, y,
                        y_pred,          # с лагом
                        y_pred_no_lag,   # без лага
                        residuals,
                        r2_lag, r2_no_lag,
                        method_label)

            # лог
            final_adj_r2 = (best_ridge_info["adj_r2"]
                            if best_ridge_info and best_ridge_info["used"]
                            else best_adj_r2)
            results_log.append({
                "Регион":        reg,
                "Сегмент":       seg,
                "Стратегия":     best_strategy,
                "Метод":         method_label,
                "N":             len(y),
                "Факторов":      best_X_clean.shape[1],
                "Удалено (VIF)": len(best_dropped),
                "R² (с лагом)":  round(r2_lag, 4),
                "R² (без лага)": round(r2_no_lag, 4),
                "adj-R²":        round(final_adj_r2, 4),
                "DW":            round(dw, 3),
                "Шапиро p":      round(shap_p, 4) if not np.isnan(shap_p) else "—",
            })

    # итог
    _print_summary(results_log)
    return results_log


# Результаты
def _print_summary(results_log):
    if not results_log:
        print("\n  Нет результатов.")
        return

    df = pd.DataFrame(results_log).sort_values("adj-R²", ascending=False).reset_index(drop=True)

    # вклад лага дельта R²
    df["Δ R² (лаг)"] = df["R² (с лагом)"] - df["R² (без лага)"]

    print(f"\n\n{'═'*120}")
    print(f"  ИТОГОВАЯ СВОДКА (adj-R² ↓)")
    print(f"{'═'*120}")
    print(df.drop(columns=["_dw_dist"], errors="ignore").to_string(index=False))

    print(f"\n{'─'*120}")
    print(f"  ТОП-5 по adj-R²:")
    print(f"{'─'*120}")
    for i, row in df.head(5).iterrows():
        print(f"    {i+1}. {row['Регион']:32s} | {row['Сегмент']:30s} | "
              f"adj-R² = {row['adj-R²']:.4f} | {row['Метод']:20s} | факторов = {row['Факторов']}")

    print(f"\n{'─'*120}")
    print(f"  ТОП-5 по DW (ближе всего к 2):")
    print(f"{'─'*120}")
    df["_dw_dist"] = abs(df["DW"] - 2.0)
    for i, (_, row) in enumerate(df.nsmallest(5, "_dw_dist").iterrows(), 1):
        print(f"    {i}. {row['Регион']:32s} | {row['Сегмент']:30s} | "
              f"DW = {row['DW']:.3f} | adj-R² = {row['adj-R²']:.4f}")

    # общие выводы
    print(f"\n{'═'*120}")
    print(f"  ОБЩИЕ ВЫВОДЫ")
    print(f"{'═'*120}")

    mean_adj = df["adj-R²"].mean()
    print(f"\n  • Среднее adj-R²: {mean_adj:.4f} "
          f"({'удовлетворительное' if mean_adj > 0.5 else 'низкое'})")

    print(f"\n  • По сегментам:")
    for seg, val in df.groupby("Сегмент")["adj-R²"].mean().sort_values(ascending=False).items():
        print(f"      {seg:42s}  {val:.4f}")

    print(f"\n  • По регионам (топ-5):")
    for reg, val in df.groupby("Регион")["adj-R²"].mean().sort_values(ascending=False).head(5).items():
        print(f"      {reg:42s}  {val:.4f}")

    print(f"\n  • Стратегия факторов:")
    for strat, cnt in df["Стратегия"].value_counts().items():
        print(f"      {strat:10s}: выбрана в {cnt} моделях")

    print(f"\n  • Метод:")
    for m, cnt in df["Метод"].value_counts().items():
        print(f"      {m:25s}: {cnt} моделей")

    dw_ok = ((df["DW"] > 1.5) & (df["DW"] < 2.5)).sum()
    print(f"\n  • DW в норме (1.5–2.5): {dw_ok} из {len(df)} моделей")
    print(f"    (добавление lag(1) цены существенно улучшает DW vs v1)")

    print(f"\n  • Среднее факторов после VIF: {df['Факторов'].mean():.1f} "
          f"(макс {df['Факторов'].max()}, мин {df['Факторов'].min()})")

    # ── вклад лага ──
    print(f"\n{'─'*120}")
    print(f"  ВКЛАД ЛАГА (насколько lag(1) цены улучшает R² сверх чистых факторов X)")
    print(f"{'─'*120}")
    mean_delta = df["Δ R² (лаг)"].mean()
    print(f"  • Среднее Δ R²: +{mean_delta:.4f}")
    print(f"  • Макс  Δ R²:   +{df['Δ R² (лаг)'].max():.4f}  "
          f"({df.loc[df['Δ R² (лаг)'].idxmax(), 'Регион']} — "
          f"{df.loc[df['Δ R² (лаг)'].idxmax(), 'Сегмент']})")
    print(f"  • Мин   Δ R²:   +{df['Δ R² (лаг)'].min():.4f}  "
          f"({df.loc[df['Δ R² (лаг)'].idxmin(), 'Регион']} — "
          f"{df.loc[df['Δ R² (лаг)'].idxmin(), 'Сегмент']})")
    print(f"\n  Интерпретация: высокий Δ означает сильную инерцию цен в регионе —")
    print(f"  цена прошлого месяца сама по себе хорошо предсказывает текущую.")
    print(f"  Низкий Δ означает, что факторы X (ставка, предложение и т.д.)")
    print(f"  объясняют цену почти без помощи лага.")

results = run_all_regressions(factors_per_region, prices, AVAILABLE_REGIONS)
