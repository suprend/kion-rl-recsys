"""Publication figures from compact aggregate JSON only; lazy Matplotlib import.

Writes SVG and PNG together. No individual user IDs, numeric arrays or models read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import textwrap

ARMS = ('C', 'RL-completion', 'RL-residual')
LABELS = ('C · исходная', 'RL · completion', 'RL · residual')
COLORS = ('#52677F', '#007F83', '#CB6A3B')
PAIR_LABELS = {'RL-completion-minus-C': 'completion − C', 'RL-residual-minus-C': 'residual − C',
               'RL-residual-minus-RL-completion': 'residual − completion'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def validate(data):
    require(data.get('contract') == 'kion-scientific3-analysis-v1' and data.get('state') == 'completed',
            'Completed aggregate analysis required')
    require(data.get('stage') in ('screen', 'full4096'), 'Unknown cohort')
    require(data['rows'] == (256 if data['stage'] == 'screen' else 4096) and
            data['metric_denominator'] == data['rows'] and data['candidate_width'] == 1024, 'Cohort differs')
    require(set(ARMS) <= set(data['summaries']), 'All scientific arms required')
    require((data['screen_diagnostics'] is not None) == (data['stage'] == 'screen'), 'Proxy cohort differs')
    return data


def cohort_caption(data):
    cohort = 'reused development screen256' if data['stage'] == 'screen' else 'reused development4096'
    return (f"{cohort}; {data['rows']} запросов, {data['unique_users']} пользователей; natural1024. "
            'C и latest1000 двух RL; без seen-filter и target injection; unknown/off-pool включены в знаменатель. '
            'Фиксированный β, не matched-KL; исследовательский результат, не причинный эффект.')


def _style(plt):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.titlesize': 12, 'axes.labelsize': 10, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.titleweight': 'semibold', 'axes.labelcolor': '#253445', 'text.color': '#253445',
        'xtick.color': '#526174', 'ytick.color': '#526174', 'grid.color': '#DDE4EB',
        'grid.linewidth': .65, 'figure.facecolor': '#FFFFFF', 'axes.facecolor': '#FFFFFF',
        'savefig.facecolor': '#FFFFFF', 'svg.fonttype': 'none'})


def _save(plt, fig, output, name, title, caption):
    fig.suptitle(title, x=.055, y=.985, ha='left', fontsize=17, weight='semibold')
    lines = textwrap.wrap(caption, width=145)
    fig.text(.055, .022, '\n'.join(lines), va='bottom', ha='left', fontsize=8, color='#526174')
    fig.tight_layout(rect=(.035, .075 + .015 * len(lines), .985, .935), w_pad=2.5, h_pad=3.2)
    for extension in ('svg', 'png'):
        fig.savefig(output / (name + '.' + extension), dpi=190, bbox_inches='tight')
    plt.close(fig)
    return {'file': name, 'title': title, 'caption': caption, 'formats': ['svg', 'png']}


def _bars(ax, values, title, percent=False, decimals=3):
    multiplier = 100 if percent else 1
    shown = [value * multiplier for value in values]
    ax.bar(range(3), shown, color=COLORS, width=.64)
    ax.set_xticks(range(3), LABELS, fontsize=9)
    ax.set_title(title, loc='left', pad=13)
    ax.grid(axis='y', zorder=0)
    ax.set_axisbelow(True)
    low, high = min(0., min(shown)), max(0., max(shown))
    span = max(high - low, 1e-5)
    ax.set_ylim(low - .12 * span, high + .25 * span)
    for j, value in enumerate(shown):
        text = (f'{value:.{decimals}f}' + ('%' if percent else ''))
        ax.annotate(text, (j, value), xytext=(0, 5 if value >= 0 else -13), textcoords='offset points',
                    ha='center', fontsize=10, weight='semibold')
    if low < 0:
        ax.axhline(0, color='#526174', linewidth=.8)


def render(data, output):
    data = validate(data)
    output = Path(output).resolve()
    require(not output.exists() or not any(output.iterdir()), 'Fresh/empty figure directory required')
    output.mkdir(parents=True, exist_ok=True)
    # The numerical/plotting import occurs only after explicit rendering.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _style(plt)
    figures = []
    caption = cohort_caption(data)
    curve = data.get('checkpoint_curve')
    if curve:
        require([point['step'] for point in curve] == [0, 250, 500, 1000], 'Complete screen curve required')
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.5))
        ax = axes[0]
        ax.plot([point['expected_effective_reward'] for point in curve], [point['map10'] for point in curve],
                color='#BAC8D5', linewidth=1.7, zorder=1)
        for point in curve:
            ax.scatter(point['expected_effective_reward'], point['map10'], s=80,
                       color='#007F83' if point['feasible'] else '#CB6A3B', zorder=3)
            ax.annotate(str(point['step']), (point['expected_effective_reward'], point['map10']),
                        xytext=(7, 6), textcoords='offset points', fontsize=10)
        ax.axhline(curve[0]['map10'] - .002, color='#52677F', linestyle='--', linewidth=1,
                   label='Порог MAP: C − 0,002')
        ax.set(title='Proxy растёт; точность и gates ограничивают выбор', xlabel='C-centered completion proxy', ylabel='MAP@10')
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=.8)
        ax = axes[1]
        for key, limit, baseline, color, label in (
            ('exact_KL', .05, 0., COLORS[0], 'KL / 0,05'),
            ('top1_max_share', .05, curve[0]['top1_max_share'], COLORS[2], 'Δ top1 share / 0,05'),
            ('unsupported_mass', .01, curve[0]['unsupported_mass'], COLORS[1], 'Δ unsupported / 0,01')):
            ax.plot([point['step'] for point in curve], [(point[key] - baseline) / limit for point in curve],
                    'o-', color=color, label=label, markersize=5)
        ax.axhline(1, color='#9F4444', linestyle='--', linewidth=1)
        ax.set(title='Доля допустимого ограничения (≤1)', xlabel='Фактический update', ylabel='Значение / предел')
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=.8)
        figures.append(_save(plt, fig, output, '01-checkpoint-tradeoff', 'Почему selected0 сохранил C',
            'Legacy completion screen256: 256 одинаковых запросов, natural1024, checkpoints 0/250/500/1000; '
            'число пользователей в исходных aggregates не сохранено. '
            'Это использованный при выборе screen; CI здесь не восстановлены. Зелёный = все gates пройдены; '
            'оранжевый = нарушено хотя бы одно. Selected0 не отменяет обучение latest1000.'))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.3))
    for j, (metric, title) in enumerate((('map10', 'MAP@10'), ('ndcg10', 'NDCG@10'), ('recall10', 'Recall@10'))):
        _bars(axes[0, j], [data['summaries'][arm]['metrics'][metric] for arm in ARMS], title, decimals=4)
    for j, (metric, title, pct, decimals) in enumerate((
        ('top1_max_item_share', 'Максимальная доля одного фильма в top1', True, 1),
        ('top10_hhi', 'HHI рекомендованного top10', False, 4),
        ('effective_catalog_items', 'Эффективный каталог · 1 / HHI', False, 1))):
        _bars(axes[1, j], [data['summaries'][arm]['exposure']['query_weighted'][metric] for arm in ARMS], title, pct, decimals)
    figures.append(_save(plt, fig, output, '02-matched-comparison', 'Три политики на одних и тех же запросах', caption))
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.7))
    for j, metric in enumerate(('map10', 'ndcg10', 'recall10')):
        ax = axes[j]
        for y, (name, comparison) in enumerate(data['comparisons'].items()):
            value = comparison['metrics'][metric]
            low, high = value['user_cluster_95ci']
            ax.plot([low, high], [y, y], color=COLORS[y], linewidth=2.8)
            ax.scatter([value['difference']], [y], color=COLORS[y], s=45, zorder=3)
        ax.axvline(0, color='#526174', linewidth=.85, linestyle='--')
        ax.set_yticks(range(3), [PAIR_LABELS[name] for name in data['comparisons']], fontsize=9)
        ax.set_title(metric.upper().replace('10', '@10'), loc='left')
        ax.set_xlabel('Парная разность; 95% user-cluster CI')
        ax.set_ylim(2.6, -.6)
        ax.grid(axis='x')
    figures.append(_save(plt, fig, output, '03-paired-uncertainty', 'Насколько устойчивы парные различия', caption +
        ' Bootstrap: 2000 общих resamples пользователей, seed20261024; query-weighted ratio; '
        'процентильные CI без поправки на множественные сравнения. Интервалы условны на единственные fitted checkpoints.'))
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
    for row, weighting in enumerate(('query_weighted', 'equal_user_weighted')):
        ax = axes[row, 0]
        exposures = {arm: data['summaries'][arm]['exposure'][weighting]['sorted_top10_exposure'] for arm in ARMS}
        for arm, label, color in zip(ARMS, LABELS, COLORS):
            shares, cumulative = exposures[arm], []
            total = 0.
            for item in shares:
                total += item['share']
                cumulative.append(total * 100)
            ax.plot(range(1, len(shares) + 1), cumulative, color=color, label=label, linewidth=1.8)
        ax.set(xlabel='Ранг фильма по exposure внутри каждой политики', ylabel='Накопленная доля top10, %', ylim=(0, 103))
        ax.set_title('Вес запросов' if row == 0 else 'Равный вес пользователей', loc='left')
        ax.legend(frameon=False, fontsize=9)
        ax.grid()
        mappings = {arm: {item['raw_id']: item['share'] for item in exposures[arm]} for arm in ARMS}
        ids = sorted(set().union(*(mapping for mapping in mappings.values())),
                     key=lambda raw_id: (-max(mapping.get(raw_id, 0.) for mapping in mappings.values()), raw_id))[:12]
        ax = axes[row, 1]
        for j, arm in enumerate(ARMS):
            ax.barh([y + (j - 1) * .23 for y in range(len(ids))],
                    [mappings[arm].get(raw_id, 0.) * 100 for raw_id in ids], height=.21, color=COLORS[j])
        ax.set_yticks(range(len(ids)), [str(raw_id) for raw_id in ids], fontsize=8)
        ax.invert_yaxis()
        ax.set(xlabel='Доля всех рекомендованных слотов top10, %', ylabel='raw item ID')
        ax.set_title('12 лидеров по максимальной доле среди arms', loc='left')
        ax.grid(axis='x')
    figures.append(_save(plt, fig, output, '04-exposure-distribution', 'Как распределяется каталог рекомендаций',
        caption + ' Кривые слева сортируют фильмы отдельно в каждой политике; bars справа сопоставляют одинаковые raw IDs. '
        'Равный вес пользователей означает вес каждого его запроса 1 / число запросов пользователя.'))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2))
    bins = list(data['summaries']['C']['rank_bins'])
    for j, arm in enumerate(ARMS):
        axes[0].bar([x + (j - 1) * .25 for x in range(len(bins))],
                    [data['summaries'][arm]['rank_bins'][label] for label in bins], width=.23, color=COLORS[j], label=LABELS[j])
    axes[0].set_xticks(range(len(bins)), bins, rotation=32, ha='right')
    axes[0].set(title='Ранг единственного observed target во всём pool', ylabel='Число запросов')
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(axis='y')
    ax = axes[1]
    for j, (name, comparison) in enumerate(data['comparisons'].items()):
        metric = comparison['metrics']['map10']
        start = 0
        for key, color in (('losses', '#CB6A3B'), ('ties', '#D9E2EA'), ('gains', '#007F83')):
            ax.barh(j, metric[key], left=start, color=color, height=.58, label=key if j == 0 else None)
            if metric[key] > .055 * data['rows']:
                ax.text(start + metric[key] / 2, j, str(metric[key]), ha='center', va='center', fontsize=9)
            start += metric[key]
    ax.set_yticks(range(3), [PAIR_LABELS[name] for name in data['comparisons']], fontsize=9)
    ax.set(title='Кому стало лучше или хуже по RR@10', xlabel='Все запросы, включая unknown/off-pool')
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.12), ncol=3)
    ax.invert_yaxis()
    figures.append(_save(plt, fig, output, '05-target-ranks', 'Изменения рангов с полным знаменателем', caption +
        ' Unknown и off-pool показаны отдельно; ранги 11–1024 имеют нулевой RR@10, но сохраняют полный target rank.'))
    if data['stage'] == 'screen':
        screen = data['screen_diagnostics']
        fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
        for j, (utility, title) in enumerate((('completion', 'Completion · нейтральный centered proxy'),
                                             ('residual', 'Residual · нейтральный centered proxy'),
                                             ('prior', 'Observed-fit prior · centered proxy'))):
            _bars(axes[0, j], [screen[arm]['utilities'][utility]['centered_supported_neutral_proxy'] for arm in ARMS], title, decimals=4)
        for j, (key, title, pct) in enumerate((
            ('observed_H63_seen_at10', 'Доля top10 из observed H63', True),
            ('mean_log1p_strict7d_count_at10', 'Средний log1p strict7d count в top10', False),
            ('unsupported_share_at10', 'Доля top10 вне поддержки RM', True))):
            _bars(axes[1, j], [screen[arm]['behavior'][key] for arm in ARMS], title, pct, 2)
        figures.append(_save(plt, fig, output, '06-screen-utilities-behavior', 'Что оптимизируется и что меняется в выдаче', caption +
            ' Для каждой utility: E_p[support·(u−E_C[u|supported,s])]; unsupported вклад равен0, все1024 действия в softmax. '
            'Prior — observed-fit item mean; conditional means не трактуются как сопоставимый expected utility. H63 — исходный truncated prefix.'))
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.1))
        for row, name in enumerate(('RL-completion-minus-C', 'RL-residual-minus-C')):
            for col, utility in enumerate(('completion', 'residual')):
                ax = axes[row, col]
                decomposition = data['item_decomposition'][name][utility]
                items = decomposition['top_items_by_absolute_contribution'][:12]
                values = [item['contribution'] for item in items]
                ax.barh(range(len(items)), values, color=['#007F83' if value >= 0 else '#CB6A3B' for value in values], height=.72)
                ax.set_yticks(range(len(items)), [str(item['raw_id']) for item in items], fontsize=8)
                ax.invert_yaxis()
                ax.axvline(0, color='#526174', linewidth=.8)
                ax.set(title=PAIR_LABELS[name] + ' · ' + utility,
                       xlabel='Точный вклад в средний сдвиг centered proxy', ylabel='raw item ID')
                ax.grid(axis='x')
        figures.append(_save(plt, fig, output, '07-item-contributions', 'Какие фильмы внесли вклад в изменение proxy', caption +
            ' Вклад = mean_s[(p_RL−p_C)·support·(u−E_C[u|supported,s])]. Показаны12 наибольших |вкладов| на панель; '
            'полная сумма по каталогу и остаток сохранены в JSON/CSV. Это арифметическое разложение, не причинная атрибуция.'))
    index = {'contract': 'kion-scientific3-figures-v1', 'state': 'completed', 'stage': data['stage'],
             'rows': data['rows'], 'figures': figures, 'requires_visual_QA': True,
             'source': {'path': Path(__file__).resolve().as_posix(),
                        'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
             'aggregate_canonical_sha256': hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                 separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest(),
             'matplotlib_version': matplotlib.__version__}
    with (output / 'figures.json').open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(index, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return index


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Aggregate plot-data.json')
    parser.add_argument('--output', type=Path, required=True, help='Fresh/empty output directory')
    args = parser.parse_args()
    result = render(read(args.input), args.output)
    print(json.dumps({'state': result['state'], 'figures': len(result['figures'])}, ensure_ascii=False))
