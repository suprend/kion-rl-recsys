"""Figure08: saved-summary-only five-step diagnostics; root renders explicitly.

Import, validate, needed_raw_ids and prepare_series are stdlib-only. render is
the only Matplotlib entrypoint. It never opens arrays, models, or the underlying
rollout/catalog; their saved pins travel into figures.json as provenance.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import textwrap

CONTRACT = 'kion-rollout-figures-v1'
SUMMARY_CONTRACT = 'kion-final3-saved-liveprefix-summary-v1'
ARMS = ('C', 'RL-completion', 'RL-residual')
MODES = ('greedy', 'categorical')
MASKS = ('none', 'exclude-generated')
POSITIONS = [253, 2, 30, 246, 204, 158, 96, 50, 128, 244, 255, 36, 12, 109, 171, 236]
LABELS = {'C': 'C · исходная', 'RL-completion': 'RL · completion', 'RL-residual': 'RL · residual'}
COLORS = {'C': '#52677F', 'RL-completion': '#007F83', 'RL-residual': '#CB6A3B'}
MARKERS = {'C': 'o', 'RL-completion': '^', 'RL-residual': 's'}
LINESTYLES = {'C': '-', 'RL-completion': '-', 'RL-residual': '--'}
FACETS = tuple((mode, mask) for mode in MODES for mask in MASKS)
FIGURES = (
    dict(file='08-five-step-panel', title='Пять рекомендаций: ожидаемый proxy и изменение политики',
         metrics=('expected_neutral_proxy', 'expected_neutral_residual_proxy', 'kl_nats')),
    dict(file='08b-five-step-behavior', title='Пять рекомендаций: энтропия, история и повторы',
         metrics=('entropy_nats', 'original_seen_fraction', 'generated_repeat_fraction')),
)
METRICS = {
    'expected_neutral_proxy': dict(title='Completion · centered proxy', unit='Средний proxy', bounds=(-1., 1.), kind='estimate'),
    'expected_neutral_residual_proxy': dict(title='Residual · centered proxy', unit='Средний proxy', bounds=(-2., 2.), kind='estimate'),
    'kl_nats': dict(title='KL к C на том же prefix', unit='KL, нат', bounds=(0., None), kind='estimate'),
    'entropy_nats': dict(title='Энтропия выбора', unit='Энтропия, нат', bounds=(0., math.log(1024)), kind='estimate'),
    'original_seen_fraction': dict(title='Выбор из исходного H63', unit='Рекомендации, %', bounds=(0., 1.), kind='fraction'),
    'generated_repeat_fraction': dict(title='Повтор уже сгенерированного', unit='Рекомендации, %', bounds=(0., 1.), kind='fraction'),
}
SCOPE = ('16 заранее выбранных контекстов, по одной categorical-траектории с seed=20261024+position; '
         '192 траектории, 960 решений. Средние описывают только этот фиксированный grid, без популяционных интервалов.')
UTILITY_SCOPE = ('RM, raw1068, observed history и clock заморожены в s0; меняются GPT prefix и q/base/source12. '
                 'Proxy = Eₚ[support·(u−E_C(s0)[u|support])], unsupported вклад равен 0; '
                 'residual u=r_RM−m_fit. Это surrogate, не наблюдаемый досмотр или retention.')
KL_SCOPE = ('KL сравнивает arm с C на ТОМ ЖЕ текущем prefix arm и с той же маской. '
            'Это не KL к распределению на независимо сгенерированной C-траектории. '
            'После первого шага prefixes разных arms могут различаться.')
BEHAVIOR_SCOPE = ('H63 — доступный observed prefix перед генерацией, не вся история пользователя. '
                  'Повтор — item, уже выбранный в этой synthetic-траектории; exclude-generated запрещает такие повторы. '
                  'Новые observed outcomes и relevance labels для шагов 2–5 не создаются.')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite(value, name):
    require(type(value) in (int, float) and math.isfinite(value), 'Finite numeric '+name+' required')
    return float(value)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical_sha(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _digest(value, name):
    require(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
            'Invalid SHA256 '+name)


def _pin(value, name):
    require(isinstance(value, dict) and isinstance(value.get('path'), str) and bool(value['path']),
            'Missing saved provenance '+name)
    _digest(value.get('sha256'), name)


def _metric(step, name):
    spec = METRICS[name]
    if spec['kind'] == 'estimate':
        value = step.get(name)
        require(isinstance(value, dict) and value.get('denominator') == value.get('total') == 16,
                'Full sixteen-context denominator required for '+name)
        result = finite(value.get('mean'), name)
    else:
        result = finite(step.get(name), name)
    low, high = spec['bounds']
    require(result >= low-1e-7 and (high is None or result <= high+1e-7), 'Metric outside bounds: '+name)
    return result


def validate(data):
    """Reject partial/old summaries, missing metrics and ambiguous denominators."""
    require(isinstance(data, dict) and data.get('contract') == SUMMARY_CONTRACT and
            data.get('state') == 'completed', 'Completed scientific3 rollout summary required')
    require(data.get('trajectory_count') == 192 and data.get('position_count') == 16 and data.get('steps') == 5 and
            data.get('seed') == 20261024 and data.get('expected_positions') == POSITIONS,
            'Exact full192/16/5 summary and original label-blind position order required')
    for field in ('rollout', 'source', 'items_catalog'):
        _pin(data.get(field), field)
    metadata = data.get('source_metadata', {})
    require(metadata.get('contract') == 'kion-final3-liveprefix-grid-v1' and metadata.get('five_step_training') is False and
            metadata.get('synthetic_prefix_only') is True and metadata.get('RM_observed_s0_fixed') is True,
            'Frozen observed-s0 synthetic inference scope required')
    for field in ('cache_sha256', 'reward_sha256', 'utility_sha256'):
        _digest(metadata.get(field), field)
    groups = data.get('group_summaries')
    require(isinstance(groups, list) and len(groups) == 12, 'All twelve mode/mask/arm groups required')
    expected = {(arm, mode, mask) for arm in ARMS for mode, mask in FACETS}
    found = {}
    for group in groups:
        key = (group.get('arm'), group.get('selection'), group.get('repeat_mask'))
        require(key in expected and key not in found and group.get('positions') == 16, 'Unexpected/duplicate/incomplete group')
        steps = group.get('per_step')
        require(isinstance(steps, list) and [step.get('step') for step in steps] == [1, 2, 3, 4, 5],
                'Exactly ordered steps1–5 required')
        for j, step in enumerate(steps):
            values = {name: _metric(step, name) for name in METRICS}
            allowed = 1024-j if key[2] == 'exclude-generated' else 1024
            require(values['entropy_nats'] <= math.log(allowed)+1e-7, 'Entropy exceeds current mask support')
            if key[2] == 'exclude-generated':
                require(abs(values['generated_repeat_fraction']) <= 1e-10, 'Exclude-generated repeat fraction must be zero')
            if key[0] == 'C':
                require(abs(values['kl_nats']) <= 1e-7, 'C-to-identical-C KL must be zero')
                if j == 0:
                    require(abs(values['expected_neutral_proxy']) <= 1e-7 and
                            abs(values['expected_neutral_residual_proxy']) <= 1e-7, 'C centered s0 expectations must be zero')
        found[key] = group
    require(set(found) == expected, 'Incomplete Cartesian summary grid')
    comparisons = data.get('matched_C_comparisons')
    require(isinstance(comparisons, list) and len(comparisons) == 8 and
            {(row.get('arm'), row.get('selection'), row.get('repeat_mask')) for row in comparisons} ==
            {(arm, mode, mask) for arm in ARMS[1:] for mode, mask in FACETS} and
            all(row.get('reference') == 'C' and row.get('matched_positions') == 16 for row in comparisons),
            'Eight complete matched-C comparisons required')
    control = data.get('RM_greedy_unique5_control', {})
    require(control.get('positions') == 16 and control.get('mean_unique_generated') == 5,
            'All sixteen static unique5 RM controls required')
    # Reject nonfinite values even in saved metrics not selected for the panels.
    canonical_sha(data)
    return data


def needed_raw_ids(data):
    """No item names appear on these aggregate time panels."""
    validate(data)
    return set()


def prepare_series(data):
    """Build an inspectable plotting plan without importing a numerical runtime."""
    validate(data)
    groups = {(row['arm'], row['selection'], row['repeat_mask']): row for row in data['group_summaries']}
    result = []
    for figure in FIGURES:
        panels = []
        for mode, mask in FACETS:
            for metric in figure['metrics']:
                multiplier = 100. if METRICS[metric]['kind'] == 'fraction' else 1.
                panels.append(dict(selection=mode, repeat_mask=mask, metric=metric,
                    series={arm: [_metric(step, metric)*multiplier for step in groups[arm, mode, mask]['per_step']]
                            for arm in ARMS}))
        caption = SCOPE+' '+(UTILITY_SCOPE+' '+KL_SCOPE if figure['file'] == '08-five-step-panel' else BEHAVIOR_SCOPE+' '+UTILITY_SCOPE)
        result.append(dict(**figure, chart_type='faceted-line', panels=panels, caption=caption))
    return result


def _limits(values, metric):
    low, high = min(values), max(values)
    if METRICS[metric]['kind'] == 'fraction':
        return 0., 100.
    if metric in ('kl_nats', 'entropy_nats'):
        return 0., max(high*1.12, .02 if metric == 'kl_nats' else 1.)
    low, high = min(low, 0.), max(high, 0.)
    padding = max((high-low)*.12, .004)
    return low-padding, high+padding


def render(data, output, labels=None):
    """Render two SVG/PNG figures from one completed saved summary dict.

    labels is accepted for the generic root wrapper but unused: no item IDs are
    plotted. Only the caller may read/pin the summary JSON file; figures.json
    includes its canonical content hash plus the saved rollout/source/catalog pins.
    """
    plans = prepare_series(data)
    output = Path(output).resolve()
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())), 'Fresh/empty figure directory required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter
    versions = {name: getattr(sys.modules.get(name), '__version__', None) for name in ('matplotlib', 'numpy')}
    require(all(isinstance(value, str) and value for value in versions.values()), 'Actual plotting/runtime versions required')
    output.mkdir(parents=True, exist_ok=True)
    style = {'font.family':'DejaVu Sans', 'font.size':10, 'axes.titlesize':12, 'axes.labelsize':10,
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.titleweight':'semibold',
        'axes.labelcolor':'#253445', 'text.color':'#253445', 'xtick.color':'#526174', 'ytick.color':'#526174',
        'grid.color':'#DDE4EB', 'grid.linewidth':.65, 'figure.facecolor':'#FFFFFF',
        'axes.facecolor':'#FFFFFF', 'savefig.facecolor':'#FFFFFF', 'svg.fonttype':'none'}
    figures = []
    with plt.rc_context(style):
        for plan in plans:
            fig, axes = plt.subplots(4, 3, figsize=(15.8, 14.5), sharex=True, squeeze=False)
            try:
                for row, (mode, mask) in enumerate(FACETS):
                    for col, metric in enumerate(plan['metrics']):
                        ax = axes[row][col]
                        panel = plan['panels'][row*3+col]
                        for arm in ARMS:
                            ax.plot(range(1,6), panel['series'][arm], color=COLORS[arm], marker=MARKERS[arm],
                                linestyle=LINESTYLES[arm], markersize=5, linewidth=1.8, label=LABELS[arm])
                        all_values = [value for other in plan['panels'] if other['metric'] == metric
                                      for values in other['series'].values() for value in values]
                        ax.set_ylim(*_limits(all_values, metric))
                        ax.set_xlim(.85,5.15)
                        ax.set_xticks([1,2,3,4,5])
                        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                        if METRICS[metric]['kind'] == 'fraction':
                            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
                        if row == 0:
                            ax.set_title(METRICS[metric]['title'],loc='left',pad=13)
                        ax.set_ylabel(METRICS[metric]['unit'])
                        if row == 3:
                            ax.set_xlabel('Шаг генерации')
                        ax.grid(axis='y')
                        ax.set_axisbelow(True)
                        if metric.startswith('expected_neutral'):
                            ax.axhline(0,color='#7A8998',linewidth=.8,zorder=0)
                        if col == 0:
                            mode_label = 'Greedy' if mode == 'greedy' else 'Categorical'
                            mask_label = 'без repeat mask' if mask == 'none' else 'exclude-generated'
                            ax.text(-.29,.5,mode_label+'\n'+mask_label,transform=ax.transAxes,
                                    ha='right',va='center',fontsize=10,weight='semibold',linespacing=1.6)
                fig.suptitle(plan['title'],x=.08,y=.985,ha='left',fontsize=18,weight='semibold')
                fig.text(.08,.958,'16 фиксированных контекстов · три arms · шаги 1–5 · mode и mask разделены',
                         fontsize=10,color='#526174')
                handles, names = axes[0][0].get_legend_handles_labels()
                fig.legend(handles,names,loc='upper left',bbox_to_anchor=(.075,.943),ncol=3,frameon=False,fontsize=10)
                caption_lines = textwrap.wrap(plan['caption'],width=165)
                footer_height = .023+.012*len(caption_lines)
                fig.text(.08,.022,'\n'.join(caption_lines),ha='left',va='bottom',fontsize=8.4,color='#526174',linespacing=1.4)
                fig.tight_layout(rect=(.075,footer_height,.99,.91),h_pad=2.9,w_pad=2.7)
                files = {}
                for extension in ('svg','png'):
                    file = output/(plan['file']+'.'+extension)
                    fig.savefig(file,dpi=190,bbox_inches='tight')
                    files[extension] = dict(path=file.as_posix(),sha256=sha(file),bytes=file.stat().st_size)
                figures.append(dict(file=plan['file'],title=plan['title'],caption=plan['caption'],chart_type='faceted-line',
                    formats=['svg','png'],files=files,metrics=list(plan['metrics']),
                    facets=[dict(selection=mode,repeat_mask=mask) for mode,mask in FACETS],
                    y_scales='shared per metric across all mode/mask facets; no dual axes'))
            finally:
                plt.close(fig)
    index = dict(contract=CONTRACT,state='completed',stage='rollout',rows=16,rows_scope='fixed initial contexts; not users',
        position_count=16,trajectory_count=192,steps=5,seed=20261024,figures=figures,versions=versions,
        requires_visual_QA=True,rendered_visual_QA=False,labels_used=False,
        input=dict(contract=SUMMARY_CONTRACT,canonical_sha256=canonical_sha(data),rollout=data['rollout'],
                   summary_source=data['source'],items_catalog=data['items_catalog'],source_metadata=data['source_metadata']),
        source=dict(path=Path(__file__).resolve().as_posix(),sha256=sha(__file__)),
        metric_scope=dict(utilities=UTILITY_SCOPE,KL=KL_SCOPE,behavior=BEHAVIOR_SCOPE,population=SCOPE))
    with (output/'figures.json').open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(index,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    return index
