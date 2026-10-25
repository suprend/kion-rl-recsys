"""Render the 65,536-query comparison and audited 256-history rollout diagnostics.
Run: pas.ps1 python -B D:/kion-rl-recsys/report-latex/figure.py
No training, raw user records, or model inference are used.
"""
from pathlib import Path
import json
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE=Path(__file__).resolve().parent

def render_evaluation():
    data=json.loads((HERE/'evaluation64k.json').read_text(encoding='utf-8'))
    assert data['rows']==65536 and data['users']==53024
    pairs=[('completion-equal-seed-mean','C','Completion\nпо сравнению с C'),
           ('residual-equal-seed-mean','C','Residual\nпо сравнению с C'),
           ('residual-equal-seed-mean','completion-equal-seed-mean','Residual\nпо сравнению с completion')]
    rows=[]
    for model,reference,label in pairs:
        found=[r for r in data['contrasts'] if r['model']==model and r['reference']==reference and r['metric']=='MAP@10']
        assert len(found)==1
        rows.append((label,found[0]))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'path',
                         'text.color':'#243746','axes.labelcolor':'#243746','xtick.color':'#536472',
                         'ytick.color':'#536472','axes.edgecolor':'#CBD3DA','axes.linewidth':.7})
    fig,ax=plt.subplots(figsize=(7.2,3.7))
    fig.subplots_adjust(left=.37,right=.97,bottom=.24,top=.81)
    ax.axvspan(-.46,0,color='#F7F2EF',zorder=0)
    ax.axvspan(0,.36,color='#EFF6F6',zorder=0)
    for y,(label,row) in zip([2,1,0],rows):
        val=row['difference']*100; lo,hi=[x*100 for x in row['ci95']]
        assert lo<=val<=hi
        color=['#167D8D','#536472','#C16B43'][y]
        ax.errorbar(val,y,xerr=[[val-lo],[hi-val]],fmt='o',color=color,capsize=4,elinewidth=2.5,markersize=7)
        ax.annotate(f'{val:+.3f}'.replace('.',','),(val,y),xytext=(0,10),textcoords='offset points',ha='center',fontsize=10)
    ax.set_yticks([2,1,0],[x[0] for x in rows]); ax.tick_params(axis='y',length=0,pad=10)
    ax.set_ylim(-.55,2.6); ax.set_xlim(-.46,.36)
    ax.set_xticks([-.4,-.2,0,.2]);ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:.1f}'.replace('.',',')))
    ax.axvline(0,color='#777',linestyle='--',linewidth=1)
    ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    ax.spines[['top','right','left']].set_visible(False)
    ax.set_xlabel('Разница MAP@10, процентные пункты',labelpad=10)
    fig.suptitle('Как изменилось качество после RL',fontsize=13,fontweight='bold',y=.98)
    fig.text(.5,.88,'65 536 запросов · среднее трёх запусков · интервалы 95%',ha='center',fontsize=9,color='#536472')
    fig.text(.37,.025,'← Первая модель хуже',fontsize=9,color='#536472')
    fig.text(.97,.025,'Первая модель лучше →',ha='right',fontsize=9,color='#167D8D')
    fig.savefig(HERE/'images/train-kl-and-map.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(HERE/'images/train-kl-and-map.svg')
    plt.close(fig)

def render_rollout():
    """Render audited saved aggregates; no model or bootstrap is run here."""
    data = json.loads((HERE / 'rollout256.json').read_text(encoding='utf-8'))
    assert data['histories'] == data['users'] == 256
    assert data['steps'] == 5 and data['repeat_mask'] == 'none'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'text.color': '#243746',
                         'axes.labelcolor': '#243746', 'xtick.color': '#536472',
                         'ytick.color': '#243746', 'axes.edgecolor': '#CBD3DA'})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2), sharey=True)
    fig.subplots_adjust(left=.18, right=.975, top=.73, bottom=.28, wspace=.35)
    families = [('C', 'C'), ('completion', 'Completion'), ('residual', 'Residual')]
    modes = [('greedy', 'Максимум вероятности (greedy)', '#C16B43', 'o', .14),
             ('categorical', 'Выбор по вероятностям (categorical)', '#167D8D', 'D', -.14)]
    panels = [('unique5', 1, 'Разных фильмов из пяти', 'Среднее на историю', (0, 5.6), [0, 1, 2, 3, 4, 5]),
              ('max_share', 100, 'Концентрация рекомендаций', 'Доля самого частого фильма, %', (0, 66), [0, 20, 40, 60])]
    for ax, (metric, scale, title, xlabel, limits, ticks) in zip(axes, panels):
        for i, (family, _) in enumerate(families):
            y = 2 - i
            ax.axhspan(y - .39, y + .39, color='#F4F7F8', zorder=0)
            for mode, label, color, marker, offset in modes:
                row = next(r for r in data['groups'] if r['family'] == family and r['selection'] == mode)
                value = row['metrics'][metric]['estimate'] * scale
                lo, hi = [v * scale for v in row['metrics'][metric]['ci95']]
                assert lo <= value <= hi
                ax.errorbar(value, y + offset, xerr=[[value - lo], [hi - value]],
                            fmt=marker, color=color, markersize=5, capsize=3,
                            elinewidth=1.5, label=label if i == 0 else None)
                text = (f'{value:.2f}' if metric == 'unique5' else f'{value:.1f}').replace('.', ',')
                ax.annotate(text, (hi, y + offset), xytext=(5, 0),
                            textcoords='offset points', va='center', fontsize=9, color=color)
        ax.set_title(title, fontsize=10, fontweight='bold', pad=13)
        ax.set_xlabel(xlabel, fontsize=9, labelpad=9)
        ax.set_xlim(*limits); ax.set_xticks(ticks)
        ax.set_ylim(-.5, 2.5); ax.set_yticks([2, 1, 0], [label for _, label in families])
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(length=0, pad=7)
        ax.grid(axis='x', color='#DCE3E7', linewidth=.6); ax.set_axisbelow(True)
    fig.suptitle('Пять рекомендаций: повторы и концентрация', fontsize=13, fontweight='bold', y=.98)
    fig.text(.5, .895, '256 историй / пользователей · без запрета повторов', ha='center', fontsize=9, color='#536472')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.5, .865), ncol=2, frameon=False, fontsize=8, columnspacing=1.2)
    fig.text(.18, .135, 'Greedy: повтор есть во всех историях каждой модели.', fontsize=8.5)
    fig.text(.18, .085, 'Completion и Residual: среднее трёх фиксированных запусков; интервалы 95%.', fontsize=7.6, color='#536472')
    fig.text(.18, .04, 'Синтетические продолжения историй, а не наблюдаемое поведение пользователей.', fontsize=7.6, color='#536472')
    fig.savefig(HERE / 'images/rollout-steps.pdf', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollout-only', action='store_true', help='Render only the rollout256 figure')
    args = parser.parse_args()
    if not args.rollout_only:
        render_evaluation()
    render_rollout()
