"""Create the four-page assignment report from recorded experiment outputs."""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak


def main(repo_url='https://github.com/Lohith600/ML_ASSIGNMNET'):
    metrics = json.loads(Path('outputs/metrics.json').read_text())
    assets = Path('outputs/figures')
    assets.mkdir(exist_ok=True)
    destination = Path('output/pdf')
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 180})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    for ax, var in zip(axes, metrics):
        cv = pd.read_csv(f'outputs/{var}_cv_results.csv')
        best = cv.loc[cv.groupby('degree').cv_mse.idxmin()].sort_values('degree')
        chosen = metrics[var]['selected']
        ax.plot(best.degree, best.cv_mse, 'o-', color='#126782', markersize=3)
        ax.scatter([chosen['degree']], [chosen['cv_mse']], s=60, color='#d66b35', zorder=5)
        ax.set(xlabel='Total polynomial degree', ylabel='Mean validation MSE (log scale)',
               title=var + ': degree selection', yscale='log')
        ax.set_xticks(range(1, int(best.degree.max()) + 1, 2))
        ax.grid(alpha=.2)
    fig.savefig(assets / 'degree_selection.png')
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.1), constrained_layout=True)
    for j, var in enumerate(metrics):
        h = pd.read_csv(f'outputs/{var}_holdout.csv')
        ax = axes[0, j]
        ax.scatter(h.y_true, h.y_pred, alpha=.6, s=12, color='#126782')
        low, high = min(h.y_true.min(), h.y_pred.min()), max(h.y_true.max(), h.y_pred.max())
        ax.plot([low, high], [low, high], '--', color='#d66b35', lw=1)
        ax.set(title=var + ': held-out predictions', xlabel='Actual y', ylabel='Predicted y')
        ax = axes[1, j]
        ax.scatter(h.y_pred, h.residual, alpha=.6, s=12, color='#126782')
        ax.axhline(0, color='#d66b35', ls='--', lw=1)
        ax.set(title=var + ': held-out residuals', xlabel='Predicted y', ylabel='Actual minus predicted')
    fig.savefig(assets / 'holdout_diagnostics.png')
    plt.close(fig)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='ReportTitle', fontName='Helvetica-Bold', fontSize=23,
                              leading=28, textColor=colors.HexColor('#17364a'), spaceAfter=12))
    styles.add(ParagraphStyle(name='Deck', fontSize=11, leading=15, textColor=colors.HexColor('#426071'), spaceAfter=14))
    styles['BodyText'].fontSize = 10
    styles['BodyText'].leading = 14
    styles['BodyText'].spaceAfter = 8
    styles['Heading2'].fontSize = 13
    styles['Heading2'].leading = 17
    styles['Heading2'].textColor = colors.HexColor('#126782')
    styles['Heading2'].spaceBefore = 12
    styles['Heading2'].spaceAfter = 7
    styles.add(ParagraphStyle(name='CaptionSmall', fontSize=8.5, leading=11,
                              textColor=colors.HexColor('#506370'), spaceAfter=9))
    styles.add(ParagraphStyle(name='CodeBlock', fontName='Courier', fontSize=8.2, leading=12,
                              backColor=colors.HexColor('#eef4f6'), borderPadding=8,
                              spaceBefore=7, spaceAfter=13))
    story = []
    def para(text, style='BodyText'):
        story.append(Paragraph(text, styles[style]))
    def heading(text):
        para(text, 'Heading2')
    def table(rows, widths):
        cells = [[Paragraph(str(cell), styles['CaptionSmall']) for cell in row] for row in rows]
        t = Table(cells, colWidths=widths, hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e4eef2')),
                               ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                               ('LEFTPADDING', (0, 0), (-1, -1), 8),
                               ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                               ('TOPPADDING', (0, 0), (-1, -1), 7),
                               ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                               ('LINEBELOW', (0, 0), (-1, 0), .7, colors.HexColor('#126782')),
                               ('LINEBELOW', (0, 1), (-1, -1), .3, colors.HexColor('#d9e2e7'))]))
        story.append(t)
        story.append(Spacer(1, 8))
    def page(title, subtitle):
        para(title, 'ReportTitle')
        para(subtitle, 'Deck')
    page('Polynomial Regression', 'Assignment 1 | BT2024260 | 8 October 2026')
    para('Two independent polynomial ridge models were trained to predict Net Power Score '
         '(var1) and Thermal Anomaly Score (var2). Degree and regularization were selected '
         'by three-fold cross-validation on development data. A separate grouped holdout '
         'was then evaluated, before each final model was refitted on all labeled data.')
    heading('1. Problems and data')
    table([['Dataset', 'Inputs and target', 'Train / test rows'],
           ['var1', 'Six plant settings, x1-x6; Net Power Score y', '1,000 / 1,000'],
           ['var2', 'Three coordinate offsets, x1-x3; Thermal Anomaly Score y', '1,000 / 1,000']],
          [58, 313, 116])
    para('Only the four files matching roll number BT2024260 were used. All input and '
         'target values were finite, with no missing entries. Input values in both problems '
         'lie in [-1, 1]. No imputation, additional input scaling, or target transformation '
         'was needed. The supplied sample submission contains one column, <b>y</b>, and '
         '1,000 rows.')
    para('Repeated input vectors occur in the training sets: one extra repeated row in var1 '
         'and nine in var2. Their responses were retained. Identical input vectors were '
         'kept together in every split to prevent a repeated location or setting from '
         'appearing on both sides of a validation split.')
    heading('2. Selected models and holdout results')
    rows = [['Problem', 'Degree', 'Terms*', 'Ridge alpha', 'MSE', 'R2']]
    for var, m in metrics.items():
        s, h = m['selected'], m['holdout']
        rows.append([var, s['degree'], s['features'], f"{s['alpha']:.6f}", f"{h['mse']:.6f}", f"{h['r2']:.6f}"])
    table(rows, [62, 56, 57, 102, 106, 104])
    para('*Nonconstant polynomial terms; a separate intercept is also fitted. '
         'Scores are measured on the reserved holdout split.', 'CaptionSmall')
    para('The mean-prediction baseline has holdout MSE 12.9872 for var1 and 46.1607 for '
         'var2. Both selected polynomial models substantially reduce this error. '
         'Larger degrees were evaluated but did not improve cross-validation MSE beyond '
         'the selected configurations.')
    heading('Deliverables')
    para('Two prediction CSVs, saved model parameters, training and inference scripts, '
         'numerical verification, and recorded validation results accompany this report. '
         'The GitHub link will be added after local review.' if not repo_url else
         'Two prediction CSVs, saved model parameters, training and inference scripts, '
         'numerical verification, and recorded validation results accompany this report. '
         'The repository link appears on page 4.')

    story.append(PageBreak())
    page('Model selection', 'Polynomial features, regularization, and validation protocol')
    heading('3. Polynomial ridge regression')
    para('The feature map contains every monomial whose total degree is at most d. '
         'For example, x1 squared times x2 is a degree-3 term. With p inputs, the number '
         'of nonconstant terms is C(p + d, d) - 1. This includes interactions between '
         'inputs, rather than only separate powers of each input.')
    para('Objective = sum of squared prediction errors<br/>+ alpha * sum of squared coefficients', 'CodeBlock')
    para('The intercept is unpenalized. Polynomial columns and the target are centered '
         'using only the current training partition. No column variance normalization '
         'is applied after expansion: ridge penalizes coefficients in the original '
         'monomial basis. Inputs are already bounded, limiting the magnitude of raw '
         'powers. The objective uses a sum of squared errors, not a mean, so alpha '
         'must be interpreted with that convention.')
    heading('4. Search and split design')
    para('Unique input groups were shuffled with random seed 42. Twenty percent of '
         'groups were reserved for the holdout: 200 rows for var1 and 198 rows for '
         'var2. The remaining 800 and 802 development rows were divided into three '
         'grouped folds. The same folds were used for all candidates.')
    para('The search covered every degree from 1 to 10 for var1 and 1 to 20 for var2, '
         'with 23 logarithmically spaced alpha values from 10^-8 to 10^3 at each degree. '
         'The configuration with the smallest mean fold MSE was selected. The holdout '
         'was not used to choose or revise these settings.')
    story.append(Image(str(assets / 'degree_selection.png'), width=487, height=203))
    para('Figure 1. Best mean cross-validation MSE at each degree, after selecting alpha '
         'within that degree. Orange markers show the chosen degrees. The vertical '
         'axis is logarithmic.', 'CaptionSmall')
    para('Var1 reaches its minimum CV MSE of 0.584824 at degree 5; degree 10 gives '
         '0.885626. Var2 reaches 0.339568 at degree 13; degree 20 gives 0.407304. '
         'The selected models therefore balance flexibility and regularization using '
         'observed validation performance.')

    story.append(PageBreak())
    page('Validation evidence', 'Predictions, residuals, and performance across the input range')
    para('After selection, each model was fitted on its full development partition and '
         'evaluated on its reserved holdout. MSE measures mean squared prediction '
         'error. R2 is 1 minus residual sum of squares divided by the total squared '
         'deviation of the evaluated targets from their own mean.')
    story.append(Image(str(assets / 'holdout_diagnostics.png'), width=487, height=345))
    para('Figure 2. Top: actual versus predicted scores, with the perfect-prediction line. '
         'Bottom: residuals against predictions. Every point is a held-out row. '
         'The plots are diagnostics; no settings were changed after examining them.', 'CaptionSmall')
    heading('5. Boundary diagnostics')
    table([['Problem', 'Boundary rows:<br/>train / test', 'Boundary<br/>holdout MSE', 'Interior<br/>holdout MSE'],
           ['var1', '87.4% / 98.2%', '0.517048', '0.334074'],
           ['var2', '57.8% / 58.6%', '0.286341', '0.211126']], [62, 163, 131, 131])
    para('A boundary row has at least one input with absolute value at least 0.999999. '
         'Var1 has more boundary rows in the test data than in training. Its random '
         'holdout may therefore not fully represent the test distribution. The '
         'boundary diagnostic is descriptive and was not used to retune the model.')
    para('Development-fit MSE is 0.192881 for var1 and 0.171147 for var2. The higher '
         'holdout errors show why training fit alone is insufficient for model selection.')

    story.append(PageBreak())
    page('Reproducibility and delivery', 'Final fitting, output verification, and remaining submission step')
    heading('6. Final training and inference')
    para('After holdout evaluation, each selected configuration was refitted on all '
         '1,000 labeled rows. The saved NPZ files contain the exponent matrix, '
         'coefficient vector, intercept, feature names, degree, and alpha. The inference '
         'script reconstructs the same polynomial features and computes one score per '
         'test row. No target clipping or postprocessing is applied.')
    para('NumPy and SciPy solve ridge regression using the smaller of the feature-space '
         'and sample-space Gram matrices. Validation reuses an eigendecomposition '
         'across alpha values; final fitting uses a positive-definite linear solve.')
    heading('7. Reproduce the work')
    para('Use Python 3.12 with the dependencies in requirements.txt. Place the four '
         'personalized datasets and sample_submission.csv beside the scripts, then run:')
    para('python -m pip install -r requirements.txt<br/>python train.py<br/>python verify.py<br/>python build_report.py', 'CodeBlock')
    table([['File', 'Purpose'],
           ['polynomial.py / train.py', 'Polynomial terms, ridge solver, grouped selection, final fitting'],
           ['predict.py', 'Load a saved model and generate predictions without retraining'],
           ['verify.py', 'Independent numerical checks and submission validation'],
           ['outputs/metrics.json', 'Selected configurations and recorded evaluation metrics'],
           ['outputs/*_cv_results.csv', 'Every searched degree/alpha and its three fold errors'],
           ['outputs/BT2024260_pred_var1.csv<br/>outputs/BT2024260_pred_var2.csv', 'Final test predictions: one y column and 1,000 rows per file']],
          [227, 260])
    heading('8. Verification and limitations')
    para('Both primal and dual ridge implementations passed comparison against an '
         'independent augmented least-squares solution. Checks also passed for known '
         'polynomial terms, group separation, saved-model predictions, finite output '
         'values, CSV column names, row counts, row order, and numerical CSV round-trip '
         'agreement. The original datasets were not modified.')
    para('The holdout is only one split. Predictions may be less reliable in sparsely sampled regions, '
         'particularly under the observed var1 input shift.')
    heading('Repository')
    if repo_url:
        safe = escape(repo_url, {'"': '&quot;'})
        para(f'<link href="{safe}" color="#126782">{safe}</link>')
    else:
        para('Repository creation is deferred until local review is complete. '
             'The GitHub link is the remaining submission item; this local report '
             'must be updated with the actual URL before final submission.')

    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor('#cfdde4'))
        canvas.line(54, 44, 541, 44)
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#506370'))
        canvas.drawString(54, 30, 'BT2024260 | Assignment 1: Polynomial Regression')
        canvas.drawRightString(541, 30, str(doc.page))
    path = destination / 'BT2024260_report.pdf'
    doc = SimpleDocTemplate(str(path), pagesize=(595.28, 841.89), leftMargin=54, rightMargin=54,
                            topMargin=44, bottomMargin=58, title='Polynomial Regression - BT2024260',
                            author='BT2024260')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f'Wrote {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-url', default='https://github.com/Lohith600/ML_ASSIGNMNET',
                        help='Repository URL to include in the report.')
    args = parser.parse_args()
    main(args.repo_url)
