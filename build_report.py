"""Build the assignment report with clear wording and saved experiment results."""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak


def main(repo_url='https://github.com/Lohith600/ML_ASSIGNMNET'):
    results = json.loads(Path('outputs/metrics.json').read_text())
    comparisons = {v: json.loads(Path(f'outputs/{v}_model_comparison.json').read_text()) for v in results}
    assets = Path('outputs/figures')
    assets.mkdir(exist_ok=True)
    output = Path('output/pdf/BT2024260_report.pdf')
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 180})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    for ax, (var, result) in zip(axes, results.items()):
        for suffix, label, color in [('_cv_results', 'Ridge', '#126782'), ('_lasso_cv_results', 'Lasso', '#6d548d')]:
            cv = pd.read_csv(f'outputs/{var}{suffix}.csv')
            if label == 'Lasso':
                cv = cv[cv.max_relative_dual_gap <= 1.01e-6]
            best = cv.loc[cv.groupby('degree').cv_mse.idxmin()].sort_values('degree')
            ax.plot(best.degree, best.cv_mse, 'o-', color=color, markersize=3, label=label)
        s = result['selected']
        ax.scatter([s['degree']], [s['cv_mse']], color='#d66b35', s=60, zorder=5)
        ax.set(title=var + ': model comparison', xlabel='Polynomial degree', ylabel='Cross-validation MSE (log scale)', yscale='log')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.savefig(assets / 'degree_selection.png')
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.1), constrained_layout=True)
    for j, var in enumerate(results):
        data = pd.read_csv(f'outputs/{var}_holdout.csv')
        ax = axes[0, j]
        ax.scatter(data.y_true, data.y_pred, s=12, alpha=.6, color='#126782')
        lo, hi = min(data.y_true.min(), data.y_pred.min()), max(data.y_true.max(), data.y_pred.max())
        ax.plot([lo, hi], [lo, hi], '--', color='#d66b35')
        ax.set(title=var + ': predictions', xlabel='Actual score', ylabel='Predicted score')
        ax = axes[1, j]
        ax.scatter(data.y_pred, data.residual, s=12, alpha=.6, color='#126782')
        ax.axhline(0, ls='--', color='#d66b35')
        ax.set(title=var + ': prediction errors', xlabel='Predicted score', ylabel='Actual minus predicted')
    fig.savefig(assets / 'holdout_diagnostics.png')
    plt.close(fig)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleCustom', fontName='Helvetica-Bold', fontSize=23, leading=28, textColor=colors.HexColor('#17364a'), spaceAfter=12))
    styles.add(ParagraphStyle(name='Subtitle', fontSize=11, leading=15, textColor=colors.HexColor('#426071'), spaceAfter=16))
    styles['BodyText'].fontSize, styles['BodyText'].leading, styles['BodyText'].spaceAfter = 10.5, 15, 10
    styles['Heading2'].fontSize = 13
    styles['Heading2'].textColor = colors.HexColor('#126782')
    styles.add(ParagraphStyle(name='Small', fontSize=9, leading=12, spaceAfter=9))
    styles.add(ParagraphStyle(name='CodeCustom', fontName='Courier', fontSize=8.2, leading=13, backColor=colors.HexColor('#eef4f6'), borderPadding=8, spaceAfter=14))
    story = []
    def p(text, style='BodyText'):
        story.append(Paragraph(text, styles[style]))
    def h(text):
        p(text, 'Heading2')
    def page(title, subtitle):
        if story:
            story.append(PageBreak())
        p(title, 'TitleCustom')
        p(subtitle, 'Subtitle')
    def table(rows, widths):
        t = Table([[Paragraph(str(c), styles['Small']) for c in row] for row in rows], colWidths=widths)
        t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e4eef2')), ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('TOPPADDING', (0, 0), (-1, -1), 8), ('BOTTOMPADDING', (0, 0), (-1, -1), 7), ('LINEBELOW', (0, 0), (-1, -1), .3, colors.HexColor('#cbdbe2'))]))
        story.append(t)
        story.append(Spacer(1, 10))
    page('Polynomial Regression', 'Assignment 1 | BT2024260 | 8 October 2026')
    h('1. Aim of the assignment')
    p('The aim is to predict a score from the given inputs using polynomial regression. Two separate models are needed: one for the power plant settings (var1) and one for the underground location coordinates (var2).')
    table([['Problem', 'Inputs', 'Score to predict'], ['var1', 'Six plant settings: x1 to x6', 'Net Power Score'], ['var2', 'Three coordinates: x1 to x3', 'Thermal Anomaly Score']], [60, 220, 207])
    h('2. Data preparation')
    p('Only the datasets for roll number BT2024260 were used. Each problem has 1,000 training rows and 1,000 test rows. There were no missing values. All input values lie between -1 and 1.')
    p('Some rows share the same inputs. These rows were kept together when splitting the data, so the same setting or location could not appear in both the fitting and validation parts. All observed responses were retained.')
    h('3. Final models')
    table([['Problem', 'Method', 'Degree', 'Alpha', 'Nonzero terms'], *[[v, m['selected']['model'].title(), m['selected']['degree'], f"{m['selected']['alpha']:.6g}", m['final_nonzero_terms']] for v, m in results.items()]], [60, 95, 65, 122, 145])
    p('The degree controls how complex the polynomial can be. Alpha controls regularization, which discourages overly large coefficients. Nonzero terms are the polynomial terms that remain active in the final model.')
    p('Both methods produce polynomial regression models. Ridge reduces coefficient sizes. Lasso can also remove terms by setting their coefficients to zero. Each problem uses the method with the lower cross-validation error.')
    page('Choosing the models', 'A fair comparison of Ridge and Lasso')
    h('4. How the comparison was done')
    p('About 20% of the input groups were set aside as a holdout: 200 rows for var1 and 198 for var2. The remaining rows were split into three validation folds using seed 42. Each candidate was fitted on two folds and checked on the third, repeating this three times. The average MSE was used to choose the model.')
    p('For both methods, degrees 1-10 were tested for var1 and degrees 1-20 for var2. Ridge used 23 alpha values from 10^-8 to 10^3. Lasso used 19 values from 0.001 to 1 for var1 and 25 from 0.0001 to 1 for var2. The values were logarithmically spaced. Every candidate used the same data splits.')
    p('Lasso features were centered and scaled using only the fitting fold. Ridge features were centered without variance scaling. Both used an unpenalized intercept. The Lasso calculations were checked before comparing scores.')
    table([['Problem', 'Best Ridge CV MSE', 'Best Lasso CV MSE', 'Selected'], *[[v, f"{c['ridge']['cv_mse']:.6f}", f"{c['lasso']['cv_mse']:.6f}", c['winner'].title()] for v, c in comparisons.items()]], [60, 151, 151, 125])
    story.append(Image(str(assets / 'degree_selection.png'), width=487, height=203))
    p('Figure 1. Lower MSE is better. Each curve shows the best alpha at each degree. The orange dot marks the selected model. The vertical scale is logarithmic.', 'Small')
    p('The existing holdout had already been viewed in the earlier Ridge analysis. It was reused for the checks below, but not to choose the method, degree, or alpha.', 'Small')
    page('Checking the results', 'Scores and plots on the holdout rows')
    p('<b>MSE:</b> the average squared prediction error; lower is better. <b>R2:</b> how much target variation the predictions explain; values closer to 1 are better.')
    table([['Problem', 'Holdout MSE', 'Holdout R2', 'Mean-only MSE'], *[[v, f"{m['holdout']['mse']:.6f}", f"{m['holdout']['r2']:.6f}", f"{m['mean_baseline_holdout']['mse']:.6f}"] for v, m in results.items()]], [65, 141, 140, 141])
    story.append(Image(str(assets / 'holdout_diagnostics.png'), width=487, height=345))
    p('Figure 2. Top: predictions close to the diagonal line are accurate. Bottom: each point shows the prediction error. Errors close to zero are better.', 'Small')
    h('5. What to keep in mind')
    p('Var1 test inputs are more often at the range boundaries than training inputs (98.2% versus 87.4% of rows have at least one input at -1 or 1). This difference may affect accuracy. The holdout is one reused split, so it is a useful check rather than a new independent evaluation.')
    page('Final files and code', 'Reproducing the work and preparing the submission')
    h('6. Final training and predictions')
    p('After choosing the settings, each model was fitted again using all 1,000 training rows. Predictions were saved in the original test-row order. Each submission contains one column named y and exactly 1,000 predictions. Predictions were not clipped or otherwise adjusted.')
    h('7. Reproduce the results')
    p('Use Python 3.12. Put the four dataset files and sample_submission.csv beside the scripts, then run:')
    p('python -m pip install -r requirements.txt<br/>python train.py<br/>python verify.py<br/>python build_report.py', 'CodeCustom')
    table([['File', 'Purpose'], ['train.py / polynomial.py', 'Train models and generate predictions'], ['compare_lasso.py', 'Compare Lasso with Ridge on the same folds'], ['predict.py', 'Make predictions from saved models'], ['verify.py', 'Check calculations and prediction files'], ['outputs/metrics.json', 'Save selected settings and results'], ['outputs/BT2024260_pred_var1.csv<br/>outputs/BT2024260_pred_var2.csv', 'Prediction files for submission']], [232, 255])
    h('8. Implementation and checks')
    p('NumPy and SciPy are used for Ridge, and scikit-learn is used for Lasso. Ridge minimizes SSE + alpha times the sum of squared coefficients. Lasso minimizes SSE/(2n) + alpha times the sum of absolute coefficients. SSE is the sum of squared errors and n is the number of fitting rows. The alpha values have different meanings.')
    p('Checks cover the Ridge calculations, the Lasso solution when selected, split separation, saved-model predictions, CSV columns, row counts, and row order. The original datasets are unchanged.')
    h('GitHub repository')
    safe = escape(repo_url, {'"': '&quot;'})
    p(f'<link href="{safe}" color="#126782">{safe}</link>')
    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor('#cfdde4'))
        canvas.line(54, 44, 541, 44)
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#506370'))
        canvas.drawString(54, 30, 'BT2024260 | Assignment 1: Polynomial Regression')
        canvas.drawRightString(541, 30, str(doc.page))
    doc = SimpleDocTemplate(str(output), pagesize=(595.28, 841.89), leftMargin=54, rightMargin=54, topMargin=44, bottomMargin=58, title='Polynomial Regression - BT2024260', author='BT2024260')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f'Wrote {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-url', default='https://github.com/Lohith600/ML_ASSIGNMNET')
    main(parser.parse_args().repo_url)
