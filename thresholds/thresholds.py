import sklearn.metrics as skm
import numpy as np
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
import pandas as pd
import ipywidgets
from mpl_toolkits.mplot3d import Axes3D


def calculate_confusion_matrix(df, threshold, label_col, prob_col='Prediction', pos_label='Y', neg_label='N'):
    temp = pd.DataFrame()
    temp['Predicted Label'] = np.where(df[prob_col] >= threshold, pos_label, neg_label)
    temp['True Label'] = df[label_col]

    # Note labels are critical to determin order
    return skm.confusion_matrix(temp['True Label'], temp['Predicted Label'], labels=[pos_label, neg_label])


def plot_distributions(df, actual_col, prediction_col, pos_label='Y', neg_label='N', bins=10, threshold=None):
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.xlim(0, 1)
    sns.set(rc={'figure.figsize':(8,3)})

    positives = df.loc[df[actual_col] == pos_label]
    negatives = df.loc[df[actual_col] == neg_label]

    pos_label = '{}  n={}'.format(pos_label, len(positives))
    neg_label = '{}  n={}'.format(neg_label, len(negatives))
    ax = sns.distplot(positives[prediction_col], kde=False, label=pos_label, bins=bins, color='g')
    ax = sns.distplot(negatives[prediction_col], kde=False, label=neg_label, bins=bins, color='b')
    ax.set_title('Prediction Distributions of {} for {}.\nn={}'.format(prediction_col, actual_col, len(df)))
    ax.legend()
    if threshold:
        plt.axvline(x=threshold, color='r')
    plt.show()


def true_positives(threshold, df, probabilty_col, label_col):
    pred_pos = df[df[probabilty_col] >= threshold]
    true_pos = pred_pos[pred_pos[label_col] == 'Y']

    return true_pos


def true_negatives(threshold, df, probabilty_col, label_col):
    pred_neg = df[df[probabilty_col] < threshold]
    true_neg = pred_neg[pred_neg[label_col] == 'N']

    return true_neg


def false_positives(threshold, df, probabilty_col, label_col):
    pred_pos = df[df[probabilty_col] >= threshold]
    false_pos = pred_pos[pred_pos[label_col] == 'N']

    return false_pos


def false_negatives(threshold, df, probabilty_col, label_col):
    pred_neg = df[df[probabilty_col] < threshold]
    false_neg = pred_neg[pred_neg[label_col] == 'Y']

    return false_neg


def dynamic_cm(df, threshold, probabilty_col, label_col, verbose=True):
    tp = len(true_positives(threshold, df, probabilty_col, label_col))
    fn = len(false_negatives(threshold, df, probabilty_col, label_col))
    tn = len(true_negatives(threshold, df, probabilty_col, label_col))
    fp = len(false_positives(threshold, df, probabilty_col, label_col))
    if verbose:
        print_cm(tp, fp, tn, fn)

    return tp, fp, tn, fn


def print_cm(tp, fp, tn, fn):
    print("""|        | Predicted Y | Predicted N |
|--------|-------------|-------------|
| True Y |{}|{}|
| True N |{}|{}|""".format(
        str(tp).center(13),
        str(fn).center(13),
        str(fp).center(13),
        str(tn).center(13),))


def above_threshold(df, threshold, probabilty_col):
    return df[df[probabilty_col] >= threshold]


def positive_predictive_value(tp, fp):
    """Positive Predictive Value or Precision: the proportion of positive cases that were correctly identified."""
    return tp / (tp + fp)


def negative_predictive_value(tn, fn):
    """Negative Predictive Value: the proportion of negative cases that were correctly identified."""
    return tn / (tn + fn)


def sensitivity(tp, fn):
    """Sensitivity or Recall: the proportion of actual positive cases which are correctly identified."""
    return tp / (tp + fn)


def specificity(tn, fp):
    """Specificity: the proportion of actual negative cases which are correctly identified."""
    return tn / (tn + fp)


def display_stats(tp, fp, tn, fn):
    """Display interesting statistics."""
    stats = {
        'PPV/Precision': positive_predictive_value(tp, fp),
        'NPV': negative_predictive_value(tn, fn),
        'Sensitivity/Recall': sensitivity(tp, fn),
        'Specificity': specificity(tn, fp)
    }

    for title, stat in stats.items():
        print('{}: {:.3f}'.format(title, stat))


def threshold_picker(
        population,
        threshold_hri,
        threshold_human,
        pop_size,
        hri_label,
        hri_prob,
        human_label,
        human_prob,
        verbose=False):
    """Graph and calculate from the holdout set."""
    df = load_population(population)
    # cm_hri = calculate_confusion_matrix(df, threshold_hri, label_col=hri_label, prob_col='HRi_prediction')
    # normalized_cm_hri = cm_hri.astype('float') / cm_hri.sum(axis=1)[:, np.newaxis]
    # expected_cm_hri = normalized_cm_hri * pop_size
    # expected_cm_hri = expected_cm_hri.astype('int')

    # if verbose:
    #     print('HRi')
    #     print(cm_hri, '\n')
    #     print(normalized_cm_hri, '\n')
    #     print(expected_cm_hri, '\n')

    hri_df = above_threshold(df, threshold_hri, hri_prob)
    hri_percent = len(hri_df) / len(df)
    hri_estimate = hri_percent * pop_size

    plot_distributions(df, 'HR_Identified', 'HRi_prediction', threshold=threshold_hri, bins=20)
    print('Holdout: {} patients with HRi scores >= {} ({:.1%} selected)'.format(len(hri_df), threshold_hri, hri_percent))

    tp1, fp1, tn1, fn1 = dynamic_cm(df, threshold_hri, hri_prob, hri_label)
    display_stats(tp1, fp1, tn1, fn1)
    print('\nEstimated: {:.0f} patients with HRi scores >= {}'.format(hri_estimate, threshold_hri))

    plot_distributions(hri_df, 'ICMP_Selected', 'Human_prediction', threshold=threshold_human, bins=20)
    human_df = above_threshold(hri_df, threshold_human, human_prob)

    human_percent = len(human_df) / len(hri_df)
    human_estimate = hri_percent * human_percent * pop_size

    print('Holdout: {} patients with Human scores >= {} ({:.1%} selected)'.format(len(human_df), threshold_human, human_percent))

    tp2, fp2, tn2, fn2 = dynamic_cm(hri_df, threshold_human, human_prob, human_label)
    display_stats(tp2, fp2, tn2, fn2)
    print('\nEstimated: {:.0f} patients with Human scores >= {}'.format(human_estimate, threshold_human))


    #     print_confusion_matrix(expected_cm_hri, class_names=['Y', 'N'])
    #     plot_distributions(human_df, 'ICMP_Selected', 'Prediction_Human', threshold=threshold_human, bins=20)
    #     confusion_matrix_plot(confusion_matrix=cm_hri, classes=['Y', 'N'])

    tp3 = len(true_positives(threshold_human, human_df, human_prob, human_label))
    fp3 = len(false_positives(threshold_human, human_df, human_prob, human_label))
    # This is subtle = you need tn1 (HRi = N) plus tn2 (Human = N) plus the sliver of (HRi = Y / Human = N) below the HRi threshold
    tn3 = tn1 + tn2 + len(df.loc[(df.HRi_prediction < threshold_hri) & (df.HR_Identified == 'Y') & (df.ICMP_Selected == 'N')])
    # This is subtle - you need fn2 plus the ICMP Y that were below the HRi threshold
    fn3 = fn2 + len(df.loc[(df.HRi_prediction < threshold_hri) & (df.ICMP_Selected == 'Y')])

    print('\n')
    print_cm(tp3, fp3, tn3, fn3)
    display_stats(tp3, fp3, tn3, fn3)
    final_percent = hri_percent * human_percent
    final_estimate = pop_size * hri_percent * human_percent
    print('\nEstimated overall (HRi + Human): {:.0f} ({:.1%}) patients will be above both thresholds'.format(final_estimate, final_percent))


    draw_sankey(pop_size, hri_estimate, human_estimate)


def draw_sankey(pop_size, hri_estimate, human_estimate):
    from ipysankeywidget import SankeyWidget
    from ipywidgets import Layout

    layout = Layout(width="800", height="300")

    n_high_risk = hri_estimate
    n_low_risk = pop_size - n_high_risk
    n_selected = human_estimate
    n_not_selected = n_high_risk - n_selected

    links = [
        {'source': 'Population', 'target': 'High Risk', 'value': n_high_risk, 'color': 'rgb(181, 57, 65)'},
        {'source': 'Population', 'target': 'Low Risk', 'value': n_low_risk, 'color': 'rgb(192, 172, 97)'},
        {'source': 'High Risk', 'target': 'Not Selected', 'value': n_not_selected, 'color': 'rgb(60, 92, 160)'},
        {'source': 'High Risk', 'target': 'Selected', 'value': n_selected, 'color': 'rgb(71, 155, 85)'},
    ]

    w = SankeyWidget(links=links, layout=layout, margins=dict(top=0, bottom=0, left=100, right=100))
    display(w)


def _attempt_both_file_locations(filename):
    """Simplify deploy by checking both known data locations."""
    try:
        df = pd.read_csv('../../' + filename)
    except FileNotFoundError:
        df = pd.read_csv(filename)
    return df


def load_population(pop):
    """Load a population."""
    if pop == 'ACO':
        df = _attempt_both_file_locations('2018-02-07_ACO_holdout_combined_predictions_scrubbed.csv')
    elif pop == 'ACO + opt-ins':
        df = _attempt_both_file_locations('2018-02-14_ACO_holdout_optins_enhanced_combined_predictions_scrubbed.csv')
    elif pop == 'Commercial + Medicaid':
        df = _attempt_both_file_locations('2018-02-08_Commercial_Medicaid_holdout_combined_predictions_scrubbed.csv')
    elif pop == 'Commercial + Medicaid + opt-ins':
        df = _attempt_both_file_locations('2018-02-22_Commercial_Medicaid_holdout_combined_predictions_scrubbed.csv')
    elif pop == 'Medicaid + opt-ins':
        df = _attempt_both_file_locations('2018-02-26_medicaid_holdout_optin_enhanced_combined_predictions_scrubbed.csv')

    return df


def threshold_tool():
    button_style = ipywidgets.Layout(width='600px')

    return ipywidgets.interact(
        threshold_picker,
        population=ipywidgets.RadioButtons(options=['ACO', 'ACO + opt-ins', 'Commercial + Medicaid', 'Commercial + Medicaid + opt-ins', 'Medicaid + opt-ins'], description='Group', layout=button_style),
        threshold_hri=ipywidgets.FloatSlider(min=0.001, max=1, step=.01, value=0.5, description='HRi', continuous_update=False, layout=button_style),
        threshold_human=ipywidgets.FloatSlider(min=0.001, max=1, step=.01, value=0.5, description='Human', continuous_update=False, layout=button_style),
        pop_size=ipywidgets.IntSlider(min=100, max=300000, step=100, value=10000, description='Population', continuous_update=False, layout=button_style),
        hri_label=ipywidgets.fixed('HR_Identified'),
        hri_prob=ipywidgets.fixed('HRi_prediction'),
        human_label=ipywidgets.fixed('ICMP_Selected'),
        human_prob=ipywidgets.fixed('Human_prediction'),
    )
