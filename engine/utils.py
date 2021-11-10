import base64
import copy
import io
import os
import re
from datetime import datetime
from datetime import timedelta
import math
import sys
from operator import attrgetter
from timeit import default_timer as timer

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import global_control
import pandas as pd
import logging as log
import pickle as pickle

# DO NOT REMOVE - those imports are used by exec to convert TPOT pipeline to Pipeline-steps format
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, \
    PowerTransformer, QuantileTransformer, Normalizer, Binarizer, scale, KernelCenterer
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.svm import LinearSVC, NuSVC
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier, \
    Perceptron
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import FastICA, PCA
from sklearn.ensemble import ExtraTreesClassifier
from tpot.builtins import StackingEstimator, ZeroCount, CombineDFs, auto_select_categorical_features, \
    _transform_selected, CategoricalSelector, ContinuousSelector, FeatureSetSelector, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import SelectPercentile
from sklearn.kernel_approximation import RBFSampler
from sklearn.kernel_approximation import Nystroem
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_selection._univariate_selection import f_classif

global pipe


def adjust_dataset(dataframe):
    if 'Class' in dataframe.columns:
        dataframe.columns = dataframe.columns.str.replace('Class', 'class')
    # If there is no class column, assume that the last column contains decision attribiute
    if 'class' not in dataframe.columns:
        dataframe.columns = [*dataframe.columns[:-1], 'class']

    for x in dataframe.keys():
        if dataframe.dtypes[x] not in ['int64', 'float64', 'bool']:
            mapping_data = {}
            # replace each value with a different number (including NaN)
            unique_values = dataframe[x].unique()
            # unique_values = dataframe[x].fillna(method='ffill').unique()
            for i, v in enumerate(unique_values):
                mapping_data[v] = i
            dataframe[x] = dataframe[x].map(mapping_data)
    return dataframe


# HIGHEST_PROTOCOL = smaller files.
# maximum file_name size of pickle = 2GB. "I think the 2GB limit was removed with protocol=4"
def save_results_gmc(population, subfolder):
    os.makedirs(os.path.join('results', subfolder), exist_ok=True)
    with open(os.path.join('results', subfolder, 'population.pickle'), 'wb') as handle:
        pickle.dump(population, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # best_individual = get_best_from_list(population.individuals)
    log.debug(f'Passed dataset: {population.dataset_name=}')
    print(f"\n{global_control.status['pipeline']=}\n{global_control.status['best_score']}")
    summary = {'dataset_name': population.dataset_name, 'cv': population.history[0][0].validation_method,
               'tool': 'GMC',
               'score': f"CV:{global_control.status['best_score']}\nTest:{global_control.status['best_test_score']}",
               'pipe_string': global_control.status['pipeline'], 'train_set_rows': population.dataset_rows,
               'train_set_attributes': population.dataset_attributes,
               'decision_classes': population.dataset_classes, 'random_state': population.random_state,
               'time': global_control.status['time'], 'plot': global_control.status['plot']}
    with open(os.path.join('results', subfolder, 'best_pipeline_summary.pickle'), 'wb') as handle:
        pickle.dump(summary, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.makedirs(os.path.join('results', subfolder), exist_ok=True)
    with open(os.path.join('results', subfolder, 'best_pipeline.pickle'), 'wb') as handle:
        pickle.dump(global_control.status['pipeline'], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join('results', subfolder, 'generation_plot.png'), 'wb') as handle:
        handle.write(global_control.status['plot_png'].getvalue())


def save_results_tpot(subfolder):
    os.makedirs(os.path.join('results', subfolder), exist_ok=True)
    global_control.tpot.export(os.path.join('results', subfolder, 'best_pipeline.py'))

    # with file name tpot will safe to file, without it, it will return string
    to_write = global_control.tpot.export('')
    to_write = [to_write.replace("\r\n", "").replace("  ", "").replace("\n", "") for x in to_write]

    # if random state was set the pipeline is between "exported_pipeline = " and "#"
    try:
        make_pipeline_string = re.search('exported_pipeline = (.+?)#', to_write[0]).group(1)
    except AttributeError:
        # if random_state=None pipeline is between "exported_pipeline = " and "exported_pipeline"
        make_pipeline_string = re.search('exported_pipeline = (.+?)exported_pipeline', to_write[0]).group(1)

    exec('global pipe; pipe=' + make_pipeline_string)

    summary = {'dataset_name': global_control.tpot_status['dataset_name'], 'cv': global_control.tpot_status['cv'],
               'tool': 'TPOT',
               'score': f"CV:{global_control.tpot_status['best_score']}\nTest:{global_control.tpot_status['best_test_score']}",
               'pipe_string': str(pipe),
               'train_set_rows': global_control.tpot_status['train_set_rows'],
               'train_set_attributes': global_control.tpot_status['train_set_attributes'],
               'decision_classes': global_control.tpot_status['decision_classes'],
               'random_state': global_control.tpot_status['random_state'],
               'time': global_control.tpot_status['time'], 'plot': global_control.tpot_status['plot']}
    with open(os.path.join('results', subfolder, 'best_pipeline_summary.pickle'), 'wb') as handle:
        pickle.dump(summary, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.makedirs(os.path.join('results', subfolder), exist_ok=True)
    try:
        with open(os.path.join('results', subfolder, 'best_pipeline.pickle'), 'wb') as handle:
            pickle.dump(pipe, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except:
        with open(os.path.join('results', subfolder, 'best_pipeline.pickle'), 'wb') as handle:
            pickle.dump(str(pipe), handle, protocol=pickle.HIGHEST_PROTOCOL)
        global_control.tpot_status['status'] += '<br/><date>' + datetime.now().strftime(
            "%d.%m.%Y|%H-%M-%S") + f':</date> WARNING! There was an error in saving pipeline. It was saves as string'
    with open(os.path.join('results', subfolder, 'generation_plot.png'), 'wb') as handle:
        handle.write(global_control.tpot_status['plot_png'].getvalue())


def save_genome(gen, subfolder):
    os.makedirs(os.path.join('results', subfolder), exist_ok=True)
    with open(os.path.join('results', subfolder, 'genome.pickle'), 'wb') as handle:
        pickle.dump(gen, handle, protocol=pickle.HIGHEST_PROTOCOL)


def generate_param_grid(transformers: []):
    param_grid = dict()
    for x in transformers:
        param_name = type(x).__name__.lower() + '__'
        params = x.get_params()
        for key in params:
            param_name += key
            param_grid[param_name] = [params[key]]
            param_name = type(x).__name__.lower() + '__'
    return param_grid


def get_best_from_list(indv=[]):
    if len(indv) == 0:
        log.warning("No individuals found")
        return 0
    pop2 = []
    best = None
    for x in indv:
        if x.validation_method is None or x.score is None:
            log.warning("Individual ignored - not tested yet")
            continue
        pop2.append(x)

    if len(pop2) != 0:
        best = max(pop2, key=attrgetter('score'))
    # best = max(indv, key=attrgetter('score'))
    log.info(f'Returning best individual, pipe:{best.pipeline} score:{best.score}')
    return best


# def get_best_from_list_tmp(indv=[]):
#     return max(indv, key=attrgetter('score'))


def average_score(individuals: []) -> float:
    pop2 = []
    for x in individuals:
        if x.validation_method is None or x.score is None:
            log.warning("Individual ignored - not tested yet")
            continue
        pop2.append(x)
    return sum(i.score for i in pop2) / len(pop2)


def average_time(individuals: []):
    deltas = []
    for x in individuals:
        if x.validation_method is None or x.score is None or x.validation_time is None:
            log.warning("Individual ignored - not tested yet")
            continue
        deltas.append(x.validation_time)
    if len(deltas) > 0:
        return sum(deltas, timedelta(0)) / len(deltas)
    else:
        return "No validated individuals."


def update_plot(population):
    # old, slow version
    # bests = []
    # avgs = []
    # for i, x in enumerate(population.history):
    #     bests.append(get_best_from_list(x).score)
    #     avgs.append(average_score(x))

    global_control.status['scores'].append(get_best_from_list(population.individuals).score)
    global_control.status['avgs'].append(average_score(population.individuals))
    bests = global_control.status['scores']
    avgs = global_control.status['avgs']

    series = pd.Series(bests)
    series.plot()
    fig = Figure()

    fig.set_dpi(120.)
    fig.set_figheight(5.4)
    fig.set_figwidth(6.4)

    axis = fig.add_subplot(1, 1, 1)
    axis.set_title(global_control.status['title'] + f" Time:{global_control.status['time']}", wrap=True, y=1.01)

    axis.set_xlabel("Generation")
    axis.set_ylabel("Score")
    axis.grid()
    axis.plot(series)
    series.update(avgs)
    axis.plot(series)
    fig.legend(['Best', 'Average'], loc='lower right',
               ncol=2, fancybox=True, shadow=True)
    """  https://gitlab.com/-/snippets/1924163 """
    # Convert plot to PNG image
    global_control.status['plot_png'] = io.BytesIO()
    FigureCanvas(fig).print_png(global_control.status['plot_png'])
    # Encode PNG image to base64 string
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(global_control.status['plot_png'].getvalue()).decode('utf8')

    global_control.status['plot'] = pngImageB64String


"""
    For better performance we may track how many scores (one each generation) was captured, unfortunately with
        periodical refreshing we may skip some generations!
    Alternative version could take advantage from the fact that the original tpot_individuals if enumerated will 
        give individuals sorted by generations
        ---------------------------------------------------------------------------------------------------------
        TPOT.evaluated_individuals_ returns various number of individuals (I think it's throwing out worst
        individuals) otherwise it would be possible to check if first/last generation is fully tested gather data
        for update
        TPOT is probably using some form of elitism or maybe even shuffles constantly among historically best.
"""


def update_plot_tpot(tpot_individuals):
    bests = []
    score_key = 'internal_cv_score'
    generation_key = 'generation'
    tpot_extracted_stats = []

    last_gen_number = 0

    # tpot_individuals.copy() if you get RuntimeError: dictionary changed size during iteration
    for key in tpot_individuals:
        # Create table [generation_number, score, pipeline]
        tpot_extracted_stats.append([tpot_individuals[key][generation_key], tpot_individuals[key][score_key], key])
        if tpot_individuals[key][generation_key] > last_gen_number:
            last_gen_number = tpot_individuals[key][generation_key]

    tmp_scores = []
    avgs = []
    if last_gen_number == 0:
        tmp_generation = []
        generation_ids = [i for i in range(0, len(tpot_extracted_stats)) if tpot_extracted_stats[i][0] == 0]
        for i in generation_ids:
            tmp_generation.append(tpot_extracted_stats[i])
            tmp_scores.append(tpot_extracted_stats[i][1])
        tmp_generation.sort(key=lambda i: i[1], reverse=True)
        try:
            bests.append(tmp_generation[0][1])
            avgs.append(np.average(tmp_scores))
        except IndexError:
            log.info('Too fast for TPOT')
            return
    else:
        for y in range(last_gen_number):
            tmp_generation = []
            generation_ids = [i for i in range(0, len(tpot_extracted_stats)) if tpot_extracted_stats[i][0] == y]
            for x in generation_ids:
                tmp_generation.append(tpot_extracted_stats[x])
                # protect from -inf
                if tpot_extracted_stats[x][1] > 0.0:
                    tmp_scores.append(tpot_extracted_stats[x][1])
                log.debug(f'{tpot_extracted_stats[x][1]=}')
            tmp_generation.sort(key=lambda i: i[1], reverse=True)
            try:
                bests.append(tmp_generation[0][1])
                avgs.append(np.average(tmp_scores))
                log.debug(f'{tmp_scores=}')
                tmp_scores = []
            except IndexError:
                log.info('Too fast for TPOT')
                return
    log.debug(f'TPOT last population size:{len(tmp_generation)}')
    if global_control.tpot_status['best_score'] < max(bests):
        global_control.tpot_status['best_score'] = max(bests)
        global_control.tpot_status['pipeline'] = tmp_generation[0][2]

    global_control.tpot_status['avgs'] = avgs
    global_control.tpot_status['bests'] = bests

    if global_control.tpot_status['last'] < last_gen_number:
        global_control.tpot_status['last'] = last_gen_number
        global_control.tpot_status['status'] += '<br/><date>' + datetime.now().strftime(
            "%d.%m.%Y|%H-%M-%S") + f':</date> Generation {last_gen_number - 1} scored. Testing generation {last_gen_number}'

    draw_plot_tpot()

    if global_control.tpot.max_time_mins != 1:
        if global_control.stop_tpot:
            global_control.tpot.max_time_mins = 1


def draw_plot_tpot():
    plot_index = []

    series = pd.Series(global_control.tpot_status['bests'])
    series.plot()
    fig = Figure()

    fig.set_dpi(120.)
    fig.set_figheight(5.4)
    fig.set_figwidth(6.4)

    axis = fig.add_subplot(1, 1, 1)
    axis.set_title(global_control.tpot_status[
                       'title'] + f" Time:{datetime.now() - global_control.tpot_status['start_time']}", wrap=True,
                   y=1.01)
    axis.set_xlabel("Generation")
    axis.set_ylabel("Score")
    axis.grid()
    axis.plot(series)
    series.update(global_control.tpot_status['avgs'])
    axis.plot(series)
    fig.legend(['Best', 'Average'], loc='lower right',
               ncol=2, fancybox=True, shadow=True)
    # Convert plot to PNG image
    global_control.tpot_status['plot_png'] = io.BytesIO()
    FigureCanvas(fig).print_png(global_control.tpot_status['plot_png'])
    # Encode PNG image to base64 string
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(global_control.tpot_status['plot_png'].getvalue()).decode('utf8')
    global_control.tpot_status['plot'] = pngImageB64String


def update_progress(progress: float, start: timer):
    status = '| Best score: ' + str(
        global_control.status['best_score']) + '| Pipeline:' + str(
        global_control.status['pipeline']) + '| Time: ' + str(
        datetime.now() - start)

    global_control.status['time'] = datetime.now() - start
    # rounding up for less glitchy of progress bar
    progress_bar = '{0}{1}{2} {3}%'.format(
        '▓' * math.ceil(progress), '▒', '░' * math.ceil((100 - progress)), "%.2f" % progress)
    global_control.status['progress_bar'] = progress_bar
    if progress > 10 and int(progress) % 3 == 0:
        sys.stderr.write('\n' + status + '\n')
    else:
        sys.stderr.write('\r' + progress_bar)


# def update_progress_nohof(progress: float, population: [], start: timer):
#
#     status = '| Best score: ' + str(
#         global_control.status['best_score']) + '| Pipeline:' + str(
#         global_control.status['pipeline']) + '| Time: ' + str(
#         datetime.now() - start)
#
#     global_control.status['time'] = datetime.now() - start
#
#
#     progress_bar = '{0}{1}{2} {3}%'.format(
#         '▓' * math.ceil(progress), '▒', '░' * math.ceil((100 - progress)), "%.2f" % progress)
#     global_control.status['progress_bar'] = progress_bar
#     if progress > 10 and int(progress) % 3 == 0:
#         sys.stderr.write('\n' + status + '\n')
#     else:
#         sys.stderr.write('\r' + progress_bar)


def update_status(status: str):
    global_control.status['status'] += '<br/><date>' + datetime.now().strftime(
        "%d.%m.%Y|%H-%M-%S") + ':</date> ' + status


# returns sorted and updated hall of fame and best individuals
def update_hall_of_fame(hall_of_fame: list, individuals: list, elitism: int):
    all_ind = hall_of_fame + individuals
    # try:
    #     hall_of_fame = get_n_best(elitism, all_ind)
    # except TypeError:
    #     print('Critical error, unable to update hof')
    hall_of_fame = get_n_best(elitism, all_ind)
    return hall_of_fame


def get_n_best(n: int, individuals: list):
    individuals.sort(key=lambda x: x.score, reverse=True)

    bests = []
    for i in range(n):
        bests.append(individuals[i])
    return bests


def display_individuals(population):
    print('\n============================== POPULATION\'S CLASSIFIERS  ====================================')
    for x in population.individuals:
        print(x.score)


def keys(item):
    return [i for i in item.__dict__.keys()]


def values(item):
    return [i for i in item.__dict__.values()]


def avgScore(scores):
    suma = 0
    for score in scores:
        suma += score
    return suma / len(scores)
