import base64
import io
import os
import threading
import traceback
from datetime import datetime
import math
import sys
# from functools import cache
from operator import attrgetter
from timeit import default_timer as timer

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


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


import pickle as pickle


# HIGHEST_PROTOCOL = smaller files.
# maximum file_name size of pickle = 2GB. "I think the 2GB limit was removed with protocol=4"
def save_results_gmc(population, pipe, subfolder):
    os.makedirs(os.path.join('results', subfolder), exist_ok=True)
    with open(os.path.join('results', subfolder, 'population.pickle'), 'wb') as handle:
        pickle.dump(population, handle, protocol=pickle.HIGHEST_PROTOCOL)
    best_individual = get_best_from_list(population.individuals)
    print(f'Prekazana {population.dataset_name=}')
    summary = {'dataset_name': population.dataset_name, 'cv': best_individual.validation_method,
               'tool': 'GMC', 'score': best_individual.score,
               'pipe_string': best_individual.pipeline_string, 'train_set_rows': population.dataset_rows,
               'train_set_attributes': population.dataset_attributes,
               'decision_classes': population.dataset_classes, 'random_state': population.random_state}
    with open(os.path.join('results', subfolder, 'best_pipeline_summary.pickle'), 'wb') as handle:
        pickle.dump(summary, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.makedirs(os.path.join('results', subfolder), exist_ok=True)
    with open(os.path.join('results', subfolder, 'best_pipeline.pickle'), 'wb') as handle:
        pickle.dump(pipe, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_results_tpot(subfolder):
    os.makedirs(os.path.join('results', subfolder), exist_ok=True)
    summary = {'dataset_name': global_control.tpot_status['dataset_name'], 'cv': global_control.tpot_status['cv'],
               'tool': 'TPOT', 'score': global_control.tpot_status['best_score'],
               'pipe_string': str(global_control.tpot_status["pipeline"]),
               'train_set_rows': global_control.tpot_status['train_set_rows'],
               'train_set_attributes': global_control.tpot_status['train_set_attributes'],
               'decision_classes': global_control.tpot_status['decision_classes'],
               'random_state': global_control.tpot_status['random_state']}
    with open(os.path.join('results', subfolder, 'best_pipeline_summary.pickle'), 'wb') as handle:
        pickle.dump(summary, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.makedirs(os.path.join('results', subfolder), exist_ok=True)
    with open(os.path.join('results', subfolder, 'best_pipeline.pickle'), 'wb') as handle:
        pickle.dump(global_control.tpot_status["pipeline"], handle, protocol=pickle.HIGHEST_PROTOCOL)

    global_control.tpot.export(os.path.join('results', subfolder, 'best_pipeline.py'))


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


import global_control
import pandas as pd
from matplotlib import pyplot as plt
import logging as log


def get_best_from_list(indv=[]):
    if len(indv) == 0:
        log.warning("No individuals found")
        return 0
    pop2 = []
    best = None
    for x in indv:
        if x.validation_method is None:
            log.warning("Individual ignored - not tested yet")
            continue
        pop2.append(x)
    if len(pop2) != 0:
        best = max(pop2, key=attrgetter('score'))
    return best


"""   plt.title('Tytul')
    https://gitlab.com/-/snippets/1924163"""


def average_score(individuals: []) -> float:
    return sum(i.score for i in individuals) / len(individuals)


def update_plot(population):
    besty = []
    avgs = []

    # todo - optimize, global besty and avgs
    for i, x in enumerate(population.history):
        besty.append(get_best_from_list(x).score)
        avgs.append(average_score(x))

    series = pd.Series(besty)
    series.plot()
    fig = Figure()

    fig.set_dpi(120.)
    fig.set_figheight(4.0)
    fig.set_figwidth(6.4)

    axis = fig.add_subplot(1, 1, 1)
    axis.set_title('GmC')
    axis.set_xlabel("Generation")
    axis.set_ylabel("Score")
    axis.grid()
    axis.plot(series)
    series.update(avgs)
    axis.plot(series)
    fig.legend(['Best', 'Average'], loc='upper right',
               ncol=2, fancybox=True, shadow=True)

    # Convert plot to PNG image
    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)
    # Encode PNG image to base64 string
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')

    global_control.status['plot'] = pngImageB64String
    # return series
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
                print(f'{tpot_extracted_stats[x][1]=}')
            tmp_generation.sort(key=lambda i: i[1], reverse=True)
            try:
                bests.append(tmp_generation[0][1])
                avgs.append(np.average(tmp_scores))
                print(f'{tmp_scores=}')
                tmp_scores = []
            except IndexError:
                log.info('Too fast for TPOT')
                return
    print(f'TPOT last population size:{len(tmp_generation)}')
    if global_control.tpot_status['best_score'] < max(bests):
        global_control.tpot_status['best_score'] = max(bests)
        global_control.tpot_status['pipeline'] = tmp_generation[0][2]

    global_control.tpot_status['avgs'] = avgs
    global_control.tpot_status['bests'] = bests

    draw_plot_tpot()

    if global_control.stop_tpot:
        global_control.tpot.max_time_mins = 1


def draw_plot_tpot():
    plot_index = []

    series = pd.Series(global_control.tpot_status['bests'])
    series.plot()
    fig = Figure()

    fig.set_dpi(120.)
    fig.set_figheight(4.0)
    fig.set_figwidth(6.4)

    axis = fig.add_subplot(1, 1, 1)
    axis.set_title("TPOT optimization")
    axis.set_xlabel("Generation")
    axis.set_ylabel("Score")
    axis.grid()
    axis.plot(series)
    series.update(global_control.tpot_status['avgs'])
    axis.plot(series)
    fig.legend(['Best', 'Average'], loc='upper right',
               ncol=2, fancybox=True, shadow=True)
    # Convert plot to PNG image
    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)

    # Encode PNG image to base64 string
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')

    global_control.tpot_status['plot'] = pngImageB64String


def update_progress(progress: float, hall_of_fame: [], start: timer):
    # after completing 10% of the task - display short summary
    # print("Progress", progress)
    # print('int(progress) % 10: '+str(int(progress) % 3))
    # globals.initialize_status()
    # global status
    if hall_of_fame:
        status = '| Best score: ' + str(
            hall_of_fame[0].score) + '| Pipeline:' + str(
            hall_of_fame[0].pipeline) + '| Time: ' + str(
            datetime.now() - start)
        global_control.status['best_score'] = hall_of_fame[0].score
        global_control.status['pipeline'] = hall_of_fame[0].pipeline
        global_control.status['time'] = datetime.now() - start
        # rounding up for less glitchy of progress bar
    progress_bar = '{0}{1}{2} {3}%'.format(
        '▓' * math.ceil(progress), '▒', '░' * math.ceil((100 - progress)), "%.2f" % progress)
    global_control.status['progress_bar'] = progress_bar
    if progress > 10 and int(progress) % 3 == 0:
        sys.stderr.write('\n' + status + '\n')
    else:
        sys.stderr.write('\r' + progress_bar)


def update_progress_nohof(progress: float, population: [], start: timer):
    if population:
        best = get_best_from_list(population.individuals)
        status = '| Best score: ' + str(
            best.score) + '| Pipeline:' + str(
            best.pipeline) + '| Time: ' + str(
            datetime.now() - start)
        if global_control.status['best_score'] < best.score:
            global_control.status['best_score'] = best.score
            print(f'{global_control.status["best_score"]= }')
            global_control.status['pipeline'] = best.pipeline
        global_control.status['time'] = datetime.now() - start

    progress_bar = '{0}{1}{2} {3}%'.format(
        '▓' * math.ceil(progress), '▒', '░' * math.ceil((100 - progress)), "%.2f" % progress)
    global_control.status['progress_bar'] = progress_bar
    if progress > 10 and int(progress) % 3 == 0:
        sys.stderr.write('\n' + status + '\n')
    else:
        sys.stderr.write('\r' + progress_bar)


def update_status(status: str):
    global_control.status['status'] += '<br/><date>' + datetime.now().strftime(
        "%d.%m.%Y|%H-%M-%S") + ':</date> ' + status


# returns sorted and updated hall of fame and best individuals
def update_hall_of_fame(hall_of_fame: list, individuals: list, elitism: int):
    all_ind = hall_of_fame + individuals
    hall_of_fame = get_n_best(elitism, all_ind)
    # print("       wsićko: ", [x.score for x in all_ind])
    # print('--- osobniki hof: ', [x.score for x in hall_of_fame])
    return hall_of_fame


def get_n_best(n: int, individuals: list):
    # print('Nieposortowane osobniki: ', [x.score for x in individuals])
    # TypeError: '<' not supported between instances of 'NoneType' and 'NoneType'
    # jakim cudem?
    # a no takim, że dodajemy fresh blood i nie testujemy
    individuals.sort(key=lambda x: x.score, reverse=True)
    # print('Posortowane osobniki   : ', [x.score for x in individuals])
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
