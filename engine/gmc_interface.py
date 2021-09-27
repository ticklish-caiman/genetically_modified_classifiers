import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score
from tpot import TPOTClassifier

import engine.gmc as pop
import threading
from datetime import datetime
import logging as log

import utils
from utils import adjust_dataset, save_results_gmc, save_results_tpot, save_genome, update_plot_tpot
import global_control

datasets = ["biodeg.csv", "breast-cancer.csv", "diabetes(pima).csv", "ionosphere.csv", "MAGIC.csv", "monks2.csv",
            "sonar.all-data.csv", "tic-tac-toeNum.csv"]
grids = ['GMC-minimal', 'GMC-big', 'GMC-extreme', 'TPOT-ish']
rows_options = ["10 rows", "100 rows", "1000 rows",
                "10000 rows"]
columns_options = ["10 columns", "20 columns", "all columns"]
modes = ["Show original data", "Show adjusted data"]
analysis_modes = ["GMC", "TPOT"]
cached_results = None


def get_csv_data(file, row_limit, column_limit, adjust):
    row_limit = int(row_limit)
    column_limit = int(column_limit)
    try:
        dataset = pd.read_csv('data-CSV/' + file, delimiter=',')
    except FileNotFoundError:
        return '<h1>Please select file_name to analyze.</h1>'

    if adjust:
        dataset = utils.adjust_dataset(dataset)

    data = dataset.head(row_limit)

    if len(data.columns) > column_limit:
        drop_list = []
        for i in range(len(data.columns)):
            drop_list.append(i + (column_limit // 2))
            if i > len(data.columns) - (column_limit + 2):
                break
        data.drop(data.columns[drop_list], axis=1, inplace=True)

    data = data.to_html(classes='mystyle')
    # pretty_html_table delivers presets to pick from, but it adds style information to every row (bad for big data)
    # data = build_table(data, 'blue_dark', font_size='large', text_align='left')
    return data


def run_evolve(x_train, y_train, n_jobs, file_name, population=30, generations=500, validation_method=10, elitism=0,
               random_state=13, selection_type='roulette', crossover_rate=0.1, early_stop=5, pipeline_time_limit=120,
               preselection=None, cv=10, cross_method='average', mutation=0.2, mutation_power=1.0):
    global_control.status['status'] += '<br/><date>' + datetime.now().strftime(
        "%d.%m.%Y|%H-%M-%S") + f':</date> Initializing population ({file_name})'
    global_control.status['best_score'] = 0.0
    pop1 = pop.evolve(population=population, generations=generations, validation_method=validation_method,
                      x_train=x_train, y_train=y_train,
                      elitism=elitism, random_state=random_state, selection_type=selection_type,
                      crossover_rate=crossover_rate, early_stop=early_stop,
                      n_jobs=n_jobs, pipeline_time_limit=pipeline_time_limit, preselection=preselection,
                      dataset_name=file_name, cv=cv, cross_method=cross_method, mutation=mutation,
                      mutation_power=mutation_power)
    subfolder_name = datetime.now().strftime("%Y%m%d-%H_%M_%S")
    if elitism:
        save_results_gmc(pop1, pop1.individuals[0].pipeline, subfolder_name)
        save_genome(pop1.individuals[0].genome, subfolder_name)
    else:
        best = pop.get_best_from_history(pop1.history, validation_method)
        pop_with_all = pop.unpack_history(pop1)
        save_results_gmc(pop_with_all, best.pipeline, subfolder_name)
        save_genome(best.genome, subfolder_name)
    print('Koniec ewolucji GMC')

    return pop1


def run_evolve_custom(file_name, validation_size=0.1, n_jobs=1, population=20, generations=100,
                      elitism=8,
                      random_state=13, selection_type='roulette', crossover_rate=0.1, early_stop=50,
                      pipeline_time_limit=120,
                      preselection=None, cv=10, cross_method='average', mutation=0.2, mutation_power=1.0,
                      grid='GMC-big', fresh_blood=True):
    try:
        dataset = pd.read_csv('data-CSV/' + file_name, delimiter=',')
    except FileNotFoundError:
        return '<h2>Please select file_name to analyze.</h2>'
    dataset = adjust_dataset(dataset)
    features = dataset.drop('class', axis=1).values
    x_train, x_test, y_train, y_test = train_test_split(features, dataset['class'].values, test_size=validation_size,
                                                        random_state=random_state)

    print(f'Tajp {type(population)=}')
    print(f'w run evolve custom{elitism=}')
    global_control.status['status'] += '<br/><date>' + datetime.now().strftime(
        "%d.%m.%Y|%H-%M-%S") + f':</date> Initializing population for {file_name}'
    global_control.status['best_score'] = 0.0
    pop1 = pop.evolve(population=population, generations=generations, validation_method=cv,
                      x_train=x_train, y_train=y_train,
                      elitism=elitism, random_state=random_state, selection_type=selection_type,
                      crossover_rate=crossover_rate, cross_method=cross_method, early_stop=early_stop,
                      n_jobs=n_jobs, pipeline_time_limit=pipeline_time_limit, preselection=preselection,
                      dataset_name=file_name, grid_type=grid, mutation_rate=mutation, mutation_power=mutation_power,
                      fresh_blood_mode=fresh_blood)
    subfolder_name = datetime.now().strftime("%Y%m%d-%H_%M_%S")
    save_results_gmc(pop1, pop1.individuals[0].pipeline, subfolder_name)
    save_genome(pop1.individuals[0].genome, subfolder_name)
    log.info('GMC finished')
    global_control.machine_info['free_threads'] += n_jobs
    return pop1


def run_gmc_thread(file_name, validation_size=0.1, n_jobs=1, population=20, generations=1000,
                   elitism=8,
                   random_state=13, selection_type='roulette', crossover_rate=0.1, early_stop=50,
                   pipeline_time_limit=120,
                   preselection=None, cv=10, cross_method='average', mutation=0.2, mutation_power=1., grid='GMC-big',
                   fresh_blood=True):
    # threading will only use one-thread in pure code, but sklearn will use n_jobs and multithreading
    running_threads = []
    for thread in threading.enumerate():
        running_threads.append(thread.name)
    print(running_threads)
    if isinstance(validation_size, int):
        validation_size = validation_size / 100
    print(f'{validation_size=}')

    if isinstance(crossover_rate, int):
        crossover_rate = crossover_rate / 100
    print(f'{mutation=}')

    if isinstance(mutation, int):
        mutation = mutation / 100
    print(f'{mutation=}')

    if isinstance(mutation_power, int):
        mutation_power = mutation_power / 100
    print(f'{mutation_power=}')

    if isinstance(cross_method, int):
        if cross_method == 0:
            cross_method = 'average'
        if cross_method == 1:
            cross_method = 'single-point'
        if cross_method == 2:
            cross_method = 'uniform'

    if isinstance(selection_type, int):
        if selection_type == 0:
            selection_type = 'roulette'
        if selection_type == 1:
            selection_type = 'tournament5'
        if selection_type == 2:
            selection_type = 'tournament10'
        if selection_type == 3:
            selection_type = 'tournament15'
        if selection_type == 4:
            selection_type = 'tournament20'

    if cv == 101:
        cv = LeaveOneOut()

    global_control.status['status'] += '<br/><date>' + datetime.now().strftime(
        "%d.%m.%Y|%H-%M-%S") + f":</date> Validation part set to {validation_size}."
    global_control.status['status'] += '<br/><date>' + datetime.now().strftime(
        "%d.%m.%Y|%H-%M-%S") + f":</date> Cross-validation: {cv}."
    global_control.status['status'] += '<br/><date>' + datetime.now().strftime(
        "%d.%m.%Y|%H-%M-%S") + f":</date> Population size set to {population}."
    global_control.status['status'] += '<br/><date>' + datetime.now().strftime(
        "%d.%m.%Y|%H-%M-%S") + f":</date> Generations number set to {generations}."
    global_control.status['status'] += '<br/><date>' + datetime.now().strftime(
        "%d.%m.%Y|%H-%M-%S") + f":</date> Elitism set to {elitism}."
    global_control.status['status'] += '<br/><date>' + datetime.now().strftime(
        "%d.%m.%Y|%H-%M-%S") + f":</date> Mutation rate set to {mutation}."
    global_control.status['status'] += '<br/><date>' + datetime.now().strftime(
        "%d.%m.%Y|%H-%M-%S") + f":</date> Mutation power set to {mutation_power}."
    global_control.status['status'] += '<br/><date>' + datetime.now().strftime(
        "%d.%m.%Y|%H-%M-%S") + f":</date> Pipeline time limit set to {pipeline_time_limit}s."
    global_control.status['status'] += '<br/><date>' + datetime.now().strftime(
        "%d.%m.%Y|%H-%M-%S") + f":</date> Early stop: {early_stop}"
    global_control.status['status'] += '<br/><date>' + datetime.now().strftime(
        "%d.%m.%Y|%H-%M-%S") + f":</date> Selection method: {selection_type}."
    global_control.status['status'] += '<br/><date>' + datetime.now().strftime(
        "%d.%m.%Y|%H-%M-%S") + f":</date> Cross-over method: {cross_method}."
    global_control.status['status'] += '<br/><date>' + datetime.now().strftime(
        "%d.%m.%Y|%H-%M-%S") + f":</date> Cross-over chance: {crossover_rate}."
    global_control.status['status'] += '<br/><date>' + datetime.now().strftime(
        "%d.%m.%Y|%H-%M-%S") + f":</date> Random state: {random_state}."
    global_control.status['status'] += '<br/><date>' + datetime.now().strftime(
        "%d.%m.%Y|%H-%M-%S") + f":</date> Preselection: {preselection}."
    global_control.status['status'] += '<br/><date>' + datetime.now().strftime(
        "%d.%m.%Y|%H-%M-%S") + f":</date> Using {n_jobs} logical cores."

    gmc_thread = threading.Thread(target=run_evolve_custom, name="gmc_thread",
                                  args=(file_name, validation_size, n_jobs, population, generations,
                                        elitism,
                                        random_state, selection_type, crossover_rate, early_stop,
                                        pipeline_time_limit,
                                        preselection, cv, cross_method, mutation, mutation_power, grid, fresh_blood))
    if n_jobs <= global_control.machine_info['free_threads']:
        # If we want the app to keep running, even when webapp gets reloaded we should use Process instead of Thread
        # gmc_thread = Process(target=run_evolve, args=(x_train, y_train), name="gmc_process")
        gmc_thread.start()
        global_control.machine_info['free_threads'] -= n_jobs
    else:
        global_control.status['status'] += '<br/><date>' + datetime.now().strftime(
            "%d.%m.%Y|%H-%M-%S") + ":</date> Not enough cores available."
        print(f'Saving thread for later: {gmc_thread=}')
        global_control.queue.append(gmc_thread)


def run_tpot_custom(file_name, validation_size=0.1, n_jobs=1, population=20, generations=100,
                    offspring=20,
                    random_state=13, crossover_rate=0.1, early_stop=50,
                    pipeline_time_limit=120, cv=10, mutation=0.2):
    try:
        dataset = pd.read_csv('data-CSV/' + file_name, delimiter=',')
    except FileNotFoundError:
        return '<h2>Please select file_name to analyze.</h2>'
    dataset = adjust_dataset(dataset)
    features = dataset.drop('class', axis=1).values
    x_train, x_test, y_train, y_test = train_test_split(features, dataset['class'].values, test_size=validation_size,
                                                        random_state=random_state)

    global_control.tpot = TPOTClassifier(cv=cv, generations=generations, verbosity=2, population_size=population,
                                         offspring_size=offspring,
                                         mutation_rate=mutation,
                                         crossover_rate=crossover_rate, early_stop=early_stop, n_jobs=n_jobs,
                                         disable_update_check=True, random_state=random_state,
                                         max_eval_time_mins=pipeline_time_limit
                                         )

    global_control.tpot_status['status'] += '<br/><date>' + datetime.now().strftime(
        "%d.%m.%Y|%H-%M-%S") + f':</date> Initializing population for {file_name}'
    global_control.tpot_status['best_score'] = 0.0
    global_control.tpot.fit(x_train, y_train)
    subfolder_name = datetime.now().strftime("%Y%m%d-%H_%M_%S")

    log.info('Tpot finished')
    global_control.machine_info['free_threads'] += n_jobs


def run_tpot_thread(file_name, validation_size=0.1, n_jobs=1, population=20, offspring=20, generations=1000,
                    random_state=13, crossover_rate=0.1, early_stop=50,
                    pipeline_time_limit=120, cv=10, mutation=0.2):
    running_threads = []
    for thread in threading.enumerate():
        running_threads.append(thread.name)
    print(running_threads)
    if isinstance(validation_size, int):
        validation_size = validation_size / 100
    print(f'{validation_size=}')

    if isinstance(crossover_rate, int):
        crossover_rate = crossover_rate / 100
    print(f'{mutation=}')

    if isinstance(mutation, int):
        mutation = mutation / 100
    print(f'{mutation=}')

    if cv == 101:
        cv = LeaveOneOut()

    global_control.tpot_status['status'] += '<br/><date>' + datetime.now().strftime(
        "%d.%m.%Y|%H-%M-%S") + f":</date> Validation part set to {validation_size}."
    global_control.tpot_status['status'] += '<br/><date>' + datetime.now().strftime(
        "%d.%m.%Y|%H-%M-%S") + f":</date> Cross-validation: {cv}."
    global_control.tpot_status['status'] += '<br/><date>' + datetime.now().strftime(
        "%d.%m.%Y|%H-%M-%S") + f":</date> Population size set to {population}."
    global_control.tpot_status['status'] += '<br/><date>' + datetime.now().strftime(
        "%d.%m.%Y|%H-%M-%S") + f":</date> Generations number set to {generations}."
    global_control.tpot_status['status'] += '<br/><date>' + datetime.now().strftime(
        "%d.%m.%Y|%H-%M-%S") + f":</date> Mutation rate set to {mutation}."
    global_control.tpot_status['status'] += '<br/><date>' + datetime.now().strftime(
        "%d.%m.%Y|%H-%M-%S") + f":</date> Pipeline time limit set to {pipeline_time_limit}s."
    global_control.tpot_status['status'] += '<br/><date>' + datetime.now().strftime(
        "%d.%m.%Y|%H-%M-%S") + f":</date> Early stop: {early_stop}"
    global_control.tpot_status['status'] += '<br/><date>' + datetime.now().strftime(
        "%d.%m.%Y|%H-%M-%S") + f":</date> Cross-over chance: {crossover_rate}."
    global_control.tpot_status['status'] += '<br/><date>' + datetime.now().strftime(
        "%d.%m.%Y|%H-%M-%S") + f":</date> Random state: {random_state}."
    global_control.tpot_status['status'] += '<br/><date>' + datetime.now().strftime(
        "%d.%m.%Y|%H-%M-%S") + f":</date> Using {n_jobs} logical cores."

    global_control.tpot_thread = threading.Thread(target=run_tpot_custom, name="tpot_thread",
                                                  args=(file_name, validation_size, n_jobs, population, generations,
                                                        offspring,
                                                        random_state, crossover_rate, early_stop,
                                                        pipeline_time_limit, cv, mutation))

    if n_jobs <= global_control.machine_info['free_threads']:
        global_control.tpot_thread.start()
        global_control.machine_info['free_threads'] -= n_jobs
        if 'status_update_thread' not in running_threads:
            global_control.status_thread = threading.Thread(name='status_update_thread', target=update_tpot_status)
            global_control.status_thread.start()
    else:
        global_control.status['status'] += '<br/><date>' + datetime.now().strftime(
            "%d.%m.%Y|%H-%M-%S") + ":</date> Not enough cores available."
        print(f'Saving thread for later: {global_control.tpot_thread=}')
        global_control.queue.append(global_control.tpot_thread)


# [status, best_score, pipeline, time, bar, plot]
def run_tpot(file_name, x_train, y_train, n_jobs, cv, random_state):
    global_control.tpot_status['status'] = '<br/><date>' + datetime.now().strftime(
        "%d.%m.%Y|%H-%M-%S") + f':</date> Initializing TPOT ({file_name})'
    global_control.tpot_status['best_score'] = 0.0
    global_control.tpot_status['dataset_name'] = file_name
    global_control.tpot_status['cv'] = cv
    global_control.tpot_status['train_set_rows'] = x_train.shape[0]
    global_control.tpot_status['train_set_attributes'] = x_train.shape[1]
    global_control.tpot_status['decision_classes'] = len(np.unique(y_train))
    global_control.tpot_status['random_state'] = random_state
    global_control.tpot_status['best_score'] = 0.0
    start = datetime.now()
    global_control.init_tpot()
    print(f'{n_jobs=}')
    print(f'{int(n_jobs)=}')
    global_control.tpot = TPOTClassifier(cv=10, generations=50, verbosity=2, population_size=10,
                                         offspring_size=10,
                                         mutation_rate=0.9,
                                         crossover_rate=0.1, early_stop=50, n_jobs=n_jobs,
                                         disable_update_check=True, random_state=13
                                         )
    running_threads = []
    for thread in threading.enumerate():
        running_threads.append(thread.name)
    if 'status_update_thread' not in running_threads:
        global_control.status_thread = threading.Thread(name='status_update_thread', target=update_tpot_status)
        global_control.status_thread.start()

    global_control.tpot.fit(x_train, y_train)
    global_control.tpot_status['status'] += '<br/><date>' + datetime.now().strftime(
        "%d.%m.%Y|%H-%M-%S") + ':</date> TPOT finished'
    global_control.tpot_status['time'] = datetime.now() - start

    subfolder_name = datetime.now().strftime("%Y%m%d-%H_%M_%S")
    save_results_tpot(subfolder_name)

    print('Koniec ewolucji tpot')


def update_tpot_status():
    while True:
        time.sleep(1.2)
        update_plot_tpot(global_control.tpot.evaluated_individuals_)
        try:
            update_plot_tpot(global_control.tpot.evaluated_individuals_)
        except AttributeError:
            log.info('TPOT not running - nothing to refresh')


def start_evolution_simple(file_name, core_balance):
    try:
        dataset = pd.read_csv('data-CSV/' + file_name, delimiter=',')
    except FileNotFoundError:
        return '<h1>Please select file_name to analyze.</h1>'
    dataset = adjust_dataset(dataset)
    features = dataset.drop('class', axis=1).values
    x_train, x_test, y_train, y_test = train_test_split(features, dataset['class'].values, test_size=0.1,
                                                        random_state=13)
    core_balance = int(core_balance)
    available_cores = global_control.machine_info['logical_cores']
    if core_balance == 0:
        gmc_threads = int(available_cores / 2)
        tpot_threads = int(available_cores / 2)
    else:
        gmc_threads = int(available_cores / 2 - core_balance)
        tpot_threads = int(available_cores / 2 + core_balance)

    print(f'{gmc_threads=}')
    print(f'{tpot_threads=}')

    # threading will only use one-thread in pure code, but sklearn will use n_jobs and multithreading
    running_threads = []
    for thread in threading.enumerate():
        running_threads.append(thread.name)
    print(running_threads)

    if gmc_threads != 0:
        gmc_thread = threading.Thread(target=run_evolve, name="gmc_thread",
                                      args=(x_train, y_train, gmc_threads, file_name))
        if 'gmc_thread' not in running_threads:
            # If we want the app to keep running, even when webapp gets reloaded we should use Process instead of Thread
            # gmc_thread = Process(target=run_evolve, args=(x_train, y_train), name="gmc_process")
            gmc_thread.start()
        else:
            global_control.status['status'] += '<br/><date>' + datetime.now().strftime(
                "%d.%m.%Y|%H-%M-%S") + ":</date> Task already running."
            print(f'Saving thread for later: {gmc_thread=}')
            global_control.queue.append(gmc_thread)

    if tpot_threads != 0:
        num_threads = str(int(tpot_threads))
        # different platforms have different names
        os.environ["OMP_NUM_THREADS"] = num_threads
        os.environ["OPENBLAS_NUM_THREADS"] = num_threads
        os.environ["MKL_NUM_THREADS"] = num_threads
        os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
        os.environ["NUMEXPR_NUM_THREADS"] = num_threads
        features = dataset.drop('class', axis=1).values
        x_train, x_test, y_train, y_test = train_test_split(features, dataset['class'].values, test_size=0.2,
                                                            random_state=13)
        # threading will only use one-thread in pure code, but sklearn will use n_jobs and multithreading
        global_control.tpot_thread = threading.Thread(target=run_tpot, name="tpot_thread",
                                                      args=(file_name, x_train, y_train, tpot_threads, 10, 13))
        # tpot_thread = multiprocessing.Process(target=run_tpot, name="tpot_thread",
        #                                       args=(file_name, x_train, y_train, tpot_threads, 10, 13))
        if 'tpot_thread' not in running_threads:
            global_control.tpot_thread.start()
        else:
            global_control.tpot_status['status'] += '<br/><date>' + datetime.now().strftime(
                "%d.%m.%Y|%H-%M-%S") + ":</date> Task already running."
            print(f'Saving thread for later: {global_control.tpot_thread}')
            global_control.queue.append(global_control.tpot_thread)


def load_best_pipelines():
    pipelines = dict()
    try:
        sub_folders = [f.path for f in os.scandir('results') if f.is_dir()]
    except FileNotFoundError:
        log.info('No results folder')
        return
    print(sub_folders)
    for folder in sub_folders:
        try:
            with open(os.path.join(folder, 'best_pipeline_summary.pickle'), 'rb') as handle:
                pipelines[folder[8:]] = pickle.load(handle)
        except FileNotFoundError:
            continue
    print(f'{pipelines=}')
    return pipelines


def unpickle_pipelines():
    pipelines = []
    try:
        sub_folders = [f.path for f in os.scandir('results') if f.is_dir()]
    except FileNotFoundError:
        log.info('No results folder')
        return
    print(sub_folders)
    for folder in sub_folders:
        try:
            with open(os.path.join(folder, 'best_pipeline.pickle'), 'rb') as handle:
                pipelines.append(pickle.load(handle))
        except FileNotFoundError:
            continue
    filtered_pipes = [x for x in pipelines if not isinstance(x, str)]
    global_control.PIPELINES = filtered_pipes


def test_pipelines(pipelines: [], file_name: str):
    try:
        dataset = pd.read_csv('data-CSV/' + file_name, delimiter=',')
    except FileNotFoundError:
        return '<h1>Please select file_name to analyze.</h1>'
    dataset = adjust_dataset(dataset)
    features = dataset.drop('class', axis=1).values
    x_train, x_test, y_train, y_test = train_test_split(features, dataset['class'].values, test_size=0.1,
                                                        random_state=13)
    cv = []
    for pipeline in global_control.PIPELINES:
        for p in pipelines:
            p = [p.replace("\r\n", "").replace("  ", "") for x in p]
            pip_string = str(pipeline)
            pip_string = [pip_string.replace("\n", "").replace("  ", "") for x in pip_string]
            if p[0] == pip_string[0]:
                cv = cross_val_score(pipeline, x_train, y_train, cv=10, n_jobs=4, error_score="raise")
                print(f'Kroswaliduję {cv=}')
