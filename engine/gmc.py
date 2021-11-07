import copy
import os

import random
import logging as log
import sys
from dataclasses import dataclass

# import mkl
import stopit
import traceback
from deepdiff import DeepDiff
from operator import attrgetter
from datetime import datetime
import cpuinfo

import numpy as np
import numpy.random.bit_generator
import pandas as pd
from stopit import TimeoutException

import global_control
from global_control import init_stop_threads
from utils import update_progress, update_hall_of_fame, update_status, update_plot, get_best_from_list, average_time

from sklearn.model_selection import cross_val_score, LeaveOneOut, RandomizedSearchCV, ParameterGrid, train_test_split

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Binarizer, PowerTransformer, \
    QuantileTransformer
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

CPU_INFO = cpuinfo.get_cpu_info()
if CPU_INFO['vendor_id_raw'] == 'GenuineIntel':
    log.info(f'Intel CPU detected. Using daal4py(intel optimized) libraries')
    from daal4py.sklearn.neighbors import KNeighborsClassifier
    from daal4py.sklearn.ensemble import RandomForestClassifier
    from daal4py.sklearn.svm import SVC
    from daal4py.sklearn.linear_model import LogisticRegression
    from daal4py.sklearn.decomposition import PCA
    from sklearn.decomposition import FastICA
    # from daal4py.sklearn.ensemble import AdaBoostClassifier
    # not fully compatible with original - no random_state
else:
    log.info(f'Not Intel CPU. Using standard libraries')
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.decomposition import FastICA, PCA


@dataclass()
class Individual:
    pipeline: Pipeline
    pipeline_string: str
    genome: tuple()
    validation_method: float = None
    score: float = None
    validation_time: str = None
    grid_used: dict() = None
    all_combinations: int = 0  # possible combinations of generated individuals


# it would be nice to convert history to set, but to do that Individual would have to be hashable
@dataclass()
class Population:
    individuals: [Individual]
    history: [Individual]
    slowpokes: [Individual] = None
    failed_to_test: [Individual] = None
    original_dataset: pd.DataFrame = None
    adjusted_dataset: pd.DataFrame = None
    dataset_name: str = None
    dataset_classes: int = 0
    dataset_rows: int = 0
    dataset_attributes: int = 0
    random_state: int = 13
    partial_explore: int = 0.0
    notes: [str] = None


ORIGINAL_X_TRAIN = pd.DataFrame
ORIGINAL_Y_TRAIN = pd.DataFrame
MAX_N_COMPONENTS = 1
RANDOM_STATE = 1
N_JOBS = 1
PRESELECTION = False
SELECTION = 'roulette'
CV = 10
TIME_LIMIT_GENERATION = 1200
TIME_LIMIT_PIPELINE = 600
GUI = True
GENERATOR_COUNTER = 1


def set_time_limits(time_s, generations):
    global TIME_LIMIT_GENERATION
    global TIME_LIMIT_PIPELINE
    TIME_LIMIT_GENERATION = time_s * generations
    TIME_LIMIT_PIPELINE = time_s
    if TIME_LIMIT_GENERATION > 1000000:
        TIME_LIMIT_GENERATION = 1000000


def increase_counter():
    global GENERATOR_COUNTER
    if GENERATOR_COUNTER > 10000000:
        GENERATOR_COUNTER = 1
    else:
        GENERATOR_COUNTER += 1


def reset_counter():
    global GENERATOR_COUNTER
    GENERATOR_COUNTER = 1


CLASSIFIERS = [BernoulliNB(), GaussianNB(), KNeighborsClassifier(), DecisionTreeClassifier(), ExtraTreeClassifier(),
               RandomForestClassifier(), GradientBoostingClassifier(), LogisticRegression(),
               GaussianProcessClassifier(), PassiveAggressiveClassifier(), RidgeClassifier(), SGDClassifier(),
               AdaBoostClassifier(), BaggingClassifier(), NearestCentroid(), Perceptron(),
               MLPClassifier(), LinearSVC(), SVC(), XGBClassifier(), NuSVC(), LinearDiscriminantAnalysis()]
# LinearDiscriminantAnalysis causes a lot of computation errors in rare cases
# another problematic one - NuSVC sometimes throwing ValueError: specified nu is infeasible
# CLASSIFIERS = [BernoulliNB()]

TRANSFORMERS = [StandardScaler(), MinMaxScaler(), RobustScaler(),
                PCA(),
                Binarizer(),
                PowerTransformer(), FastICA()]
TRANSFORMERS_NAMES = ['standardscaler', 'minmaxscaler', 'robustscaler',
                      'pca',
                      'binarizer',
                      'powertransformer',
                      'quantiletransformer', 'FastICA']


def evolve(population, generations: int, validation_method, x_train, y_train, elitism: int, random_state: int,
           dataset_name: str,
           selection_type='roulette', crossover_rate=1.0, cross_method='average',
           early_stop=20,
           pipeline_time_limit=600, n_jobs=1,
           preselection=False, fresh_blood_mode=True, grid_type='GMC-big', mutation_rate=0.5,
           mutation_power=1.0, partial_explore=0.0) -> Population:
    set_max_n_components(x_train, validation_method)
    set_random_state(random_state)
    set_time_limits(pipeline_time_limit, generations)
    set_njobs(n_jobs)
    set_threads(n_jobs)
    set_selection(selection_type)
    set_preselection(preselection)
    global_control.status['scores'] = []
    global_control.status['avgs'] = []

    log.info(f'MAX_N_COMPONENTS was set to: {MAX_N_COMPONENTS}')
    log.info(f'TIME_LIMIT_POPULATION was set to: {TIME_LIMIT_GENERATION}')
    log.info(f'TIME_LIMIT_PIPELINE was set to: {TIME_LIMIT_PIPELINE}')

    if partial_explore != 0.0:
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=partial_explore,
                                                            random_state=random_state)
    backup_data(x_train, y_train)

    if early_stop != 0 and early_stop < 4:
        early_stop = 4
    if early_stop == 0:
        early_stop = 100000000
    if isinstance(population, int):
        pop = init_population(population, grid_type)
    elif isinstance(population, Population):
        pop = population
    else:
        log.critical('Population must be int or Population!')
        global_control.status['status'] += '<br/><date>' + datetime.now().strftime(
            "%d.%m.%Y|%H-%M-%S") + ":</date> Got unexpected type in population parameter, restoring default value (20)."
        log.critical('Setting population to 20')
        pop = init_population(20, grid_type)

    if elitism >= len(pop.individuals):
        log.error('Elitism must be smaller than population size. Auto adjusted to (pop_size-2)')
        if GUI:
            global_control.status['status'] += '<br/><date>' + datetime.now().strftime(
                "%d.%m.%Y|%H-%M-%S") + ':</date> Elitism must be smaller than population size. Auto adjusted to (pop_size-2)'
        elitism = len(pop.individuals) - 2

    if grid_type == 'GMC-minimal':
        global_control.status['status'] += '<br/><date>' + datetime.now().strftime(
            "%d.%m.%Y|%H-%M-%S") + ":</date> Using minimal grid."
    if grid_type == 'GMC-big':
        global_control.status['status'] += '<br/><date>' + datetime.now().strftime(
            "%d.%m.%Y|%H-%M-%S") + ":</date> Using big grid."
    if grid_type == 'GMC-extreme':
        global_control.status['status'] += '<br/><date>' + datetime.now().strftime(
            "%d.%m.%Y|%H-%M-%S") + ":</date> Using extreme grid."
    if grid_type == 'TPOT-ish':
        global_control.status['status'] += '<br/><date>' + datetime.now().strftime(
            "%d.%m.%Y|%H-%M-%S") + ":</date> Using grid similar to TPOT one."

    if fresh_blood_mode:
        global_control.status['status'] += '<br/><date>' + datetime.now().strftime(
            "%d.%m.%Y|%H-%M-%S") + ":</date> Fresh genes allowed."

    hall_of_fame = []
    best = 0.
    stop_counter = early_stop
    pop.dataset_name = dataset_name
    pop.dataset_classes = len(np.unique(y_train))
    log.debug(f'Class attribute variety {pop.dataset_classes=}')
    pop.dataset_rows = x_train.shape[0]
    pop.dataset_attributes = x_train.shape[1]
    pop.partial_explore = partial_explore

    # For results reproducibility
    random.seed(random_state)
    np.random.seed(random_state)

    start = datetime.now()
    init_stop_threads()
    for i in range(generations):
        progress = ((i + 1) / generations * 100)
        if global_control.stop_threads:
            log.info('Task stopped - returning last population')
            update_status('Task stopped - returning last population')
            # pop.individuals.sort(key=lambda x: x.score is not None, reverse=True)
            init_stop_threads()
            log.info(f"Stop requested, returning...\n{best=}\n{generation_best_individual.pipeline}")
            reset_counter()
            return pop
        if GUI:
            update_status(f'Validating generation {i}/{generations}')
        with stopit.ThreadingTimeout(TIME_LIMIT_GENERATION) as to_ctx_mgr:
            assert to_ctx_mgr.state == to_ctx_mgr.EXECUTING
            pop = test_population(pop, validation_method, x_train, y_train, grid_type)
        if to_ctx_mgr.state == to_ctx_mgr.EXECUTED:
            log.debug('Generation tested')
        elif to_ctx_mgr.state == to_ctx_mgr.EXECUTING:
            log.error('Time limitations error - executing was passed after execution')
            continue
        elif to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
            log.warning(f'Time limitation exceeded - generation {i} not tested')
            continue
        unique = list()
        for individual in pop.individuals:
            # be VERY carefully with objects on list!
            # appending object to another list only links to it, operating on either changes both!
            unique.append(copy.deepcopy(individual))
        for u in unique:
            # don't store pipeline in history to safe memory
            u.pipeline_string = str(u.pipeline)
            u.pipeline = None
        pop.history.append(unique)
        generation_best_individual = get_best_from_list(pop.individuals)
        generation_best_score = generation_best_individual.score
        log.info(f"leader candidate...\n{best=}\n{generation_best_individual.pipeline}")
        if generation_best_score <= best:
            stop_counter -= 1
        else:
            stop_counter = early_stop
            best = generation_best_score
            global_control.status['best_score'] = best
            global_control.status['pipeline'] = copy.deepcopy(generation_best_individual.pipeline)
            log.info(f"New leader found...\n{best=}\n{generation_best_individual.pipeline}")

        if stop_counter < 1:
            log.info(f'Early stop! No change in {early_stop} generations')
            update_status(f'Early stop! No change in {early_stop} generations')
            log.info(
                f"\nStop counter triggered, returning...\n{global_control.status['best_score']}\n{global_control.status['pipeline']}")
            reset_counter()
            return pop

        if stop_counter <= early_stop // 2 and fresh_blood_mode:
            log.info(
                f'Approaching early stop! Adding fresh genes')
            update_status(
                f'Approaching early stop! Adding fresh genes')
            fresh_pop = Population(individuals=[], history=pop.history, dataset_name=pop.dataset_name,
                                   dataset_rows=pop.dataset_rows,
                                   dataset_attributes=pop.dataset_attributes, dataset_classes=pop.dataset_classes,
                                   random_state=RANDOM_STATE, failed_to_test=pop.failed_to_test,
                                   slowpokes=pop.slowpokes)
            for x in range(len(pop.individuals) - len(hall_of_fame)):
                fresh_pop.individuals.append(generate_random_individual(grid_type))
            fresh_pop.individuals.append(hall_of_fame)
            fresh_pop = test_population(pop, validation_method, x_train, y_train, grid_type)
            pop = fresh_pop
            generation_best_individual = get_best_from_list(pop.individuals)
            generation_best_score = generation_best_individual.score
            if generation_best_score <= best:
                # stop_counter -= 1
                None
            else:
                stop_counter = early_stop
                best = generation_best_score
                global_control.status['best_score'] = best
                global_control.status['pipeline'] = copy.deepcopy(generation_best_individual.pipeline)
                log.info(f"New-Fresh-lider found...\n{best=}\n{generation_best_individual.pipeline}")

        update_progress(progress, start)
        if GUI:
            update_plot(pop)
            if 'time' in global_control.status:
                #                                   print(f"{type(global_control.status['time'])=}")
                update_status(
                    f"Time elapsed:{global_control.status['time']}|Last generation average fitting time:{average_time(pop.individuals)}")

        log.info('Selection and crossover')
        pop, hall_of_fame = selection_and_crossover(population=pop, elitism=elitism, hall_of_fame=hall_of_fame,
                                                    selection_type=selection_type, crossover_rate=crossover_rate,
                                                    crossover_method=cross_method, mutation_rate=mutation_rate,
                                                    mutation_power=mutation_power)

        log.info('Population size: ' + str(len(pop.individuals)))
    if GUI:
        update_status('Final validation')
    pop = test_population(pop, validation_method, x_train, y_train, grid_type)
    pop.history.append(pop.individuals)
    generation_best_individual = get_best_from_list(pop.individuals)
    generation_best_score = generation_best_individual.score
    if generation_best_score <= best:
        None
    else:
        best = generation_best_score
        global_control.status['best_score'] = best
        global_control.status['pipeline'] = copy.deepcopy(generation_best_individual.pipeline)
        log.info(f"New leader found in last generation...\n{best=}\n{generation_best_individual.pipeline}")
    if GUI:
        update_status('Task finished')
    log.info(f"Generations limit reached, returning...\n{best=}\n{generation_best_individual.pipeline}")
    reset_counter()
    return pop


def init_population(pop_size: int, grid_type: str):
    pop = Population(individuals=[], history=[], failed_to_test=[], slowpokes=[])
    for x in range(pop_size):
        pop.individuals.append(generate_random_individual(grid_type))
        log.info('Individual generated: ' + str(pop.individuals[x].pipeline))
    return pop


def generate_random_individual(grid_type):
    # counter is used to avoid generating same individuals each time, but generate same results at re-run
    np.random.seed(RANDOM_STATE + GENERATOR_COUNTER)
    increase_counter()
    transformers = set()
    # randomly choose transformers
    for x in range(np.random.randint(0, 3)):
        transformers.add(np.random.choice(TRANSFORMERS))

    param_grid, name_object_tuples = get_min_param_grid_and_tuple_list(transformers)

    clf = np.random.choice(CLASSIFIERS)
    log.debug(f'Random classifier:{clf=}')
    # pipeline = make_pipeline(*name_object_tuples, clf)
    pipeline = Pipeline(steps=[*name_object_tuples, (type(clf).__name__.lower(), clf)])
    ind = Individual(pipeline=pipeline, genome=None, validation_method=None, score=None, validation_time=None,
                     pipeline_string=str(pipeline))

    if grid_type == 'GMC-minimal':
        param_grid = update_param_grid_minimal(clf, param_grid)
    if grid_type == 'GMC-big':
        param_grid = update_param_grid_big(clf, param_grid)
    if grid_type == 'GMC-extreme':
        param_grid = update_param_grid_extreme(clf, param_grid)
    if grid_type == 'TPOT-ish':
        param_grid = update_param_grid_identical_as_tpot(clf, param_grid)

    grid = ParameterGrid(param_grid)
    log.debug(f'All possible combinations for this individual:{len(grid)}')
    ind.__setattr__('all_combinations', len(grid))
    ind.__setattr__('grid_used', grid)

    if PRESELECTION:
        with stopit.ThreadingTimeout(TIME_LIMIT_PIPELINE * 5) as to_ctx_mgr:
            assert to_ctx_mgr.state == to_ctx_mgr.EXECUTING
            search = RandomizedSearchCV(pipeline, param_grid, n_jobs=N_JOBS, cv=2, n_iter=10)
            # search = HalvingRandomSearchCV(pipeline, param_grid, n_jobs=N_JOBS, cv=2)
            # search = GridSearchCV(pipeline, param_grid, n_jobs=4, cv=2)
            search.fit(ORIGINAL_X_TRAIN, ORIGINAL_Y_TRAIN)
            pipeline = search.best_estimator_
            ind.__setattr__('pipeline', pipeline)
        if to_ctx_mgr.state == to_ctx_mgr.EXECUTED:
            log.info('Pipeline preselection was successful')
            return ind
        elif to_ctx_mgr.state == to_ctx_mgr.EXECUTING:
            log.error('Time limitations error - executing was passed after execution')
        elif to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
            log.warning(f'Preselection time limitation exceeded - selecting at random')

    # IndexError: ParameterGrid index out of range
    # e.g.
    #  random_grid_index=3796231240
    #          len(grid)=5836800000
    # it seems to only happen with TPOTish grid and every call, even grid[0] throws IE
    # for now recursive fix was applied
    random_grid_index = np.random.randint(0, len(grid) - 1)
    try:
        random_individual_grid = grid[random_grid_index]
    except IndexError:
        log.error(f'Random grid selection failed.')
        return generate_random_individual(grid_type)

    pipeline.set_params(**random_individual_grid)
    log.debug(f'Random pipeline: {pipeline}')
    ind.__setattr__('pipeline', pipeline)
    return ind


# https://www.python.org/dev/peps/pep-0281/
def selection_and_crossover(population: Population, elitism: int, hall_of_fame: list, crossover_rate: float,
                            selection_type: str, crossover_method: str, mutation_rate: float, mutation_power: float):
    # initialize next_generation, pass history from population
    next_generation = Population(individuals=[], history=population.history, dataset_name=population.dataset_name,
                                 dataset_rows=population.dataset_rows, dataset_attributes=population.dataset_attributes,
                                 dataset_classes=population.dataset_classes, random_state=RANDOM_STATE,
                                 failed_to_test=population.failed_to_test, slowpokes=population.slowpokes)

    random.seed(RANDOM_STATE + GENERATOR_COUNTER)
    increase_counter()

    log.debug(f'BEFORE CROSSOVER {len(next_generation.individuals)=}')
    # creating new generation
    for x in range(len(population.individuals) - elitism):
        # crossover with selection - add new individual to next_generation
        if crossover_rate > random.random():
            if selection_type == 'roulette':
                next_generation.individuals.append(
                    crossover(roulette(population.individuals), roulette(population.individuals), crossover_method))
            if selection_type == 'tournament5':
                next_generation.individuals.append(
                    crossover(tournament(population.individuals, 5), tournament(population.individuals, 5),
                              crossover_method))
            if selection_type == 'tournament10':
                next_generation.individuals.append(
                    crossover(tournament(population.individuals, 10), tournament(population.individuals, 10),
                              crossover_method))
            if selection_type == 'tournament15':
                next_generation.individuals.append(
                    crossover(tournament(population.individuals, 15), tournament(population.individuals, 15),
                              crossover_method))
            if selection_type == 'tournament20':
                next_generation.individuals.append(
                    crossover(tournament(population.individuals, 20), tournament(population.individuals, 20),
                              crossover_method))
        else:
            if selection_type == 'roulette':
                next_generation.individuals.append(copy.deepcopy(roulette(population.individuals)))
            if selection_type == 'tournament5':
                next_generation.individuals.append(copy.deepcopy(tournament(population.individuals, 5)))
            if selection_type == 'tournament10':
                next_generation.individuals.append(copy.deepcopy(tournament(population.individuals, 10)))
            if selection_type == 'tournament15':
                next_generation.individuals.append(copy.deepcopy(tournament(population.individuals, 15)))
            if selection_type == 'tournament20':
                next_generation.individuals.append(copy.deepcopy(tournament(population.individuals, 20)))
    log.debug(f'AFTER CROSSOVER {len(next_generation.individuals)=}')

    log.info(
        'New generation, scores before mutation:{scores}'.format(scores=[x.score for x in next_generation.individuals]))

    # for x in range(len(next_generation.individuals) - elitism):
    for x, individual in enumerate(next_generation.individuals):
        if mutation_rate > random.random():
            try:
                genes = mutate(create_genome(individual.pipeline), mutation_rate,
                               mutation_power)
            except OverflowError:
                log.error('Too aggressive mutation, individual not mutated')
                continue
            individual.pipeline = create_pipeline(genes)
            individual.genome = genes
            individual.score = None
            individual.validation_method = None
            individual.genome = None
            individual.pipeline_string = None
            individual.validation_time = None

    log.debug(f'AFTER MUTATION {len(next_generation.individuals)=}')
    if 0 < elitism:
        # update HoF
        hall_of_fame = update_hall_of_fame(hall_of_fame, population.individuals, elitism)
        log.debug(
            'Add best individuals to new generation, scores:{scores}'.format(scores=[x.score for x in hall_of_fame]))

        log.info(
            'Add best individuals to new generation, scores:{scores}'.format(scores=[x.score for x in hall_of_fame]))

        for b in hall_of_fame:
            next_generation.individuals.append(b)
        # for i in range(len(next_generation.individuals)):
        #     print(next_generation.individuals[i].score)
    log.debug(f'AFTER ELITISM {len(next_generation.individuals)=}')
    log.info('New generation created, scores:{scores}'.format(scores=[x.score for x in next_generation.individuals]))
    return next_generation, hall_of_fame


def test_individual(population: Population, x: Individual, validation_method, x_train, y_train, grid_type):
    log.info(f'Before testing...\n {x.pipeline=} \ntested in: {x.validation_time}, score:{x.score}')
    # if 'pipeline' in global_control.status:
    #     print(f"{global_control.status['pipeline']=}")
    if hasattr(x.pipeline, 'random_state'):
        setattr(x.pipeline, 'random_state', int(RANDOM_STATE))
    if x.genome is None:
        log.info('No genome found. Generating genome.')
        x.genome = create_genome(x.pipeline)
    if x.score is None:
        for y in unpack_history(population).individuals:
            # if x.genome == y.genome and y.score is not None:
            # DeepDiff also returns what exactly didn't match
            if y.score is not None and not DeepDiff(x.genome, y.genome, get_deep_distance=True):
                log.warning(
                    f'\nIdentical pipeline was already tested. If you get this message often - increase genes variety ('
                    f'bigger pop, more mutation/crossover, wider param_grid).\nPipeline 1:{x.pipeline} \nPipeline 2:{y.pipeline} \nGenes 1:{x.genome} \nGenes 2:{y.genome}')
                # recreate pipeline for historical individual
                y.pipeline = create_pipeline(y.genome)
                return y, population
                # log.debug('Generating random individual')
                # print(f'{DeepDiff(x.genome, y.genome, significant_digits=10)=}')
                # x = generate_random_individual(grid_type)
                # x.genome = create_genome(x.pipeline)
                # return test_individual(population, generate_random_individual(grid_type), validation_method,
                #                        x_train,
                #                        y_train, grid_type)
        log.debug('Passing pipeline to test:' + str(x.pipeline))
        with stopit.ThreadingTimeout(TIME_LIMIT_PIPELINE) as to_ctx_mgr:
            assert to_ctx_mgr.state == to_ctx_mgr.EXECUTING
            start = datetime.now()
            try:
                cv = cross_val_score(x.pipeline, x_train, y_train, cv=validation_method, n_jobs=N_JOBS,
                                     error_score="raise")
                # print(f'\n\nTesting:{x.pipeline}\n{sum(cv)/len(cv)=}\n\n{x_train}')
            except (TypeError, ValueError) as e:
                population.failed_to_test.append(x)
                log.error(f"FITTING ERROR\n{x.pipeline}\nRESTORING TRAINING SET, GENERATING NEW INDIVIDUAL")
                log.error(e.__class__)
                log.error(e)
                x_train = ORIGINAL_X_TRAIN.copy()
                y_train = ORIGINAL_Y_TRAIN.copy()
                # x = generate_random_individual(grid_type)
                start = datetime.now()
                try:
                    return test_individual(population, generate_random_individual(grid_type), validation_method,
                                           x_train, y_train, grid_type)
                except (TypeError, ValueError) as e:
                    log.critical(
                        'Critical error - unable to generate random individual')
                    log.error(e.__class__)
                    log.error(e)
                    population.failed_to_test.append(x)
                except Exception:
                    log.critical(
                        f'Critical error - panic allowed, traceback:{traceback.print_exc()}, formatted traceback:{traceback.format_exc()}, sys.exec_info:{sys.exc_info()}')
            except TimeoutException:
                log.error(f'Timeout exception')
                population.slowpokes.append(x)
                # x = generate_random_individual(grid_type)
                return test_individual(population, generate_random_individual(grid_type), validation_method, x_train,
                                       y_train, grid_type)
            except Exception:
                log.critical(
                    f'Critical error - panic allowed, traceback:{traceback.print_exc()}, formatted traceback:{traceback.format_exc()}, sys.exec_info:{sys.exc_info()}')
            x.validation_time = datetime.now() - start
            x.validation_method = validation_method
            try:
                x.score = sum(cv) / len(cv)
            except UnboundLocalError:
                log.critical(f'Cross-validation error - individual not tested:{x.pipeline}')
                x.score = 0.0

            log.info(f'Pipeline {x.pipeline} \ntested in: {x.validation_time}, score:{x.score}')
        if to_ctx_mgr.state == to_ctx_mgr.EXECUTED:
            log.info('Pipeline validation was successfull')
        elif to_ctx_mgr.state == to_ctx_mgr.EXECUTING:
            log.error('Time limitations error - executing was passed after execution')
        elif to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
            log.error(f'Time limitation exceeded, skipped pipeline: {x.pipeline}')
            population.slowpokes.append(x)
            log.info('Generating random individual')
            # x = generate_random_individual(grid_type)
            return test_individual(population, generate_random_individual(grid_type), validation_method, x_train,
                                   y_train, grid_type)

        elif to_ctx_mgr.state == to_ctx_mgr.INTERRUPTED:
            log.warning('Time limitation interrupted')
            # x = generate_random_individual(grid_type)
            return test_individual(population, generate_random_individual(grid_type), validation_method, x_train,
                                   y_train, grid_type)
        elif to_ctx_mgr.state == to_ctx_mgr.CANCELED:
            print('Oh you called to_ctx_mgr.cancel() method within the block but it')
        else:
            print('That\'s not possible')
            x.score = 0.0

        x.pipeline_string = str(x.pipeline)
        log.info(f'Returning after test...\n {x.pipeline=} \ntested in: {x.validation_time}, score:{x.score}')
    return x, population


def test_population(population: Population, validation_method, x_train, y_train, grid_type):
    tested_population = Population(individuals=[], history=population.history, dataset_name=population.dataset_name,
                                   dataset_rows=population.dataset_rows,
                                   dataset_attributes=population.dataset_attributes,
                                   dataset_classes=population.dataset_classes,
                                   random_state=RANDOM_STATE, failed_to_test=population.failed_to_test,
                                   slowpokes=population.slowpokes)
    log.info(f'Population size: {len(population.individuals)}')
    for x in population.individuals:
        x, population = test_individual(population, x, validation_method, x_train, y_train, grid_type)
        try:
            if not np.isnan(x.score):
                tested_population.individuals.append(x)
            else:
                log.critical(
                    f'Critical error, individual not scored. Try increasing time limit or use less extreme grid')
                x = generate_random_individual(grid_type)
                x, population = test_individual(population, x, validation_method, x_train, y_train, grid_type)
                if not np.isnan(x.score):
                    tested_population.individuals.append(x)
                else:
                    x.score = 0.0
                    tested_population.individuals.append(x)
        except (TypeError, ValueError):
            x.score = 0.0
            tested_population.individuals.append(x)
            log.critical(
                f'Critical error, individual not scored. Try increasing time limit or use less extreme grid')

    return validate_tested_population(tested_population)


def validate_tested_population(population: Population):
    tested_population_validated = Population(individuals=[], history=population.history,
                                             dataset_name=population.dataset_name,
                                             dataset_rows=population.dataset_rows,
                                             dataset_attributes=population.dataset_attributes,
                                             dataset_classes=population.dataset_classes, random_state=RANDOM_STATE,
                                             failed_to_test=population.failed_to_test, slowpokes=population.slowpokes)
    for x in population.individuals:
        if isinstance(x.score, float):
            log.info('Individual verified, score:' + str(x.score))
            tested_population_validated.individuals.append(x)
        else:
            log.error(f'Individual validation failed: {x.pipeline}')
    # log.info(f'Average score of population:{average_score(population.individuals)}')
    # if GUI:
    #     global_control.status[
    #         'status'] += f' | average:{average_score(population.individuals)} | best:{global_control.status["best_score"]}'
    return tested_population_validated


# https://en.wikipedia.org/wiki/Fitness_proportionate_selection#Pseudocode
# I implemented it in slightly different way
def roulette(individuals):
    score_sum = sum(i.score for i in individuals)
    wheel_sum = 0.0
    choice = np.random.uniform(1.0, 0.0)
    for x in individuals:
        wheel_sum += (x.score / score_sum)
        if wheel_sum > choice:
            return x


def tournament(individuals, tournament_size):
    arena = random.choices(individuals, k=tournament_size)
    arena.sort(key=lambda x: x.score, reverse=True)
    return arena[0]


def get_best_from_pop_test(pop, x_train, y_train, validation_method):
    scores = []
    pop2 = []
    for x in pop.individuals:
        x.pipeline.fit(x_train, y_train)
        cv = cross_val_score(x.pipeline.x_train, y_train, cv=validation_method, n_jobs=N_JOBS)
        pop2.append(x.pipeline.
                    scores.append(str(sum(cv) / len(cv))))
        best = pop2[np.argmax(scores)]
    return best


def get_best_from_pop(pop, validation_method):
    if len(pop.individuals) == 0:
        log.warning('No Individuals found in Population!')
        return 0
    pop2 = []
    best = None
    for x in pop.individuals:
        if x.validation_method is None:
            log.warning("Individual ignored - not tested yet")
            continue
        if x.validation_method != validation_method:
            log.warning("Individual ignored - different cross-validation values were used in testing.")
            continue
        pop2.append(x)
    if len(pop2) != 0:
        best = max(pop2, key=attrgetter('score'))
    return best


def single_point_crossover(key1, key2):
    binary_1 = to_bit_array(key1)
    binary_2 = to_bit_array(key2)
    # we need equal binary arrays, let's add zeroes to left side of smaller one
    if len(binary_1) != len(binary_2):
        if len(binary_1) > len(binary_2):
            binary_2 = [0] * (len(binary_1) - len(binary_2)) + binary_2
        else:
            binary_1 = [0] * (len(binary_2) - len(binary_1)) + binary_1
    # random point
    point = random.randint(0, len(binary_2))
    # random side
    if random.random() > 0.5:
        x = binary_1[:point] + binary_2[point:]
    else:
        x = binary_2[:point] + binary_1[point:]
    # convert array of bits to int
    value = 0
    for bit in x:
        value = (value << 1) | bit
    return value


def uniform_crossover(key1, key2):
    binary_1 = to_bit_array(key1)
    binary_2 = to_bit_array(key2)
    # we need equal binary arrays, let's add zeroes to left side of smaller one
    if len(binary_1) != len(binary_2):
        if len(binary_1) > len(binary_2):
            binary_2 = [0] * (len(binary_1) - len(binary_2)) + binary_2
        else:
            binary_1 = [0] * (len(binary_2) - len(binary_1)) + binary_1

    x = []
    # randomly assign bits
    for i in range(len(binary_1)):
        if random.random() > 0.5:
            x.append(binary_1[i])
        else:
            x.append(binary_2[i])
    value = 0
    for bit in x:
        value = (value << 1) | bit
    return value


def crossover(inv1: Individual, inv2: Individual, crossover_method):
    g1 = create_genome(inv1.pipeline)
    g2 = create_genome(inv2.pipeline)
    # Base child on random parent
    if random.random() > 0.5:
        g3 = g1
    else:
        g3 = g2
    # print(f'First parent genes:{g1}')
    # print(f'Second parent genes{g2}: ')
    keys_to_skip = {'classifier_type', 'verbose', 'random_state', 'gpu_id'}

    # mixing common keys
    for key in g1:
        if key in g2:
            # print(f'Common key:{key}')
            if key not in keys_to_skip:
                # skip nans
                try:
                    if np.isnan(g1[key]) or np.isnan(g2[key]):
                        continue
                except (TypeError, ValueError):
                    continue
                # always check if is bool first if you also checking if is int, remember to add continue/break
                if isinstance(g1[key], bool) and isinstance(g2[key], bool):
                    if random.random() > 0.5:
                        g3[key] = g1[key]
                    else:
                        g3[key] = g2[key]
                    continue

                if isinstance(g1[key], str) and isinstance(g2[key], str):
                    if random.random() > 0.5:
                        g3[key] = g1[key]
                    else:
                        g3[key] = g2[key]
                    # print(f'Random string from one of parents:{g3[key]}')
                    continue

                if crossover_method == 'average':

                    if isinstance(g1[key], float) and isinstance(g2[key], float):
                        g3[key] = (g1[key] + g2[key]) / 2
                        log.debug(f'float after crossover:{g3[key]}')
                        continue
                    if isinstance(g1[key], int) and isinstance(g2[key], int):
                        g3[key] = int((g1[key] + g2[key]) / 2)
                        log.debug(f'int after crossover:{g3[key]}')
                        continue

                elif crossover_method == 'single-point':
                    if isinstance(g1[key], float) and isinstance(g2[key], float):
                        split1 = g1[key] // 1, g1[key] % 1
                        split2 = g2[key] // 1, g2[key] % 1
                        # print(f'{split1=}')
                        # print(f'{split2=}')
                        before_dec = single_point_crossover(int(split1[0]), int(split2[0]))
                        # scientific notation eg 1e-09 will cause IndexError in split function
                        # therefore we have to make sure it's in positional notation
                        if split1[1] != 0.0:
                            split1 = split1[0], np.format_float_positional(split1[1])
                        if split2[1] != 0.0:
                            split2 = split2[0], np.format_float_positional(split2[1])
                        after_dec = single_point_crossover(
                            int(str(split1[1]).split('.')[1]),
                            int(str(split2[1]).split('.')[1]))

                        g3[key] = float(f'{before_dec}.{after_dec}')
                        continue
                    if isinstance(g1[key], int) and isinstance(g2[key], int):
                        g3[key] = single_point_crossover(g1[key], g2[key])
                        continue
                elif crossover_method == 'uniform':
                    if isinstance(g1[key], float) and isinstance(g2[key], float):
                        # print(f'{g1[key]=}')
                        # print(f'{g2[key]=}')
                        split1 = g1[key] // 1, g1[key] % 1
                        split2 = g2[key] // 1, g2[key] % 1
                        # print(f'{split1=}')
                        # print(f'{split2=}')
                        before_dec = uniform_crossover(int(split1[0]), int(split2[0]))
                        if split1[1] != 0.0:
                            split1 = split1[0], np.format_float_positional(split1[1])
                        if split2[1] != 0.0:
                            split2 = split2[0], np.format_float_positional(split2[1])
                        after_dec = uniform_crossover(int(str(split1[1]).split('.')[1]),
                                                      int(str(split2[1]).split('.')[1]))

                        g3[key] = float(f'{before_dec}.{after_dec}')
                        # print(f'float after uniform crossover:{g3[key]}')
                        continue
                    if isinstance(g1[key], int) and isinstance(g2[key], int):
                        if g1[key] == -1 or g2[key] == -1:
                            continue
                        g3[key] = uniform_crossover(g1[key], g2[key])
                        continue
        else:
            log.debug('No common keys')

    # To aviod "OverflowError: int too big to convert"
    for key in g3:
        if isinstance(g3[key], int):
            if g3[key] > sys.maxsize:
                g3[key] = sys.maxsize - 10

    # values above ~2146000000 will cause "OverflowError: Python int too large to convert to C long"
    if 'linearsvc__max_iter' in g3:
        if g3['linearsvc__max_iter'] > 2146000000:
            g3['linearsvc__max_iter'] = 2146000000

    if 'logisticregression__max_iter' in g3:
        if g3['logisticregression__max_iter'] > 2146000000:
            g3['logisticregression__max_iter'] = 2146000000

    # limit cache size to 8000 MB
    if 'nusvc__cache_size' in g3:
        if g3['nusvc__cache_size'] > 8000:
            g3['nusvc__cache_size'] = 8000
    if 'svc__cache_size' in g3:
        if g3['svc__cache_size'] > 8000:
            g3['svc__cache_size'] = 8000

    # limit degree to 1000
    if 'nusvc__degree' in g3:
        if g3['nusvc__degree'] > 1000:
            g3['nusvc__degree'] = 1000

    # print(f'Creating child from genome:{g3}')
    pipeline = create_pipeline(g3)
    # print(f'Created pipeline:{pipeline}')
    return Individual(pipeline=pipeline, genome=g3, validation_method=None, score=None, validation_time=None,
                      pipeline_string=str(pipeline))


def to_bit_array(n: int):
    return [1 if digit == '1' else 0 for digit in bin(n)[2:]]


def get_transformers_names(genes):
    names = set()
    for key in genes:
        if '__' in key:
            mutator_name, separator, parameter_name = key.partition('__')
            names.add(mutator_name)
    return names


# deletes transformer from genes
def del_transformer(transformer_name, genes):
    if transformer_name == 'standardscaler':
        del genes['standardscaler__copy']
        del genes['standardscaler__with_mean']
        del genes['standardscaler__with_std']
    if transformer_name == 'minmaxscaler':
        del genes['minmaxscaler__copy']
        del genes['minmaxscaler__feature_range']
    if transformer_name == 'robustscaler':
        del genes['robustscaler__copy']
        del genes['robustscaler__quantile_range']
        del genes['robustscaler__with_centering']
        del genes['robustscaler__with_scaling']
    if transformer_name == 'pca':
        del genes['pca__copy']
        del genes['pca__iterated_power']
        del genes['pca__n_components']
        del genes['pca__random_state']
        del genes['pca__svd_solver']
        del genes['pca__tol']
        del genes['pca__whiten']
    if transformer_name == 'binarizer':
        del genes['binarizer__copy']
        del genes['binarizer__threshold']
    if transformer_name == 'powertransformer':
        del genes['powertransformer__copy']
        del genes['powertransformer__method']
        del genes['powertransformer__standardize']
    if transformer_name == 'quantiletransformer':
        del genes['quantiletransformer__copy']
        del genes['quantiletransformer__ignore_implicit_zeros']
        del genes['quantiletransformer__n_quantiles']
        del genes['quantiletransformer__output_distribution']
        del genes['quantiletransformer__random_state']
        del genes['quantiletransformer__subsample']
    return genes


def add_transformer(transformer_name, genotype):
    if transformer_name == 'standardscaler':
        genotype['standardscaler__copy'] = True
        genotype['standardscaler__with_mean'] = bool(random.getrandbits(1))
        genotype['standardscaler__with_std'] = bool(random.getrandbits(1))
    if transformer_name == 'minmaxscaler':
        genotype['minmaxscaler__copy'] = True
        spectrum = [random.randint(0, 10), random.randint(0, 10)]
        while spectrum[0] == spectrum[1]:
            spectrum = [random.randint(0, 10), random.randint(0, 10)]
        genotype['minmaxscaler__feature_range'] = (min(spectrum), max(spectrum))
    if transformer_name == 'robustscaler':
        genotype['robustscaler__copy'] = True
        spectrum = [random.randint(5, 100), random.randint(5, 100)]
        while spectrum[0] == spectrum[1]:
            spectrum = [random.randint(5, 100), random.randint(5, 100)]
        genotype['robustscaler__quantile_range'] = (min(spectrum), max(spectrum))
        genotype['robustscaler__with_centering'] = bool(random.getrandbits(1))
        genotype['robustscaler__with_scaling'] = bool(random.getrandbits(1))
    if transformer_name == 'pca':
        log.debug(f'{MAX_N_COMPONENTS=}')
        svd_solvers = ['auto', 'randomized', 'full', 'arpack']
        genotype['pca__copy'] = True
        genotype['pca__iterated_power'] = random.randint(1, 20)
        #             # n_components must be strictly less than min(n_samples, n_features)=41 with svd_solver='arpack
        #             # therefore -> MAX_N_COMPONENTS - 1
        # n_components must be between 1 and min(n_samples, n_features)=6 with svd_solver='randomized'
        genotype['pca__n_components'] = random.randint(1, MAX_N_COMPONENTS - 1)
        genotype['pca__random_state'] = RANDOM_STATE
        genotype['pca__svd_solver'] = random.choice(svd_solvers)
        genotype['pca__tol'] = random.uniform(0., 2.)
        genotype['pca__whiten'] = bool(random.getrandbits(1))
    if transformer_name == 'binarizer':
        genotype['binarizer__copy'] = True
        genotype['binarizer__threshold'] = random.uniform(0., 1.)
    if transformer_name == 'powertransformer':
        genotype['powertransformer__copy'] = True
        genotype['powertransformer__method'] = 'yeo-johnson'
        # powertransformer__standardize = False causes trouble
        genotype['powertransformer__standardize'] = True
    if transformer_name == 'quantiletransformer':
        genotype['quantiletransformer__copy'] = True
        genotype['quantiletransformer__ignore_implicit_zeros'] = False
        # ValueError: The number of quantiles cannot be greater than the number of samples used. Got 357 quantiles and 166 samples.
        if (len(ORIGINAL_Y_TRAIN) // CV) > 10:
            genotype['quantiletransformer__n_quantiles'] = random.randint(10, len(ORIGINAL_Y_TRAIN) // CV)
        else:
            genotype['quantiletransformer__n_quantiles'] = random.randint(1, 5)
        # test_samples/cv -> if LeaveOneOut test_samples - 1
        genotype['quantiletransformer__output_distribution'] = 'uniform'
        genotype['quantiletransformer__random_state'] = RANDOM_STATE
        genotype['quantiletransformer__subsample'] = random.randint(10, 1000000)
    if transformer_name == 'FastICA':
        genotype['FastICA__tol'] = random.uniform(0., 1.)
        genotype['FastICA__random_state'] = RANDOM_STATE

    return genotype


def average_score(individuals: [Individual]) -> float:
    return sum(i.score for i in individuals) / len(individuals)


def mutate(genotype, mutation_rate, mutation_amount):
    # exclude unnecessary and problematic keys
    keys_to_skip = {'classifier_type', 'verbose', 'random_state', 'gpu_id', 'verbosity',
                    'standardscaler__copy', 'minmaxscaler__copy', 'robustscaler__copy', 'pca__copy', 'binarizer__copy',
                    'powertransformer__copy', 'quantiletransformer__copy',
                    'quantiletransformer__ignore_implicit_zeros',
                    'pca__n_components', 'pca__random_state',
                    'svc__random_state', 'decisiontreeclassifier__random_state',
                    'extratreeclassifier__random_state', 'randomforestclassifier__random_state',
                    'gradientboostingclassifier__random_state', 'logisticregression__random_state',
                    'gaussianprocessclassifier__random_state', 'passiveaggressiveclassifier__random_state',
                    'ridgeclassifier__random_state', 'sgdclassifier__random_state', 'adaboostclassifier__random_state',
                    'baggingclassifier__random_state', 'perceptron__random_state', 'mlpclassifier__random_state',
                    'nusvc__random_state', 'svc__random_state',
                    'linearsvc__random_state', 'xgbclassifier__random_state',
                    # KNN
                    'kneighborsclassifier__p',
                    'powertransformer__standardize',
                    # MLPC
                    #   beta must be >= 0 and <= 1
                    #   momentum must be >= 0 and <= 1, got 1.0007580111129932
                    'mlpclassifier__beta_1', 'mlpclassifier__beta_2', 'mlpclassifier__momentum',
                    # GBC
                    # ValueError: subsample must be in (0,1] but was 1.006697956381516
                    'gradientboostingclassifier__subsample',
                    # Bagging
                    'baggingclassifier__max_features',
                    # XGB
                    'xgbclassifier__nthread', 'xgbclassifier__n_jobs',
                    # The min_impurity_split parameter is deprecated ... Use the min_impurity_decrease parameter instead.
                    'randomforestclassifier__min_impurity_split', 'gradientboostingclassifier__min_impurity_split',
                    'extratreeclassifier__min_impurity_split', 'decisiontreeclassifier__min_impurity_split',
                    'bernoullinb__binarize', 'quantiletransformer__n_quantiles'
                    }

    for key in genotype:
        if key in keys_to_skip:
            continue
        # print(f'{key=}')
        # print(f'{genotype[key]=}')
        if random.random() > mutation_rate:
            continue
        if isinstance(genotype[key], tuple):
            try:
                spectrum = [random.randint(0, genotype[key][1]), random.randint(0, genotype[key][1])]
                while spectrum[0] == spectrum[1]:
                    spectrum = [random.randint(0, 10), random.randint(0, 10)]
                genotype[key] = (min(spectrum), max(spectrum))
            except IndexError:
                continue
        # REMEMBER! IN PYTHON ----> EVERY bool is int, not every int is bool
        # mutating boolean parameters often cause problems
        if isinstance(genotype[key], bool):
            continue
        if isinstance(genotype[key], float):
            if genotype[key] > 1.0:
                genotype[key] += abs(genotype[key] * (np.random.uniform(-0.5, 0.5) * mutation_amount))
            else:
                if genotype[key] == 0.:
                    genotype[key] = np.random.uniform(0.00, 0.1)
                if genotype[key] != 0.0:
                    genotype[key] += genotype[key] * (np.random.uniform(-0.1, 0.1) * mutation_amount)
                    if genotype[key] <= 0.:
                        genotype[key] = np.random.uniform(0.01, 0.1)
                    # in case the upper limit of value is less than 1.0
                    while genotype[key] >= 1.:
                        genotype[key] += genotype[key] * (np.random.uniform(-0.05, -0.01) * mutation_amount)
            # print(f'After: {genotype[key]=}')
            continue
        if isinstance(genotype[key], int):
            if genotype[key] > 0:
                genotype[key] += abs(int(genotype[key] * (np.random.uniform(-0.2, 0.2)) * mutation_amount))
            else:
                genotype[key] += int(genotype[key] * (np.random.uniform(-0.2, 0.2)) * mutation_amount)
                continue
            # To aviod "OverflowError: int too big to convert"
            if genotype[key] > sys.maxsize:
                genotype[key] = sys.maxsize - 10

    # drop random transformer
    if random.random() < mutation_rate / 5:
        transformers_names = get_transformers_names(genotype)
        if transformers_names:
            transformer_name = random.choice(list(transformers_names))
            genotype = del_transformer(transformer_name, genotype)

    # add random transformer
    if random.random() < mutation_rate / 5:
        random_name = random.choice(TRANSFORMERS_NAMES)
        if random_name not in genotype:
            log.debug(f'Random transformer name:{random_name}')
            genotype = add_transformer(random_name, genotype)

    # values above ~2146000000 will cause "OverflowError: Python int too large to convert to C long"
    if 'linearsvc__max_iter' in genotype:
        if genotype['linearsvc__max_iter'] > 2146000000:
            genotype['linearsvc__max_iter'] = 2146000000

    if 'logisticregression__max_iter' in genotype:
        if genotype['logisticregression__max_iter'] > 2146000000:
            genotype['logisticregression__max_iter'] = 2146000000

    # limit cache size to 8000 MB
    if 'nusvc__cache_size' in genotype:
        if genotype['nusvc__cache_size'] > 4000:
            genotype['nusvc__cache_size'] = 4000
    if 'svc__cache_size' in genotype:
        if genotype['svc__cache_size'] > 4000:
            genotype['svc__cache_size'] = 4000

    # limit degree to 1000
    if 'nusvc__degree' in genotype:
        if genotype['nusvc__degree'] > 1000:
            genotype['nusvc__degree'] = 1000

    return genotype


def get_random_classifier():
    return random.choice(CLASSIFIERS)


def generate_all_individuals():
    all_individuals = []
    for x in CLASSIFIERS:
        all_individuals.append(x)
    return all_individuals


def update_param_grid_big(clf, param_grid):
    if isinstance(clf, BernoulliNB):
        param_grid.update(dict(
            bernoullinb__alpha=[0.1, 1.0],
            bernoullinb__binarize=[None]))
        # bernoullinb__binarize=np.linspace(start=0.0, stop=1.0, num=3)))

    if isinstance(clf, GaussianNB):
        param_grid.update(dict(
            gaussiannb__var_smoothing=np.linspace(start=1e-14, stop=1e-1, num=3)))

    if isinstance(clf, KNeighborsClassifier):
        param_grid.update({
            # Expected n_neighbors <= n_samples.
            # 'kd_tree' causes -> DataConversionWarning: A column-vector y was passed when a 1d array was expected
            'kneighborsclassifier__n_neighbors': np.arange(start=1, stop=MAX_N_COMPONENTS,
                                                           step=5),
            'kneighborsclassifier__algorithm': ['auto', 'ball_tree', 'brute'],
            'kneighborsclassifier__leaf_size': np.arange(start=1, stop=500, step=5),
            'kneighborsclassifier__p': [2],
            'kneighborsclassifier__metric': ['minkowski'],
            'kneighborsclassifier__weights': ['uniform', 'distance']
        })

        # WARNING! isinstance(extra_trees_c_instance, DecisionTreeClassifier) -> True
    if str(clf).__contains__('DecisionTreeClassifier'):
        criteria = ["gini", "entropy"]
        splitter_list = ["best", "random"]
        class_weights = [None, "balanced"]

        param_grid.update({
            'decisiontreeclassifier__criterion': criteria,
            'decisiontreeclassifier__splitter': splitter_list,
            'decisiontreeclassifier__max_depth': [None, 100],
            'decisiontreeclassifier__min_samples_split': [2, 100],
            'decisiontreeclassifier__min_samples_leaf': [1, 100],
            'decisiontreeclassifier__min_weight_fraction_leaf': [0.0, 0.3],
            'decisiontreeclassifier__max_features': ['auto', 'sqrt', 'log2'],
            'decisiontreeclassifier__random_state': [RANDOM_STATE],
            'decisiontreeclassifier__max_leaf_nodes': [None, 100],
            'decisiontreeclassifier__min_impurity_decrease': [0., 0.2],
            # The min_impurity_split parameter is deprecated.
            # The min_impurity_split parameter is deprecated. Its default value has changed from 1e-7 to 0 in version 0.23, and it will be removed in 1.0 (renaming of 0.25). Use the min_impurity_decrease parameter instead.
            # 'decisiontreeclassifier__min_impurity_split': [None],
            'decisiontreeclassifier__class_weight': class_weights,
            'decisiontreeclassifier__ccp_alpha': [0.0, 0.2],
        })

    if str(clf).__contains__('ExtraTreeClassifier'):
        max_features = ["sqrt", "log2", np.random.uniform(0.01, 0.99), None]
        criteria = ["gini", "entropy"]
        splitter_list = ["best", "random"]
        class_weights = [None, "balanced"]

        param_grid.update({
            'extratreeclassifier__criterion': criteria,
            'extratreeclassifier__splitter': splitter_list,
            'extratreeclassifier__max_depth': [None, 100],
            'extratreeclassifier__min_samples_split': [2, 100],
            'extratreeclassifier__min_samples_leaf': [1, 100],
            'extratreeclassifier__min_weight_fraction_leaf': [0.0, 0.3],
            'extratreeclassifier__max_features': ['auto', 'sqrt', 'log2', 0.05, 1.01, 0.05],
            'extratreeclassifier__random_state': [RANDOM_STATE],
            'extratreeclassifier__max_leaf_nodes': [None, 100],
            'extratreeclassifier__min_impurity_decrease': [0., 0.2],
            # The min_impurity_split parameter is deprecated.
            # 'extratreeclassifier__min_impurity_split': [None],
            'extratreeclassifier__class_weight': class_weights,
            'extratreeclassifier__ccp_alpha': [0.0, 0.2],
        })

    if isinstance(clf, RandomForestClassifier):
        max_features = ["sqrt", "log2", np.random.uniform(0.01, 0.99), None]
        criteria = ["gini", "entropy"]
        class_weights = ["balanced", "balanced_subsample"]
        # class_weight presets "balanced" or "balanced_subsample" are not recommended for warm_start if the fitted data differs from the full dataset.

        param_grid.update({
            'randomforestclassifier__n_estimators': [100, 1000],
            'randomforestclassifier__criterion': criteria,
            'randomforestclassifier__max_depth': [None, 1000],
            'randomforestclassifier__min_samples_split': [2, 100],
            'randomforestclassifier__min_samples_leaf': [1, 100],
            'randomforestclassifier__min_weight_fraction_leaf': [0.0, 0.3],
            'randomforestclassifier__max_features': ['auto', 'sqrt', 'log2'],
            'randomforestclassifier__max_leaf_nodes': [None, 100],
            'randomforestclassifier__min_impurity_decrease': [0., 0.2],
            # The min_impurity_split parameter is deprecated.
            # 'randomforestclassifier__min_impurity_split': [None],
            'randomforestclassifier__bootstrap': [True, False],
            #     oob_score : bool, default=False
            #         Whether to use out-of-bag samples to estimate the generalization score.
            #         Only available if bootstrap=True.
            'randomforestclassifier__random_state': [RANDOM_STATE],
            'randomforestclassifier__warm_start': [False],
            'randomforestclassifier__class_weight': class_weights,
            'randomforestclassifier__ccp_alpha': [0.0, 0.2],
            #     max_samples : int or float, default=None
            #         If bootstrap is True, the number of samples to draw from X
            #         to train each base estimator.
            'randomforestclassifier__max_samples': [None, 20],
        })

    if isinstance(clf, GradientBoostingClassifier):
        # "criterion='mae' was deprecated in version 0.24. Use `criterion='squared_error'` which is equivalent.
        criteria = ['friedman_mse', 'squared_error']

        param_grid.update({
            'gradientboostingclassifier__loss': ['deviance', 'exponential'],
            'gradientboostingclassifier__learning_rate': [0.01, 0.1],
            'gradientboostingclassifier__n_estimators': [100, 10000],
            # subsample must be in (0,1]
            'gradientboostingclassifier__subsample': [0.1, 1.],
            'gradientboostingclassifier__criterion': criteria,
            'gradientboostingclassifier__min_samples_split': [2, 100],
            'gradientboostingclassifier__min_samples_leaf': [1, 100],
            'gradientboostingclassifier__min_weight_fraction_leaf': [0.0, 0.3],
            'gradientboostingclassifier__max_depth': [3, 10],
            'gradientboostingclassifier__min_impurity_decrease': [0., 0.2],
            # The min_impurity_split parameter is deprecated.
            # 'gradientboostingclassifier__min_impurity_split': [None, 0.1],
            'gradientboostingclassifier__random_state': [RANDOM_STATE],
            'gradientboostingclassifier__max_features': ['auto', 'sqrt', 'log2'],
            'gradientboostingclassifier__max_leaf_nodes': [None, 100],
            # n_estimators=100 must be larger or equal to estimators_.shape[0]=10000 when warm_start==True
            'gradientboostingclassifier__warm_start': [False],
            'gradientboostingclassifier__validation_fraction': [0.1, 0.2],
            'gradientboostingclassifier__n_iter_no_change': [None],
            'gradientboostingclassifier__tol': [1e-5, 1e-3],
            'gradientboostingclassifier__ccp_alpha': [0.0, 0.2]
        })

    if isinstance(clf, LogisticRegression):
        param_grid.update({'logisticregression__max_iter': [100, 10000],
                           'logisticregression__tol': [0.001, 0.2],
                           # solver saga = not wroking
                           'logisticregression__solver': ['lbfgs', 'newton-cg', 'sag'],
                           # solver='liblinear' does not support a multinomial backend
                           # warning! It's 'none', not None!
                           'logisticregression__penalty': ['l2', 'none'],
                           'logisticregression__multi_class': ['multinomial', 'auto'],  # ovr
                           'logisticregression__random_state': [RANDOM_STATE],
                           #  Invalid parameter max_squared_sum for estimator LogisticRegression
                           # 'logisticregression__max_squared_sum': [None],
                           'logisticregression__l1_ratio': [None, 0.5],
                           'logisticregression__C': [1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10.],
                           'logisticregression__dual': [False]
                           })

    if isinstance(clf, GaussianProcessClassifier):
        param_grid.update({
            'gaussianprocessclassifier__optimizer': [0.1, 1.0],
            'gaussianprocessclassifier__n_restarts_optimizer': [0, 20],
            'gaussianprocessclassifier__max_iter_predict': [100, 1000, 10000],
            'gaussianprocessclassifier__multi_class': ['one_vs_rest', 'one_vs_one'],
            'gaussianprocessclassifier__copy_X_train': [True],
            'gaussianprocessclassifier__random_state': [RANDOM_STATE],
        })

        params = {'random_state': 13,
                  'max_iter_predict': random.randint(1, 200), 'n_restarts_optimizer': random.randint(0, 20)}

    if isinstance(clf, PassiveAggressiveClassifier):
        param_grid.update({
            'passiveaggressiveclassifier__C': [0.1, 1.0],
            'passiveaggressiveclassifier__fit_intercept': [True, False],
            'passiveaggressiveclassifier__max_iter': [1000, 100000],
            'passiveaggressiveclassifier__tol': [1e-4, 1e-2],
            'passiveaggressiveclassifier__early_stopping': [True, False],
            'passiveaggressiveclassifier__validation_fraction': [0.1, 0.3],
            'passiveaggressiveclassifier__n_iter_no_change': [5, 30],
            'passiveaggressiveclassifier__shuffle': [True],
            'passiveaggressiveclassifier__random_state': [RANDOM_STATE],
            'passiveaggressiveclassifier__average': [False, True]
        })

    if isinstance(clf, RidgeClassifier):
        # Current sag implementation does not handle the case step_size * alpha_scaled == 1
        param_grid.update({
            'ridgeclassifier__alpha': [0.5, 1.0],
            'ridgeclassifier__fit_intercept': [True],
            'ridgeclassifier__normalize': [False],
            'ridgeclassifier__copy_X': [True],
            'ridgeclassifier__max_iter': [None, 100000],
            'ridgeclassifier__tol': [1e-3, 1e-6, 1e-1],
            'ridgeclassifier__solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
            'ridgeclassifier__random_state': [RANDOM_STATE]
        })

    if isinstance(clf, SGDClassifier):
        param_grid.update({
            'sgdclassifier__loss': ['hinge', 'log', 'modified_huber',
                                    'squared_hinge', 'perceptron', 'squared_loss',
                                    'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
            'sgdclassifier__penalty': ['l2', 'l1', 'elasticnet'],
            'sgdclassifier__alpha': [0.0001, 0.00001, 0.001],
            'sgdclassifier__l1_ratio': [0.15, 0.01, 0.4],
            'sgdclassifier__fit_intercept': [True],
            'sgdclassifier__max_iter': [1000, 15000],
            'sgdclassifier__tol': [1e-3, 1e-5, 1e-1],
            'sgdclassifier__shuffle': [True],
            'sgdclassifier__epsilon': [0.1, 0.01, 0.001],
            'sgdclassifier__learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
            # eta0 must be > 0
            'sgdclassifier__eta0': [0.2, 0.01, 0.001],
            'sgdclassifier__power_t': [0.5, 0.1, 0.7],
            'sgdclassifier__early_stopping': [False],
            'sgdclassifier__validation_fraction': [0.1, 0.2],
            'sgdclassifier__n_iter_no_change': [5, 20],
            'sgdclassifier__average': [False, True],
            'sgdclassifier__random_state': [RANDOM_STATE]
        })

    if isinstance(clf, AdaBoostClassifier):
        param_grid.update({
            'adaboostclassifier__n_estimators': [50, 500, 1000],
            'adaboostclassifier__learning_rate': [1., 0.1, 0.5],
            'adaboostclassifier__algorithm': ['SAMME', 'SAMME.R'],
            'adaboostclassifier__random_state': [RANDOM_STATE]
        })

        # This little fellow is meta-classifier
    if isinstance(clf, BaggingClassifier):
        param_grid.update({
            'baggingclassifier__n_estimators': [10, 20, 30, 50, 200],
            # max_samples must be in (0, n_samples]
            'baggingclassifier__max_samples': [1.0, 0.5, 0.1],
            'baggingclassifier__max_features': [1.0, 0.5, 0.2],
            'baggingclassifier__random_state': [RANDOM_STATE]
        })

    if isinstance(clf, LinearDiscriminantAnalysis):
        param_grid.update({
            'lineardiscriminantanalysis__solver': ['svd', 'lsqr'],  # 'eigen' causes trouble
            'lineardiscriminantanalysis__shrinkage': [None],
            # 'svd', is not compatible with shrinkage | 'auto', 0.1, 0.2
            'lineardiscriminantanalysis__n_components': [None],  # min(n_features, n_classes - 1)
            'lineardiscriminantanalysis__store_covariance': [False, True],
            'lineardiscriminantanalysis__tol': [1e-6, 1e-2],
        })

    if isinstance(clf, NearestCentroid):
        param_grid.update({
            'nearestcentroid__metric': ['euclidean', 'manhattan'],
            'nearestcentroid__shrink_threshold': [None, 0.1, 0.2, 0.9]
        })

    if isinstance(clf, Perceptron):
        param_grid.update({
            'perceptron__penalty': ['l2', 'l1', 'elasticnet'],
            'perceptron__alpha': [0.0001, 0.01, 0.1],
            'perceptron__l1_ratio': [0.15, 0.05, 0.3],
            'perceptron__fit_intercept': [True],
            'perceptron__max_iter': [1000, 50000],
            'perceptron__tol': [1e-3, 1e-7, 1e-1],
            'perceptron__shuffle': [True],
            'perceptron__eta0': [1., 1.2, 0.8],
            'perceptron__random_state': [RANDOM_STATE],
            'perceptron__validation_fraction': [0.1, 0.2],
            'perceptron__n_iter_no_change': [5, 20]
        })

    if isinstance(clf, MLPClassifier):
        layers_num = random.randint(1, 3)
        layers = []
        for x in range(layers_num):
            layers.append(random.randint(1, 100))
        param_grid.update({
            'mlpclassifier__hidden_layer_sizes': tuple(layers),
            'mlpclassifier__activation': ['identity', 'logistic', 'tanh', 'relu'],
            'mlpclassifier__solver': ['lbfgs', 'sgd', 'adam'],
            'mlpclassifier__alpha': [0.0001, 0.1, 0.00001],
            'mlpclassifier__learning_rate': ['constant', 'invscaling', 'adaptive'],
            'mlpclassifier__learning_rate_init': [0.0001, 0.01, 0.00001],  # 0.0000001 is too little
            'mlpclassifier__power_t': [0.5, 0.1, 0.8],
            'mlpclassifier__max_iter': [200, 20000],  # 2000000 is too much
            'mlpclassifier__shuffle': [True],
            'mlpclassifier__random_state': [RANDOM_STATE],
            'mlpclassifier__tol': [1e-4, 1e-8, 1e-2],
            'mlpclassifier__momentum': [0.9, 0.95, 0.6],
            'mlpclassifier__early_stopping': [False],
            'mlpclassifier__validation_fraction': [0.1, 0.2],
            'mlpclassifier__beta_1': [0.9, 0.8],
            'mlpclassifier__beta_2': [0.999, 0.8],
            'mlpclassifier__n_iter_no_change': [100],
            'mlpclassifier__max_fun': [15000, 30000]
        })

    if isinstance(clf, LinearSVC):
        #  Unsupported set of arguments:
        #  The combination of penalty='l1' and loss='squared_hinge'
        #  are not supported when dual=True, Parameters: penalty='l1', loss='squared_hinge', dual=True
        # The combination of penalty='l2' and loss='hinge' are not supported when dual=False,
        # Parameters: penalty='l2', loss='hinge', dual=False
        # Parameters: penalty='l1', loss='hinge', dual=False -> also not supported
        param_grid.update({
            'linearsvc__penalty': ['l2'],  # 'l1',
            'linearsvc__loss': ['hinge', 'squared_hinge'],
            'linearsvc__dual': [True],
            'linearsvc__tol': [1e-6, 1e-4, 1e-2],
            'linearsvc__C': [1., 0.5],
            'linearsvc__multi_class': ['ovr', 'crammer_singer'],
            'linearsvc__random_state': [RANDOM_STATE],
            'linearsvc__max_iter': [1000, 5000, 50000],
        })
    if isinstance(clf, NuSVC):
        param_grid.update({
            'nusvc__nu': [0.1, 0.5],
            # precomputed kernel requires the kernel matrix
            'nusvc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'nusvc__degree': [2, 3],
            'nusvc__gamma': ['scale', 'auto'],
            'nusvc__coef0': [0., 0.1],
            'nusvc__shrinking': [True, False],
            'nusvc__probability': [False, True],
            'nusvc__tol': [1e-3, 1e-7, 1e-1],
            'nusvc__cache_size': [200, 1000],
            'nusvc__max_iter': [-1],
            'nusvc__decision_function_shape': ['ovo', 'ovr'],
            # break_ties must be False when decision_function_shape is 'ovo'
            'nusvc__break_ties': [False],
            'nusvc__random_state': [RANDOM_STATE],
        })

    if isinstance(clf, SVC):
        # O(n_samples^2 * n_features)
        # TODO - gamma, coef0
        param_grid.update({
            'svc__C': np.linspace(start=1e-14, stop=1e2, num=3),
            'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'svc__degree': [2, 3],
            'svc__random_state': [RANDOM_STATE],
            # precomputed requires the kernel matrix
            'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid']
        })

    if isinstance(clf, XGBClassifier):
        param_grid.update({
            # 'xgbclassifier__early_stopping_rounds': [2],
            'xgbclassifier__n_estimators': [2, 10, 20],
            'xgbclassifier__learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
            # 'xgbclassifier__subsample': np.arange(0.05, 1.01, 0.05),

            'xgbclassifier__random_state': [RANDOM_STATE],
            'xgbclassifier__booster': ['gbtree', 'gblinear', 'dart'],
            # max_depth can not be greater than 31 as that might generate 2 ** 32 - 1 nodes
            'xgbclassifier__max_depth': [6, 20],
            'xgbclassifier__n_jobs': [N_JOBS],
            # missing = np.nan
            'xgbclassifier__eta': [0.3, 0.1, 0.01],
            # on GTX1060-6GB 'gpu_hist' works few times slower, at least on <5MB datasets
            'xgbclassifier__tree_method': ['auto'],  # auto, exact, approx, hist, gpu_hist
            'xgbclassifier__use_label_encoder': [False]
        })
    return param_grid


def update_param_grid_minimal(clf, param_grid):
    if isinstance(clf, BernoulliNB):
        param_grid.update(dict(
            bernoullinb__alpha=[0.1, 1.0],
            bernoullinb__binarize=[None]))

    if isinstance(clf, GaussianNB):
        param_grid.update(dict(
            gaussiannb__var_smoothing=np.linspace(start=1e-14, stop=1e-1, num=3)))

    if isinstance(clf, KNeighborsClassifier):
        param_grid.update({
            'kneighborsclassifier__n_neighbors': np.arange(start=1, stop=MAX_N_COMPONENTS,
                                                           step=2),
            'kneighborsclassifier__algorithm': ['auto', 'ball_tree', 'brute'],
            'kneighborsclassifier__leaf_size': np.arange(start=1, stop=500, step=2),
            'kneighborsclassifier__p': [2],
            'kneighborsclassifier__metric': ['minkowski'],
            'kneighborsclassifier__weights': ['uniform', 'distance']
        })

        # WARNING! isinstance(extra_trees_c_instance, DecisionTreeClassifier) -> True
    if str(clf).__contains__('DecisionTreeClassifier'):
        criteria = ["gini", "entropy"]
        splitter_list = ["best", "random"]
        class_weights = [None, "balanced"]

        param_grid.update({
            'decisiontreeclassifier__criterion': criteria,
            'decisiontreeclassifier__splitter': splitter_list,
            'decisiontreeclassifier__max_depth': [None, 100],
            'decisiontreeclassifier__min_samples_split': [2, 100],
            'decisiontreeclassifier__min_samples_leaf': [1, 100],
            'decisiontreeclassifier__min_weight_fraction_leaf': [0.0, 0.3],
            'decisiontreeclassifier__max_features': ['auto', 'sqrt', 'log2'],
            'decisiontreeclassifier__random_state': [RANDOM_STATE],
            'decisiontreeclassifier__max_leaf_nodes': [None, 100],
            'decisiontreeclassifier__min_impurity_decrease': [0., 0.2],
            'decisiontreeclassifier__class_weight': class_weights,
            'decisiontreeclassifier__ccp_alpha': [0.0, 0.2],
        })

    if str(clf).__contains__('ExtraTreeClassifier'):
        max_features = ["sqrt", "log2", np.random.uniform(0.01, 0.99), None]
        criteria = ["gini", "entropy"]
        splitter_list = ["best", "random"]
        class_weights = [None, "balanced"]
        param_grid.update({
            'extratreeclassifier__criterion': criteria,
            'extratreeclassifier__splitter': splitter_list,
            'extratreeclassifier__max_depth': [None, 100],
            'extratreeclassifier__min_samples_split': [2, 100],
            'extratreeclassifier__min_samples_leaf': [1, 100],
            'extratreeclassifier__min_weight_fraction_leaf': [0.0, 0.3],
            'extratreeclassifier__max_features': ['auto', 'sqrt', 'log2', 0.05, 1.01, 0.05],
            'extratreeclassifier__random_state': [RANDOM_STATE],
            'extratreeclassifier__max_leaf_nodes': [None, 100],
            'extratreeclassifier__min_impurity_decrease': [0., 0.2],
            'extratreeclassifier__class_weight': class_weights,
            'extratreeclassifier__ccp_alpha': [0.0, 0.2],
        })

    if isinstance(clf, RandomForestClassifier):
        criteria = ["gini", "entropy"]
        class_weights = ["balanced", "balanced_subsample"]

        param_grid.update({
            'randomforestclassifier__n_estimators': [10, 100],
            'randomforestclassifier__criterion': criteria,
            'randomforestclassifier__max_depth': [None, 1000],
            'randomforestclassifier__min_samples_split': [2, 100],
            'randomforestclassifier__min_samples_leaf': [1, 100],
            'randomforestclassifier__min_weight_fraction_leaf': [0.0, 0.3],
            'randomforestclassifier__max_features': ['auto', 'sqrt', 'log2'],
            'randomforestclassifier__max_leaf_nodes': [None, 100],
            'randomforestclassifier__min_impurity_decrease': [0., 0.2],
            # The min_impurity_split parameter is deprecated.
            # 'randomforestclassifier__min_impurity_split': [None],
            'randomforestclassifier__bootstrap': [True, False],
            #     oob_score : bool, default=False
            #         Whether to use out-of-bag samples to estimate the generalization score.
            #         Only available if bootstrap=True.
            'randomforestclassifier__random_state': [RANDOM_STATE],
            'randomforestclassifier__warm_start': [False],
            'randomforestclassifier__class_weight': class_weights,
            'randomforestclassifier__ccp_alpha': [0.0, 0.2],
            #     max_samples : int or float, default=None
            #         If bootstrap is True, the number of samples to draw from X
            #         to train each base estimator.
            'randomforestclassifier__max_samples': [None, 20],
        })

    if isinstance(clf, GradientBoostingClassifier):
        # "criterion='mae' was deprecated in version 0.24. Use `criterion='squared_error'` which is equivalent.
        criteria = ['friedman_mse', 'squared_error']

        param_grid.update({
            'gradientboostingclassifier__loss': ['deviance', 'exponential'],
            'gradientboostingclassifier__learning_rate': [0.01, 0.1],
            'gradientboostingclassifier__n_estimators': [10, 100],
            'gradientboostingclassifier__subsample': [0.1, 1.],
            'gradientboostingclassifier__criterion': criteria,
            'gradientboostingclassifier__min_samples_split': [2, 100],
            'gradientboostingclassifier__min_samples_leaf': [1, 100],
            'gradientboostingclassifier__min_weight_fraction_leaf': [0.0, 0.3],
            'gradientboostingclassifier__max_depth': [3, 10],
            'gradientboostingclassifier__min_impurity_decrease': [0., 0.2],
            'gradientboostingclassifier__random_state': [RANDOM_STATE],
            'gradientboostingclassifier__max_features': ['auto', 'sqrt', 'log2'],
            'gradientboostingclassifier__max_leaf_nodes': [None, 100],
            'gradientboostingclassifier__warm_start': [False],
            'gradientboostingclassifier__validation_fraction': [0.1, 0.2],
            'gradientboostingclassifier__n_iter_no_change': [None],
            'gradientboostingclassifier__tol': [1e-5, 1e-3],
            'gradientboostingclassifier__ccp_alpha': [0.0, 0.2]
        })

    if isinstance(clf, LogisticRegression):
        param_grid.update({'logisticregression__max_iter': [100, 1000],
                           'logisticregression__tol': [0.001, 0.2],
                           'logisticregression__solver': ['lbfgs', 'newton-cg', 'sag'],
                           'logisticregression__penalty': ['l2', 'none'],
                           'logisticregression__multi_class': ['multinomial', 'auto'],  # ovr
                           'logisticregression__random_state': [RANDOM_STATE],
                           'logisticregression__l1_ratio': [None, 0.5],
                           'logisticregression__C': [0.2, 0.7, 1.0, 5.0],
                           'logisticregression__dual': [False]
                           })

    if isinstance(clf, GaussianProcessClassifier):
        param_grid.update({
            'gaussianprocessclassifier__optimizer': [0.1, 1.0],
            'gaussianprocessclassifier__n_restarts_optimizer': [0, 10],
            'gaussianprocessclassifier__max_iter_predict': [100, 1000],
            'gaussianprocessclassifier__multi_class': ['one_vs_rest', 'one_vs_one'],
            'gaussianprocessclassifier__copy_X_train': [True],
            'gaussianprocessclassifier__random_state': [RANDOM_STATE],
        })

        params = {'random_state': 13,
                  'max_iter_predict': random.randint(1, 200), 'n_restarts_optimizer': random.randint(0, 20)}

    if isinstance(clf, PassiveAggressiveClassifier):
        param_grid.update({
            'passiveaggressiveclassifier__C': [0.1, 1.0],
            'passiveaggressiveclassifier__fit_intercept': [True, False],
            'passiveaggressiveclassifier__max_iter': [10, 100],
            'passiveaggressiveclassifier__tol': [1e-4, 1e-2],
            'passiveaggressiveclassifier__early_stopping': [True, False],
            'passiveaggressiveclassifier__validation_fraction': [0.1, 0.3],
            'passiveaggressiveclassifier__n_iter_no_change': [5, 30],
            'passiveaggressiveclassifier__shuffle': [True],
            'passiveaggressiveclassifier__random_state': [RANDOM_STATE],
            'passiveaggressiveclassifier__average': [False, True]
        })

    if isinstance(clf, RidgeClassifier):
        param_grid.update({
            'ridgeclassifier__alpha': [0.5, 1.0],
            'ridgeclassifier__fit_intercept': [True],
            'ridgeclassifier__normalize': [False],
            'ridgeclassifier__copy_X': [True],
            'ridgeclassifier__max_iter': [100, 10000],
            'ridgeclassifier__tol': [1e-3, 1e-6, 1e-1],
            'ridgeclassifier__solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
            'ridgeclassifier__random_state': [RANDOM_STATE]
        })

    if isinstance(clf, SGDClassifier):
        param_grid.update({
            'sgdclassifier__loss': ['hinge', 'log', 'modified_huber',
                                    'squared_hinge', 'perceptron', 'squared_loss',
                                    'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
            'sgdclassifier__penalty': ['l2', 'l1', 'elasticnet'],
            'sgdclassifier__alpha': [0.0001, 0.00001, 0.001],
            'sgdclassifier__l1_ratio': [0.15, 0.01, 0.4],
            'sgdclassifier__fit_intercept': [True],
            'sgdclassifier__max_iter': [100, 1500],
            'sgdclassifier__tol': [1e-3, 1e-5, 1e-1],
            'sgdclassifier__shuffle': [True],
            'sgdclassifier__epsilon': [0.1, 0.01, 0.001],
            'sgdclassifier__learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
            # eta0 must be > 0
            'sgdclassifier__eta0': [0.2, 0.01, 0.001],
            'sgdclassifier__power_t': [0.5, 0.1, 0.7],
            'sgdclassifier__early_stopping': [False],
            'sgdclassifier__validation_fraction': [0.1, 0.2],
            'sgdclassifier__n_iter_no_change': [5, 20],
            'sgdclassifier__average': [False, True],
            'sgdclassifier__random_state': [RANDOM_STATE]
        })

    if isinstance(clf, AdaBoostClassifier):
        param_grid.update({
            'adaboostclassifier__n_estimators': [10, 100],
            'adaboostclassifier__learning_rate': [1., 0.1, 0.5],
            'adaboostclassifier__algorithm': ['SAMME', 'SAMME.R'],
            'adaboostclassifier__random_state': [RANDOM_STATE]
        })

    if isinstance(clf, BaggingClassifier):
        param_grid.update({
            'baggingclassifier__n_estimators': [10, 20, 30],
            'baggingclassifier__max_samples': [1.0, 0.5, 0.1],
            'baggingclassifier__max_features': [1.0, 0.5, 0.2],
            'baggingclassifier__random_state': [RANDOM_STATE]
        })

    if isinstance(clf, LinearDiscriminantAnalysis):
        param_grid.update({
            'lineardiscriminantanalysis__solver': ['svd', 'lsqr'],  # 'eigen' causes trouble
            'lineardiscriminantanalysis__shrinkage': [None],
            'lineardiscriminantanalysis__n_components': [None],
            'lineardiscriminantanalysis__store_covariance': [False, True],
            'lineardiscriminantanalysis__tol': [1e-6, 1e-2],
        })

    if isinstance(clf, NearestCentroid):
        param_grid.update({
            'nearestcentroid__metric': ['euclidean', 'manhattan'],
            'nearestcentroid__shrink_threshold': [None, 0.1, 0.2, 0.9]
        })

    if isinstance(clf, Perceptron):
        param_grid.update({
            'perceptron__penalty': ['l2', 'l1', 'elasticnet'],
            'perceptron__alpha': [0.0001, 0.01, 0.1],
            'perceptron__l1_ratio': [0.15, 0.05, 0.3],
            'perceptron__fit_intercept': [True],
            'perceptron__max_iter': [1000, 5000],
            'perceptron__tol': [1e-3, 1e-7, 1e-1],
            'perceptron__shuffle': [True],
            'perceptron__eta0': [1., 1.2, 0.8],
            'perceptron__random_state': [RANDOM_STATE],
            'perceptron__validation_fraction': [0.1, 0.2],
            'perceptron__n_iter_no_change': [5, 20]
        })

    if isinstance(clf, MLPClassifier):
        layers_num = random.randint(1, 2)
        layers = []
        for x in range(layers_num):
            layers.append(random.randint(1, 100))
        param_grid.update({
            'mlpclassifier__hidden_layer_sizes': tuple(layers),
            'mlpclassifier__activation': ['identity', 'logistic', 'tanh', 'relu'],
            'mlpclassifier__solver': ['lbfgs', 'sgd', 'adam'],
            'mlpclassifier__alpha': [0.0001, 0.1, 0.00001],
            'mlpclassifier__learning_rate': ['constant', 'invscaling', 'adaptive'],
            'mlpclassifier__learning_rate_init': [0.0001, 0.01],
            'mlpclassifier__power_t': [0.5, 0.1, 0.8],
            'mlpclassifier__max_iter': [200, 2000],
            'mlpclassifier__shuffle': [True],
            'mlpclassifier__random_state': [RANDOM_STATE],
            'mlpclassifier__tol': [1e-4, 1e-8, 1e-2],
            'mlpclassifier__momentum': [0.9, 0.95, 0.6],
            'mlpclassifier__early_stopping': [False],
            'mlpclassifier__validation_fraction': [0.1, 0.2],
            'mlpclassifier__beta_1': [0.9, 0.8],
            'mlpclassifier__beta_2': [0.999, 0.8],
            'mlpclassifier__n_iter_no_change': [100],
            'mlpclassifier__max_fun': [15000, 30000]
        })

    if isinstance(clf, LinearSVC):
        param_grid.update({
            'linearsvc__penalty': ['l2'],  # 'l1',
            'linearsvc__loss': ['hinge', 'squared_hinge'],
            'linearsvc__dual': [True],
            'linearsvc__tol': [1e-6, 1e-4, 1e-2],
            'linearsvc__C': [1., 0.5],
            'linearsvc__multi_class': ['ovr', 'crammer_singer'],
            'linearsvc__random_state': [RANDOM_STATE],
            'linearsvc__max_iter': [100, 500, 5000],
        })
    if isinstance(clf, NuSVC):
        param_grid.update({
            'nusvc__nu': [0.1, 0.5],
            'nusvc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'nusvc__degree': [2, 3],
            'nusvc__gamma': ['scale', 'auto'],
            'nusvc__coef0': [0., 0.1],
            'nusvc__shrinking': [True, False],
            'nusvc__probability': [False, True],
            'nusvc__tol': [1e-3, 1e-7, 1e-1],
            'nusvc__cache_size': [200, 1000],
            'nusvc__max_iter': [-1],
            'nusvc__decision_function_shape': ['ovo', 'ovr'],
            'nusvc__break_ties': [False],
            'nusvc__random_state': [RANDOM_STATE],
        })

    if isinstance(clf, SVC):
        param_grid.update({
            'svc__C': np.linspace(start=1e-14, stop=1e2, num=3),
            'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'svc__degree': [2, 3],
            'svc__random_state': [RANDOM_STATE],
            'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid']
        })

    if isinstance(clf, XGBClassifier):
        param_grid.update({
            'xgbclassifier__n_estimators': [2, 10],
            'xgbclassifier__learning_rate': [1e-2, 1e-1, 0.5, 1.],
            'xgbclassifier__random_state': [RANDOM_STATE],
            'xgbclassifier__booster': ['gbtree', 'gblinear', 'dart'],
            'xgbclassifier__max_depth': [6, 20],
            'xgbclassifier__n_jobs': [N_JOBS],
            'xgbclassifier__eta': [0.3, 0.1, 0.01],
            'xgbclassifier__tree_method': ['auto'],
            'xgbclassifier__use_label_encoder': [False]
        })
    # print(f'param_grid=}')
    return param_grid


def update_param_grid_extreme(clf, param_grid):
    if isinstance(clf, BernoulliNB):
        param_grid.update(dict(
            bernoullinb__alpha=[0.1, 1.0],
            bernoullinb__binarize=[None]))

    if isinstance(clf, GaussianNB):
        param_grid.update(dict(
            gaussiannb__var_smoothing=np.linspace(start=1e-14, stop=1e-1, num=5)))

    if isinstance(clf, KNeighborsClassifier):
        param_grid.update({
            'kneighborsclassifier__n_neighbors': np.arange(start=1, stop=MAX_N_COMPONENTS,
                                                           step=5),
            'kneighborsclassifier__algorithm': ['auto', 'ball_tree', 'brute'],
            'kneighborsclassifier__leaf_size': np.arange(start=1, stop=500, step=5),
            'kneighborsclassifier__p': [2],
            'kneighborsclassifier__metric': ['minkowski'],
            'kneighborsclassifier__weights': ['uniform', 'distance']
        })

        # WARNING! isinstance(extra_trees_c_instance, DecisionTreeClassifier) -> True
    if str(clf).__contains__('DecisionTreeClassifier'):
        criteria = ["gini", "entropy"]
        splitter_list = ["best", "random"]
        class_weights = [None, "balanced"]

        param_grid.update({
            'decisiontreeclassifier__criterion': criteria,
            'decisiontreeclassifier__splitter': splitter_list,
            'decisiontreeclassifier__max_depth': [None, 100, 10000],
            'decisiontreeclassifier__min_samples_split': [2, 100],
            'decisiontreeclassifier__min_samples_leaf': [1, 100],
            'decisiontreeclassifier__min_weight_fraction_leaf': [0.0, 0.3],
            'decisiontreeclassifier__max_features': ['auto', 'sqrt', 'log2'],
            'decisiontreeclassifier__random_state': [RANDOM_STATE],
            'decisiontreeclassifier__max_leaf_nodes': [None, 100],
            'decisiontreeclassifier__min_impurity_decrease': [0., 0.2],
            'decisiontreeclassifier__class_weight': class_weights,
            'decisiontreeclassifier__ccp_alpha': [0.0, 0.2],
        })

    if str(clf).__contains__('ExtraTreeClassifier'):
        criteria = ["gini", "entropy"]
        splitter_list = ["best", "random"]
        class_weights = [None, "balanced"]
        param_grid.update({
            'extratreeclassifier__criterion': criteria,
            'extratreeclassifier__splitter': splitter_list,
            'extratreeclassifier__max_depth': [None, 100, 10000],
            'extratreeclassifier__min_samples_split': [2, 100],
            'extratreeclassifier__min_samples_leaf': [1, 100],
            'extratreeclassifier__min_weight_fraction_leaf': [0.0, 0.3],
            'extratreeclassifier__max_features': ['auto', 'sqrt', 'log2', 0.05, 1.01, 0.05],
            'extratreeclassifier__random_state': [RANDOM_STATE],
            'extratreeclassifier__max_leaf_nodes': [None, 100],
            'extratreeclassifier__min_impurity_decrease': [0., 0.2],
            'extratreeclassifier__class_weight': class_weights,
            'extratreeclassifier__ccp_alpha': [0.0, 0.2],
        })

    if isinstance(clf, RandomForestClassifier):
        criteria = ["gini", "entropy"]
        class_weights = ["balanced", "balanced_subsample"]

        param_grid.update({
            'randomforestclassifier__n_estimators': [10, 100, 10000],
            'randomforestclassifier__criterion': criteria,
            'randomforestclassifier__max_depth': [None, 1000, 1000],
            'randomforestclassifier__min_samples_split': [2, 100],
            'randomforestclassifier__min_samples_leaf': [1, 100],
            'randomforestclassifier__min_weight_fraction_leaf': [0.0, 0.3],
            'randomforestclassifier__max_features': ['auto', 'sqrt', 'log2'],
            'randomforestclassifier__max_leaf_nodes': [None, 100],
            'randomforestclassifier__min_impurity_decrease': [0., 0.2],
            # The min_impurity_split parameter is deprecated.
            # 'randomforestclassifier__min_impurity_split': [None],
            'randomforestclassifier__bootstrap': [True, False],
            #     oob_score : bool, default=False
            #         Whether to use out-of-bag samples to estimate the generalization score.
            #         Only available if bootstrap=True.
            'randomforestclassifier__random_state': [RANDOM_STATE],
            'randomforestclassifier__warm_start': [False],
            'randomforestclassifier__class_weight': class_weights,
            'randomforestclassifier__ccp_alpha': [0.0, 0.2],
            #     max_samples : int or float, default=None
            #         If bootstrap is True, the number of samples to draw from X
            #         to train each base estimator.
            'randomforestclassifier__max_samples': [None, 20],
        })

    if isinstance(clf, GradientBoostingClassifier):
        # "criterion='mae' was deprecated in version 0.24. Use `criterion='squared_error'` which is equivalent.
        criteria = ['friedman_mse', 'squared_error']

        param_grid.update({
            'gradientboostingclassifier__loss': ['deviance', 'exponential'],
            'gradientboostingclassifier__learning_rate': [0.001, 0.01, 0.1],
            'gradientboostingclassifier__n_estimators': [10, 100, 1000],
            'gradientboostingclassifier__subsample': [0.1, 1.],
            'gradientboostingclassifier__criterion': criteria,
            'gradientboostingclassifier__min_samples_split': [2, 100],
            'gradientboostingclassifier__min_samples_leaf': [1, 100],
            'gradientboostingclassifier__min_weight_fraction_leaf': [0.0, 0.3],
            'gradientboostingclassifier__max_depth': [3, 10],
            'gradientboostingclassifier__min_impurity_decrease': [0., 0.2],
            'gradientboostingclassifier__random_state': [RANDOM_STATE],
            'gradientboostingclassifier__max_features': ['auto', 'sqrt', 'log2'],
            'gradientboostingclassifier__max_leaf_nodes': [None, 100],
            'gradientboostingclassifier__warm_start': [False],
            'gradientboostingclassifier__validation_fraction': [0.1, 0.2],
            'gradientboostingclassifier__n_iter_no_change': [None],
            'gradientboostingclassifier__tol': [1e-5, 1e-3],
            'gradientboostingclassifier__ccp_alpha': [0.0, 0.2]
        })

    if isinstance(clf, LogisticRegression):
        param_grid.update({'logisticregression__max_iter': [100000, 500000, 5000000],
                           'logisticregression__tol': [0.001, 0.2],
                           'logisticregression__solver': ['lbfgs', 'newton-cg', 'sag'],
                           'logisticregression__penalty': ['l2', 'none'],
                           'logisticregression__multi_class': ['multinomial', 'auto'],  # ovr
                           'logisticregression__random_state': [RANDOM_STATE],
                           'logisticregression__l1_ratio': [None, 0.5],
                           'logisticregression__C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 150.],
                           'logisticregression__dual': [False]
                           })

    if isinstance(clf, GaussianProcessClassifier):
        param_grid.update({
            'gaussianprocessclassifier__optimizer': [0.1, 1.0],
            'gaussianprocessclassifier__n_restarts_optimizer': [0, 10],
            'gaussianprocessclassifier__max_iter_predict': [100, 1000, 10000],
            'gaussianprocessclassifier__multi_class': ['one_vs_rest', 'one_vs_one'],
            'gaussianprocessclassifier__copy_X_train': [True],
            'gaussianprocessclassifier__random_state': [RANDOM_STATE],
        })

        params = {'random_state': 13,
                  'max_iter_predict': random.randint(1, 200), 'n_restarts_optimizer': random.randint(0, 20)}

    if isinstance(clf, PassiveAggressiveClassifier):
        param_grid.update({
            'passiveaggressiveclassifier__C': [0.1, 1.0],
            'passiveaggressiveclassifier__fit_intercept': [True, False],
            'passiveaggressiveclassifier__max_iter': [10, 100, 10000],
            'passiveaggressiveclassifier__tol': [1e-4, 1e-2],
            'passiveaggressiveclassifier__early_stopping': [True, False],
            'passiveaggressiveclassifier__validation_fraction': [0.1, 0.3],
            'passiveaggressiveclassifier__n_iter_no_change': [5, 30],
            'passiveaggressiveclassifier__shuffle': [True],
            'passiveaggressiveclassifier__random_state': [RANDOM_STATE],
            'passiveaggressiveclassifier__average': [False, True]
        })

    if isinstance(clf, RidgeClassifier):
        param_grid.update({
            'ridgeclassifier__alpha': [0.5, 1.0],
            'ridgeclassifier__fit_intercept': [True],
            'ridgeclassifier__normalize': [False],
            'ridgeclassifier__copy_X': [True],
            'ridgeclassifier__max_iter': [100, 10000, 100000],
            'ridgeclassifier__tol': [1e-3, 1e-6, 1e-1],
            'ridgeclassifier__solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
            'ridgeclassifier__random_state': [RANDOM_STATE]
        })

    if isinstance(clf, SGDClassifier):
        param_grid.update({
            'sgdclassifier__loss': ['hinge', 'log', 'modified_huber',
                                    'squared_hinge', 'perceptron', 'squared_loss',
                                    'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
            'sgdclassifier__penalty': ['l2', 'l1', 'elasticnet'],
            'sgdclassifier__alpha': [0.0001, 0.00001, 0.001],
            'sgdclassifier__l1_ratio': [0.15, 0.01, 0.4],
            'sgdclassifier__fit_intercept': [True],
            'sgdclassifier__max_iter': [100, 1500, 100000],
            'sgdclassifier__tol': [1e-3, 1e-5, 1e-1],
            'sgdclassifier__shuffle': [True],
            'sgdclassifier__epsilon': [0.1, 0.01, 0.001],
            'sgdclassifier__learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
            # eta0 must be > 0
            'sgdclassifier__eta0': [0.2, 0.01, 0.001],
            'sgdclassifier__power_t': [0.5, 0.1, 0.7],
            'sgdclassifier__early_stopping': [False],
            'sgdclassifier__validation_fraction': [0.1, 0.2],
            'sgdclassifier__n_iter_no_change': [5, 20],
            'sgdclassifier__average': [False, True],
            'sgdclassifier__random_state': [RANDOM_STATE]
        })

    if isinstance(clf, AdaBoostClassifier):
        param_grid.update({
            'adaboostclassifier__n_estimators': [10, 100, 1000, 10000],
            'adaboostclassifier__learning_rate': [1., 0.1, 0.5],
            'adaboostclassifier__algorithm': ['SAMME', 'SAMME.R'],
            'adaboostclassifier__random_state': [RANDOM_STATE]
        })

    if isinstance(clf, BaggingClassifier):
        param_grid.update({
            'baggingclassifier__n_estimators': [10, 20, 30, 500],
            'baggingclassifier__max_samples': [1.0, 0.5, 0.1],
            'baggingclassifier__max_features': [1.0, 0.5, 0.2],
            'baggingclassifier__random_state': [RANDOM_STATE]
        })

    if isinstance(clf, LinearDiscriminantAnalysis):
        param_grid.update({
            'lineardiscriminantanalysis__solver': ['svd', 'lsqr'],  # 'eigen' causes trouble
            'lineardiscriminantanalysis__shrinkage': [None],
            'lineardiscriminantanalysis__n_components': [None],
            'lineardiscriminantanalysis__store_covariance': [False, True],
            'lineardiscriminantanalysis__tol': [1e-6, 1e-2],
        })

    if isinstance(clf, NearestCentroid):
        param_grid.update({
            'nearestcentroid__metric': ['euclidean', 'manhattan'],
            'nearestcentroid__shrink_threshold': [None, 0.1, 0.2, 0.9]
        })

    if isinstance(clf, Perceptron):
        param_grid.update({
            'perceptron__penalty': ['l2', 'l1', 'elasticnet'],
            'perceptron__alpha': [0.0001, 0.01, 0.1],
            'perceptron__l1_ratio': [0.15, 0.05, 0.3],
            'perceptron__fit_intercept': [True],
            'perceptron__max_iter': [1000, 5000, 50000],
            'perceptron__tol': [1e-3, 1e-7, 1e-1],
            'perceptron__shuffle': [True],
            'perceptron__eta0': [1., 1.2, 0.8],
            'perceptron__random_state': [RANDOM_STATE],
            'perceptron__validation_fraction': [0.1, 0.2],
            'perceptron__n_iter_no_change': [5, 20]
        })

    if isinstance(clf, MLPClassifier):
        layers_num = random.randint(1, 5)
        layers = []
        for x in range(layers_num):
            layers.append(random.randint(1, 300))
        param_grid.update({
            'mlpclassifier__hidden_layer_sizes': tuple(layers),
            'mlpclassifier__activation': ['identity', 'logistic', 'tanh', 'relu'],
            'mlpclassifier__solver': ['lbfgs', 'sgd', 'adam'],
            'mlpclassifier__alpha': [0.0001, 0.1, 0.00001],
            'mlpclassifier__learning_rate': ['constant', 'invscaling', 'adaptive'],
            'mlpclassifier__learning_rate_init': [0.0001, 0.01, 0.00001],
            'mlpclassifier__power_t': [0.5, 0.1, 0.8],
            'mlpclassifier__max_iter': [200, 2000, 100000],
            'mlpclassifier__shuffle': [True],
            'mlpclassifier__random_state': [RANDOM_STATE],
            'mlpclassifier__tol': [1e-4, 1e-8, 1e-2],
            'mlpclassifier__momentum': [0.9, 0.95, 0.6],
            'mlpclassifier__early_stopping': [False],
            'mlpclassifier__validation_fraction': [0.1, 0.2],
            'mlpclassifier__beta_1': [0.9, 0.8],
            'mlpclassifier__beta_2': [0.999, 0.8],
            'mlpclassifier__n_iter_no_change': [100],
            'mlpclassifier__max_fun': [15000, 30000]
        })

    if isinstance(clf, LinearSVC):
        param_grid.update({
            'linearsvc__penalty': ['l2'],  # 'l1',
            'linearsvc__loss': ['hinge', 'squared_hinge'],
            'linearsvc__dual': [True],
            'linearsvc__tol': [1e-6, 1e-4, 1e-2],
            'linearsvc__C': [1., 0.5],
            'linearsvc__multi_class': ['ovr', 'crammer_singer'],
            'linearsvc__random_state': [RANDOM_STATE],
            'linearsvc__max_iter': [100, 500, 5000, 50000],
        })
    if isinstance(clf, NuSVC):
        param_grid.update({
            'nusvc__nu': [0.1, 0.5],
            'nusvc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'nusvc__degree': [2, 3],
            'nusvc__gamma': ['scale', 'auto'],
            'nusvc__coef0': [0., 0.1],
            'nusvc__shrinking': [True, False],
            'nusvc__probability': [False, True],
            'nusvc__tol': [1e-3, 1e-7, 1e-1],
            'nusvc__cache_size': [200, 1000],
            'nusvc__max_iter': [-1],
            'nusvc__decision_function_shape': ['ovo', 'ovr'],
            'nusvc__break_ties': [False],
            'nusvc__random_state': [RANDOM_STATE],
        })

    if isinstance(clf, SVC):
        param_grid.update({
            'svc__C': np.linspace(start=1e-14, stop=1e2, num=3),
            'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'svc__degree': [2, 3],
            'svc__random_state': [RANDOM_STATE],
            'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid']
        })

    if isinstance(clf, XGBClassifier):
        param_grid.update({
            'xgbclassifier__n_estimators': [2, 10, 100],
            'xgbclassifier__learning_rate': [1e-4, 1e-2, 1e-1, 0.5, 1.],
            'xgbclassifier__random_state': [RANDOM_STATE],
            'xgbclassifier__booster': ['gbtree', 'gblinear', 'dart'],
            'xgbclassifier__max_depth': [6, 20, 300],
            'xgbclassifier__n_jobs': [N_JOBS],
            'xgbclassifier__eta': [0.3, 0.1, 0.01],
            'xgbclassifier__tree_method': ['auto'],
            'xgbclassifier__use_label_encoder': [False]
        })
    return param_grid


def get_random_param_grid_and_tuple_list(transformers: []):
    name_object_tuples = []
    param_grid = dict()
    svd_solvers = ['auto', 'randomized', 'full', 'arpack']
    for x in transformers:
        # print('Transformer:', x)
        # Copy parameter must be set to True, otherwise training set will get overwritten
        if isinstance(x, StandardScaler):
            param_grid['standardscaler__copy'] = [True]
            param_grid['standardscaler__with_mean'] = [True, False]
            param_grid['standardscaler__with_std'] = [True, False]
            name_object_tuples.append(('standardscaler', x))
        if isinstance(x, MinMaxScaler):
            param_grid['minmaxscaler__copy'] = [True]
            spectrum = [random.randint(0, 10), random.randint(0, 10)]
            while spectrum[0] == spectrum[1]:
                spectrum = [random.randint(0, 10), random.randint(0, 10)]
            param_grid['minmaxscaler__feature_range'] = [(min(spectrum), max(spectrum))]
            name_object_tuples.append(('minmaxscaler', x))
        if isinstance(x, RobustScaler):
            param_grid['robustscaler__copy'] = [True]
            spectrum = [random.randint(5, 100), random.randint(5, 100)]
            while spectrum[0] == spectrum[1]:
                spectrum = [random.randint(5, 100), random.randint(5, 100)]
            param_grid['robustscaler__quantile_range'] = [(min(spectrum), max(spectrum))]
            param_grid['robustscaler__with_centering'] = [True, False]
            param_grid['robustscaler__with_scaling'] = [True, False]
            name_object_tuples.append(('robustscaler', x))
        if isinstance(x, PCA):
            param_grid['pca__copy'] = [True]
            param_grid['pca__iterated_power'] = [1, 20]
            param_grid['pca__n_components'] = np.arange(start=1, stop=MAX_N_COMPONENTS,
                                                        step=4)
            param_grid['pca__random_state'] = [RANDOM_STATE]
            param_grid['pca__svd_solver'] = svd_solvers
            param_grid['pca__tol'] = np.linspace(start=0.0, stop=2., num=4)
            param_grid['pca__whiten'] = [True, False]
            name_object_tuples.append(('pca', x))
        if isinstance(x, Binarizer):
            param_grid['binarizer__copy'] = [True]
            param_grid['binarizer__threshold'] = np.linspace(start=0.0, stop=1., num=4)
            name_object_tuples.append(('binarizer', x))
        if isinstance(x, PowerTransformer):
            param_grid['powertransformer__copy'] = [True]
            param_grid['powertransformer__method'] = ['yeo-johnson']
            # powertransformer__standardize = False causes trouble
            param_grid['powertransformer__standardize'] = [True]
            name_object_tuples.append(('powertransformer', x))
        if isinstance(x, QuantileTransformer):
            param_grid['quantiletransformer__copy'] = [True]
            param_grid['quantiletransformer__ignore_implicit_zeros'] = [False]
            # The number of quantiles cannot be greater than the number of samples used. Got 151 quantiles and 10 samples.
            param_grid['quantiletransformer__n_quantiles'] = np.arange(start=1, stop=200, step=50)
            param_grid['quantiletransformer__output_distribution'] = ['uniform', 'normal']
            param_grid['quantiletransformer__random_state'] = [RANDOM_STATE]
            param_grid['quantiletransformer__subsample'] = np.arange(start=10, stop=1000000, step=200000)
            name_object_tuples.append(('quantiletransformer', x))
    return param_grid, name_object_tuples


def get_random_param_grid_and_tuple_list_min(transformers: []):
    name_object_tuples = []
    param_grid = dict()
    svd_solvers = ['auto', 'randomized', 'full', 'arpack']
    for x in transformers:
        # Copy parameter must be set to True, otherwise training set will get overwritten
        if isinstance(x, StandardScaler):
            param_grid['standardscaler__copy'] = [True]
            param_grid['standardscaler__with_mean'] = [True, False]
            param_grid['standardscaler__with_std'] = [True, False]
            name_object_tuples.append(('standardscaler', x))
        if isinstance(x, MinMaxScaler):
            param_grid['minmaxscaler__copy'] = [True]
            param_grid['minmaxscaler__feature_range'] = [(min(spectrum), max(spectrum))]
            name_object_tuples.append(('minmaxscaler', x))
        if isinstance(x, RobustScaler):
            param_grid['robustscaler__copy'] = [True]
            spectrum = [random.randint(5, 100), random.randint(5, 100)]
            while spectrum[0] == spectrum[1]:
                spectrum = [random.randint(5, 100), random.randint(5, 100)]
            param_grid['robustscaler__quantile_range'] = [(min(spectrum), max(spectrum))]
            param_grid['robustscaler__with_centering'] = [True, False]
            param_grid['robustscaler__with_scaling'] = [True, False]
            name_object_tuples.append(('robustscaler', x))
        if isinstance(x, PCA):
            param_grid['pca__copy'] = [True]
            param_grid['pca__iterated_power'] = [1, 20]
            param_grid['pca__n_components'] = np.arange(start=1, stop=MAX_N_COMPONENTS,
                                                        step=4)
            param_grid['pca__random_state'] = [RANDOM_STATE]
            param_grid['pca__svd_solver'] = svd_solvers
            param_grid['pca__tol'] = np.linspace(start=0.0, stop=2., num=4)
            param_grid['pca__whiten'] = [True, False]
            name_object_tuples.append(('pca', x))
        if isinstance(x, Binarizer):
            param_grid['binarizer__copy'] = [True]
            param_grid['binarizer__threshold'] = np.linspace(start=0.0, stop=1., num=4)
            name_object_tuples.append(('binarizer', x))
        if isinstance(x, PowerTransformer):
            param_grid['powertransformer__copy'] = [True]
            param_grid['powertransformer__method'] = ['yeo-johnson']
            # powertransformer__standardize = False causes trouble
            param_grid['powertransformer__standardize'] = [True]
            name_object_tuples.append(('powertransformer', x))
        if isinstance(x, QuantileTransformer):
            param_grid['quantiletransformer__copy'] = [True]
            param_grid['quantiletransformer__ignore_implicit_zeros'] = [False]
            # The number of quantiles cannot be greater than the number of samples used. Got 151 quantiles and 10 samples.
            param_grid['quantiletransformer__n_quantiles'] = np.arange(start=1, stop=200, step=50)
            param_grid['quantiletransformer__output_distribution'] = ['uniform', 'normal']
            param_grid['quantiletransformer__random_state'] = [RANDOM_STATE]
            param_grid['quantiletransformer__subsample'] = np.arange(start=10, stop=1000000, step=200000)
            name_object_tuples.append(('quantiletransformer', x))
    return param_grid, name_object_tuples


def get_min_param_grid_and_tuple_list(transformers: []):
    name_object_tuples = []
    param_grid = dict()
    svd_solvers = ['randomized', 'full', 'arpack']
    for x in transformers:
        # Copy parameter must be set to True, otherwise training set will get overwritten
        if isinstance(x, StandardScaler):
            param_grid['standardscaler__copy'] = [True]
            param_grid['standardscaler__with_mean'] = [True, False]
            param_grid['standardscaler__with_std'] = [True, False]
            name_object_tuples.append(('standardscaler', x))
        if isinstance(x, MinMaxScaler):
            param_grid['minmaxscaler__copy'] = [True]
            param_grid['minmaxscaler__feature_range'] = [(0, 10), (6, 9)]
            name_object_tuples.append(('minmaxscaler', x))
        if isinstance(x, RobustScaler):
            param_grid['robustscaler__copy'] = [True]
            param_grid['robustscaler__quantile_range'] = [(5, 100), (20, 50)]
            param_grid['robustscaler__with_centering'] = [True, False]
            param_grid['robustscaler__with_scaling'] = [True, False]
            name_object_tuples.append(('robustscaler', x))
        if isinstance(x, PCA):
            param_grid['pca__copy'] = [True]
            param_grid['pca__iterated_power'] = [1, 20]
            # n_components must be strictly less than min(n_samples, n_features)=41 with svd_solver='arpack
            # therefore -> MAX_N_COMPONENTS - 1
            param_grid['pca__n_components'] = [1, MAX_N_COMPONENTS - 1]
            param_grid['pca__random_state'] = [RANDOM_STATE]
            param_grid['pca__svd_solver'] = svd_solvers
            param_grid['pca__tol'] = [0., 2.]
            param_grid['pca__whiten'] = [True, False]
            name_object_tuples.append(('pca', x))
        if isinstance(x, Binarizer):
            param_grid['binarizer__copy'] = [True]
            param_grid['binarizer__threshold'] = [0., 1.]
            name_object_tuples.append(('binarizer', x))
        if isinstance(x, PowerTransformer):
            param_grid['powertransformer__copy'] = [True]
            param_grid['powertransformer__method'] = ['yeo-johnson']
            # powertransformer__standardize = False causes trouble
            param_grid['powertransformer__standardize'] = [True]
            name_object_tuples.append(('powertransformer', x))
        if isinstance(x, QuantileTransformer):
            param_grid['quantiletransformer__copy'] = [True]
            param_grid['quantiletransformer__ignore_implicit_zeros'] = [False]
            # The number of quantiles cannot be greater than the number of samples used. Got 151 quantiles and 10 samples.
            param_grid['quantiletransformer__n_quantiles'] = [1, 100]
            param_grid['quantiletransformer__output_distribution'] = ['uniform', 'normal']
            param_grid['quantiletransformer__random_state'] = [RANDOM_STATE]
            param_grid['quantiletransformer__subsample'] = [200, 1000000]
            name_object_tuples.append(('quantiletransformer', x))
        if isinstance(x, FastICA):
            param_grid['FastICA__tol'] = [0., 1.]
            param_grid['FastICA__random_state'] = [RANDOM_STATE]
            name_object_tuples.append(('FastICA', x))

    return param_grid, name_object_tuples


def create_genome(pipeline):
    genome = dict()
    pipeline_params = pipeline.get_params()
    for key, value in pipeline_params['steps']:
        for key_2, value_2 in value.get_params().items():
            s = str(key) + '__' + key_2
            genome[s] = value_2

    # extracting classifier from pipeline (always [last][second] in 'steps'
    clf = pipeline_params['steps'][len(pipeline_params['steps']) - 1][1]
    genome['classifier_type'] = type(clf).__name__
    # for key, value in clf.get_params().items():
    #     # print(f'{key=}, {value=}')
    #     genome[key] = value
    log.debug(f'Generated genotype: {genome}')
    return genome


def create_pipeline(genome):
    clf = globals()[genome['classifier_type']]()
    transformers = []
    keys_to_skip = ['classifier_type']
    all_mutators = ('standardscaler', 'minmaxscaler', 'robustscaler', 'pca', 'binarizer', 'powertransformer',
                    'quantiletransformer', 'FastICA')
    params_standardscaler, params_minmaxscaler, params_robustscaler, params_pca, params_binarizer, \
    params_powertransformer, params_quantiletransformer, params_FastICA = dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict()
    # (dict(),) * 7
    mutators = set()
    for key in genome:
        if key in keys_to_skip:
            continue

        if not key.startswith(all_mutators):
            # if key is classifier parameter
            mutator_name, separator, parameter_name = key.partition('__')
            clf.__setattr__(parameter_name, genome[key])
        else:
            # .partition(separator) - [0] - head (before separator), [1] - sepataror, [2] - tail (first after)
            mutator_name, separator, parameter_name = key.partition('__')
            if mutator_name == 'standardscaler':
                if parameter_name == 'random_state':
                    params_standardscaler[parameter_name] = RANDOM_STATE
                    continue
                params_standardscaler[parameter_name] = genome[key]
            if mutator_name == 'minmaxscaler':
                if parameter_name == 'random_state':
                    params_minmaxscaler[parameter_name] = RANDOM_STATE
                    continue
                params_minmaxscaler[parameter_name] = genome[key]
            if mutator_name == 'robustscaler':
                if parameter_name == 'random_state':
                    params_robustscaler[parameter_name] = RANDOM_STATE
                    continue
                params_robustscaler[parameter_name] = genome[key]
            if mutator_name == 'pca':
                if parameter_name == 'random_state':
                    params_pca[parameter_name] = RANDOM_STATE
                    continue
                params_pca[parameter_name] = genome[key]
            if mutator_name == 'binarizer':
                if parameter_name == 'random_state':
                    params_binarizer[parameter_name] = RANDOM_STATE
                    continue
                params_binarizer[parameter_name] = genome[key]
            if mutator_name == 'powertransformer':
                if parameter_name == 'random_state':
                    params_powertransformer[parameter_name] = RANDOM_STATE
                    continue
                params_powertransformer[parameter_name] = genome[key]
            if mutator_name == 'quantiletransformer':
                if parameter_name == 'random_state':
                    params_quantiletransformer[parameter_name] = RANDOM_STATE
                    continue
                params_quantiletransformer[parameter_name] = genome[key]
            if mutator_name == 'FastICA':
                if parameter_name == 'random_state':
                    params_FastICA[parameter_name] = RANDOM_STATE
                    continue
                params_FastICA[parameter_name] = genome[key]

    if params_standardscaler:
        transformers.append(StandardScaler(**params_standardscaler))
    if params_minmaxscaler:
        transformers.append(MinMaxScaler(**params_minmaxscaler))
    if params_robustscaler:
        transformers.append(RobustScaler(**params_robustscaler))
    if params_pca:
        transformers.append(PCA(**params_pca))
    if params_binarizer:
        transformers.append(Binarizer(**params_binarizer))
    if params_powertransformer:
        transformers.append(PowerTransformer(**params_powertransformer))
    if params_quantiletransformer:
        transformers.append(QuantileTransformer(**params_quantiletransformer))
    if params_FastICA:
        transformers.append(FastICA(**params_FastICA))

    pipeline = make_pipeline(*transformers, clf)
    log.debug(f'Individual transformers: {transformers}')
    log.debug(f'Individual generated based on genome: {pipeline}')

    return pipeline


def update_param_grid_identical_as_tpot(clf, param_grid):
    if isinstance(clf, BernoulliNB):
        param_grid.update(dict(
            bernoullinb__alpha=[1e-3, 1e-2, 1e-1, 1., 10., 100.],
            bernoullinb__fit_prior=[True, False]))
        return param_grid

        # MultinomialNB only on defaults (but it can always mutate)

    if isinstance(clf, KNeighborsClassifier):
        param_grid.update({
            # Expected n_neighbors <= n_samples.
            # 'kd_tree' causes -> DataConversionWarning: A column-vector y was passed when a 1d array was expected
            'kneighborsclassifier__n_neighbors': list(range(1, 101)),
            'kneighborsclassifier__p': [1, 2],
            'kneighborsclassifier__weights': ['uniform', 'distance']
        })

        # WARNING! isinstance(extra_trees_c_instance, DecisionTreeClassifier) -> True
    if str(clf).__contains__('DecisionTreeClassifier'):
        param_grid.update({
            'decisiontreeclassifier__criterion': ["gini", "entropy"],
            'decisiontreeclassifier__max_depth': list(range(1, 11)),
            'decisiontreeclassifier__min_samples_split': list(range(2, 21)),
            'decisiontreeclassifier__min_samples_leaf': list(range(1, 21)),
            'decisiontreeclassifier__random_state': [RANDOM_STATE]
        })

    if str(clf).__contains__('ExtraTreeClassifier'):
        param_grid.update({
            # 'extratreeclassifier__n_estimators': [100],
            'extratreeclassifier__criterion': ["gini", "entropy"],
            'extratreeclassifier__max_features': np.arange(0.05, 1.01, 0.05),
            'extratreeclassifier__min_samples_split': list(range(2, 21)),
            'extratreeclassifier__min_samples_leaf': list(range(1, 21)),
            'extratreeclassifier__min_weight_fraction_leaf': [0.0, 0.3],
            # 'extratreeclassifier__bootstrap': [True, False],
            'extratreeclassifier__random_state': [RANDOM_STATE],
        })
    # ExtraTreeClassifier
    if isinstance(clf, RandomForestClassifier):
        param_grid.update({
            'randomforestclassifier__n_estimators': [100],
            'randomforestclassifier__criterion': ["gini", "entropy"],
            'randomforestclassifier__max_features': np.arange(0.05, 1.01, 0.05),
            'randomforestclassifier__min_samples_split': list(range(2, 21)),
            'randomforestclassifier__min_samples_leaf': list(range(1, 21)),
            'randomforestclassifier__bootstrap': [True, False],
            'randomforestclassifier__random_state': [RANDOM_STATE],
        })

    if isinstance(clf, GradientBoostingClassifier):
        # "criterion='mae' was deprecated in version 0.24 and
        param_grid.update({
            'gradientboostingclassifier__n_estimators': [100],
            'gradientboostingclassifier__learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
            'gradientboostingclassifier__max_depth': list(range(1, 11)),
            'gradientboostingclassifier__min_samples_split': list(range(2, 21)),
            'gradientboostingclassifier__min_samples_leaf': list(range(1, 21)),
            'gradientboostingclassifier__subsample': np.arange(0.05, 1.01, 0.05),
            'gradientboostingclassifier__max_features': np.arange(0.05, 1.01, 0.05),
            'gradientboostingclassifier__random_state': [RANDOM_STATE],
        })

    if isinstance(clf, LogisticRegression):
        # Penalty was changed
        param_grid.update({'logisticregression__penalty': ['l2', 'none'],
                           'logisticregression__C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
                           # only False works
                           'logisticregression__dual': [False],
                           'logisticregression__random_state': [RANDOM_STATE],
                           })

    if isinstance(clf, SGDClassifier):
        param_grid.update({
            'sgdclassifier__loss': ['log', 'hinge', 'modified_huber', 'squared_hinge', 'perceptron'],
            'sgdclassifier__penalty': ['elasticnet'],
            'sgdclassifier__alpha': [0.0, 0.01, 0.001],
            'sgdclassifier__learning_rate': ['invscaling', 'constant'],
            'sgdclassifier__fit_intercept': [True, False],
            'sgdclassifier__l1_ratio': [0.25, 0.0, 1.0, 0.75, 0.5],
            'sgdclassifier__eta0': [0.1, 1.0, 0.01],
            'sgdclassifier__power_t': [0.5, 0.0, 1.0, 0.1, 100.0, 10.0, 50.0],
            'sgdclassifier__random_state': [RANDOM_STATE]
        })

    if isinstance(clf, MLPClassifier):
        layers_num = random.randint(1, 3)
        layers = []
        for x in range(layers_num):
            layers.append(random.randint(1, 100))
        param_grid.update({
            'mlpclassifier__alpha': [1e-4, 1e-3, 1e-2, 1e-1],
            # 'mlpclassifier__learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
            'mlpclassifier__learning_rate': ['constant', 'invscaling', 'adaptive'],
            'mlpclassifier__learning_rate_init': [1e-3, 1e-2, 1e-1, 0.5, 1.],
            'mlpclassifier__random_state': [RANDOM_STATE]
        })

    if isinstance(clf, LinearSVC):
        param_grid.update({
            'linearsvc__penalty': ['l2'],  # 'l1' won't work with 'hinge'
            'linearsvc__loss': ['hinge', 'squared_hinge'],
            'linearsvc__dual': [True, False],
            'linearsvc__tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'linearsvc__C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
            'linearsvc__random_state': [RANDOM_STATE]
        })

    if isinstance(clf, XGBClassifier):
        param_grid.update({

            'xgbclassifier__n_estimators': [100],
            'xgbclassifier__max_depth': list(range(1, 11)),
            'xgbclassifier__learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
            'xgbclassifier__subsample': np.arange(0.05, 1.01, 0.05),
            'xgbclassifier__min_child_weight': list(range(1, 21)),
            'xgbclassifier__n_jobs': [N_JOBS],
            'xgbclassifier__verbosity': [0],
            'xgbclassifier__random_state': [RANDOM_STATE],

        })
    log.info(f'{param_grid=}')
    return param_grid


def set_threads(num_threads):
    # mkl.set_num_threads(num_threads)
    num_threads = str(num_threads)
    # names differ depending on system
    os.environ["OMP_NUM_THREADS"] = num_threads
    os.environ["OPENBLAS_NUM_THREADS"] = num_threads
    os.environ["MKL_NUM_THREADS"] = num_threads
    os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
    os.environ["NUMEXPR_NUM_THREADS"] = num_threads


def set_max_n_components(x_train, validation_method):
    # generating max value for PCA and others, based on validation method and data shape
    # n_components must be strictly less than min(n_samples, n_features) with svd_solver='arpack'
    global MAX_N_COMPONENTS
    if isinstance(validation_method, LeaveOneOut):
        MAX_N_COMPONENTS = min(int(len(x_train) / x_train.shape[1] - 1), x_train.shape[1])
        log.info(
            ' CV:{validation_method} | Train set shape:{shape} | max n_components value was set to:{stop}'.format(
                validation_method=validation_method, shape=x_train.shape, stop=MAX_N_COMPONENTS))
    else:
        MAX_N_COMPONENTS = min(int(len(x_train) / validation_method), x_train.shape[1])
        log.info(
            ' CV:{validation_method} | Train set shape:{shape} | max n_components value was set to:{stop}'.format(
                validation_method=validation_method, shape=x_train.shape, stop=MAX_N_COMPONENTS))


def backup_data(x_train, y_train):
    global ORIGINAL_X_TRAIN
    global ORIGINAL_Y_TRAIN
    ORIGINAL_X_TRAIN = x_train.copy()
    ORIGINAL_Y_TRAIN = y_train.copy()


def set_random_state(random_state):
    global RANDOM_STATE
    RANDOM_STATE = random_state


def set_njobs(njobs):
    global N_JOBS
    N_JOBS = njobs


def set_preselection(preselection):
    global PRESELECTION
    PRESELECTION = preselection


def set_cv(cv):
    global CV
    CV = cv


def set_selection(selection_type: str):
    global SELECTION
    SELECTION = selection_type


def unpack_history(population):
    all_individuals = []
    for i, pop in enumerate(population.history):
        for x in pop:
            all_individuals.append(x)
    heaven = Population(individuals=all_individuals, history=[], dataset_name=population.dataset_name,
                        dataset_rows=population.dataset_rows,
                        dataset_attributes=population.dataset_attributes,
                        dataset_classes=population.dataset_classes, random_state=population.random_state,
                        failed_to_test=population.failed_to_test)
    return heaven


'''
   !-- OTHER PARAMETERS RESTRCTIONS --!
   PCA:
    * n_components must be between 0 and min(n_samples, n_features)=41 with svd_solver='full'
    * robustscaler__copy = (0-99, 1-100)

    TREES:
        *  min_weight_fraction_leaf must in [0, 0.5]
        * if int - max_features must be in (0, n_features] - depends on data

        setting alpha = 1.0e-10
#   warnings.warn('alpha too small will result in numeric errors, '
# break_ties must be False when decision_function_shape is 'ovo'

        # sometimes -> Intel MKL ERROR: Parameter 10 was incorrect on entry to DGESDD.
        # I think PowerTransformer is not a good combination
        # Soo.. if MKL won't limit cores we better get rid of it
        
        
        # momentum must be >= 0 and <= 1
        
        
            # Parameters: { "gamma", "max_depth", "min_child_weight", "tree_method" } might not be used.
            #
            #   This may not be accurate due to some parameters are only used in language bindings but
            #   passed down to XGBoost core.  Or some parameters are not used but slip through this
            #   verification. Please open an issue if you find above cases.
            # 'xgbclassifier__gamma': [0, 0.1],
            # 'xgbclassifier__min_child_weight': [1, 2, 5],
            # Parameters: { "sampling_method" } might not be used.
            #
            #   This may not be accurate due to some parameters are only used in language bindings but
            #   passed down to XGBoost core.  Or some parameters are not used but slip through this
            #   verification. Please open an issue if you find above cases.
            # 'xgbclassifier__sampling_method': ['uniform', 'gradient_based'],
            
            
            # Logistic Regression supports only penalties in ['l1', 'l2', 'elasticnet', 'none'], got None.
            
            
            
            
            
            LEGACY
            
            def get_best_from_history(pops, validation_method):
    for i, pop in enumerate(pops):
        if len(pop) == 0:
            log.warning('No Individuals found in Population!')
            return 0
        pop2 = []
        best = None
        for x in pop:
            if x.validation_method is None:
                log.warning("Individual ignored - not tested yet")
                continue
            if x.validation_method != validation_method:
                log.warning("Individual ignored - different cross-validation values were used in testing.")
                continue
            pop2.append(x)
        if len(pop2) != 0:
            best = max(pop2, key=attrgetter('score'))
        return best



'''
