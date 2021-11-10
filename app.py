from datetime import datetime
import logging

from flask import Flask, render_template, request, redirect
from engine import gmc_interface as gmc_interface
import global_control

# Setting log to debug may cause printing logs from used liberties (e.g. matplotlib)
log = logging.getLogger()
log.setLevel(logging.ERROR)
log.debug("this will get printed")
log.info("this will get printed - info")
log.warning("this will get printed - warning")
log.error("this will get printed - error")
log.critical("this will get printed - critical")


def main():
    global_control.init_control()
    log.info(f'Available threads:{len(global_control.threads)}')

    app = Flask(__name__)

    @app.context_processor
    def inject_status():
        return {'gmc_status': global_control.status, 'tpot_status': global_control.tpot_status,
                'dataset': global_control.cached_data, 'cpu': global_control.current_cpu_usage,
                'memory': global_control.machine_info['memory_usage'], 'plots': global_control.plots,
                'test_status': global_control.TEST_STATUS['status']}

    @app.route('/')
    def summary():
        pipelines = gmc_interface.load_best_pipelines()
        if pipelines:
            selected_pipeline = request.args.get('pipelines', pipelines[list(pipelines.keys())[0]])
        return render_template('summary.html', machine_info=global_control.machine_info,
                               cpu_usage=global_control.update_usage(), pipelines=pipelines)

    @app.route('/simple', methods=["GET", "POST"])
    def simple():
        if 'dataset' in global_control.last_selections:
            selected_dataset = global_control.last_selections['dataset']
        else:
            selected_dataset = gmc_interface.datasets[0]
        state = {'dataset': selected_dataset}
        return render_template('simple.html', datasets=gmc_interface.datasets, state=state,
                               cores=global_control.machine_info['logical_cores'],
                               cpu_name=global_control.machine_info['cpu_name'])

    @app.route('/gmc')
    def gmc():
        if 'dataset' in global_control.last_selections:
            selected_dataset = global_control.last_selections['dataset']
        else:
            selected_dataset = gmc_interface.datasets[0]

        if 'grid' in global_control.last_selections:
            selected_grid = global_control.last_selections['grid']
        else:
            selected_grid = gmc_interface.grids[0]

        if 'partial_explore' in global_control.last_selections:
            chosen_partial = global_control.last_selections['partial_explore']
        else:
            chosen_partial = 0

        selected_pop_size = global_control.last_selections_gmc['pop_size']
        selected_gen_size = global_control.last_selections_gmc['gen_size']
        selected_elitism = global_control.last_selections_gmc['elitism']
        selected_cv = global_control.last_selections_gmc['cv']
        selected_early_stop = global_control.last_selections_gmc['early_stop']
        selected_cross_chance = global_control.last_selections_gmc['cross_chance']
        selected_cross_method = global_control.last_selections_gmc['cross_method']
        selected_random_state = global_control.last_selections_gmc['random_state']
        selected_selection_method = global_control.last_selections_gmc['selection_method']
        selected_mutation = global_control.last_selections_gmc['mutation']
        selected_mutation_power = global_control.last_selections_gmc['mutation_power']
        selected_pipe_time = global_control.last_selections_gmc['pipe_time']
        selected_validation_part = global_control.last_selections_gmc['validation_part']

        preselection = global_control.last_selections_gmc['preselection']
        fresh_genes = global_control.last_selections_gmc['fresh_genes']

        state = {'dataset': selected_dataset, 'validation_size': selected_validation_part,
                 'pop_size': selected_pop_size, 'gen_size': selected_gen_size, 'elitism': selected_elitism,
                 'cv': selected_cv, 'cross_chance': selected_cross_chance, 'cross_method': selected_cross_method,
                 'selection_method': selected_selection_method, 'early_stop': selected_early_stop,
                 'pipe_time': selected_pipe_time, 'mutation': selected_mutation,
                 'mutation_power': selected_mutation_power, 'random_state': selected_random_state,
                 'grid': selected_grid, 'preselection': preselection, 'fresh_genes': fresh_genes,
                 'partial_explore': chosen_partial}
        return render_template('gmc.html', datasets=gmc_interface.datasets, grids=gmc_interface.grids,
                               free_threads=global_control.machine_info['free_threads'], state=state)

    @app.route('/tpot')
    def tpot():
        if 'dataset' in global_control.last_selections:
            selected_dataset = global_control.last_selections['dataset']
        else:
            selected_dataset = gmc_interface.datasets[0]

        selected_pop_size = global_control.last_selections_tpot['pop_size']
        selected_offspring_size = global_control.last_selections_tpot['offspring_size']
        selected_gen_size = global_control.last_selections_tpot['gen_size']
        selected_cv = global_control.last_selections_tpot['cv']
        selected_early_stop = global_control.last_selections_tpot['early_stop']
        selected_cross_chance = global_control.last_selections_tpot['cross_chance']
        selected_random_state = global_control.last_selections_tpot['random_state']
        selected_mutation = global_control.last_selections_tpot['mutation']
        selected_pipe_time = global_control.last_selections_tpot['pipe_time']
        selected_validation_part = global_control.last_selections_tpot['validation_part']

        state = {'dataset': selected_dataset, 'validation_size': selected_validation_part,
                 'pop_size': selected_pop_size, 'offspring_size': selected_offspring_size,
                 'gen_size': selected_gen_size,
                 'cv': selected_cv, 'cross_chance': selected_cross_chance, 'early_stop': selected_early_stop,
                 'pipe_time': selected_pipe_time, 'mutation': selected_mutation, 'random_state': selected_random_state}
        return render_template('tpot.html', datasets=gmc_interface.datasets,
                               free_threads=global_control.machine_info['free_threads'], state=state)

    @app.route('/data_explorer')
    def data_explorer():
        if 'dataset' in global_control.last_selections:
            selected_dataset = global_control.last_selections['dataset']
        else:
            selected_dataset = gmc_interface.datasets[0]
        if 'rows_option' in global_control.last_selections:
            rows_option = global_control.last_selections['rows_option']
        if 'columns_option' in global_control.last_selections:
            columns_option = global_control.last_selections['columns_option']
        if 'adjust' in global_control.last_selections:
            adjusted = global_control.last_selections['adjust']
        else:
            adjusted = True
        state = {'dataset': selected_dataset, 'rows_option': rows_option, 'columns_option': columns_option,
                 'adjust': adjusted}
        return render_template('data_explorer.html', datasets=gmc_interface.datasets,
                               state=state, dataset_table=global_control.cached_data,
                               rows=global_control.last_selections['rows'],
                               columns=global_control.last_selections['columns'],
                               classes=global_control.last_selections['classes'])

    @app.route('/test')
    def test():
        gmc_interface.unpickle_pipelines()
        try:
            selected_pipe = global_control.PIPELINES[0]
        except IndexError:
            selected_pipe = 'No Pipelines to display'

        if 'dataset' in global_control.last_selections:
            selected_dataset = global_control.last_selections['dataset']
        else:
            selected_dataset = gmc_interface.datasets[0]

        selected_validation_part = global_control.last_selections_gmc['validation_part']
        selected_cv = global_control.last_selections_gmc['cv']
        selected_random_state = global_control.last_selections_gmc['random_state']

        if global_control.TEST_PIPELINES:
            p = global_control.TEST_PIPELINES.pop()
            global_control.TEST_PIPELINES.add(p)
            test_pipeline = p
        else:
            test_pipeline = 'No pipelines selected for test'
        state = {'pipeline': selected_pipe, 'test_pipeline': test_pipeline, 'dataset': selected_dataset,
                 'cv': selected_cv, 'validation_size': selected_validation_part, 'random_state': selected_random_state}
        return render_template('test.html', datasets=gmc_interface.datasets, pipelines=global_control.PIPELINES,
                               test_pipelines=global_control.TEST_PIPELINES,
                               state=state, free_threads=global_control.machine_info['free_threads'],
                               test_status=global_control.TEST_STATUS['status'], plots=global_control.plots['test'],
                               plots_text=global_control.plots['test_texts'])

    ''' =========================================================================================================
                                    DISPLAY AND TASK FUNCTIONS
        ========================================================================================================= '''

    @app.route('/show_data', methods=["GET", "POST"])
    def show_data():
        if request.method == "GET" or request.method == "POST":
            selected_dataset = request.form["datasets"]
            rows_option = request.form["rows_n"]
            columns_option = request.form["columns_n"]
            try:
                adjusting = request.form["adjust"]
                # checkbox returns nothing if not checked
            except KeyError:
                adjusting = False
            global_control.last_selections['dataset'] = selected_dataset
            global_control.last_selections['rows_option'] = rows_option
            global_control.last_selections['columns_option'] = columns_option
            global_control.last_selections['adjust'] = adjusting

            global_control.cached_data = gmc_interface.get_csv_data(request.form.get('datasets'),
                                                                    request.form.get('rows_n'),
                                                                    request.form.get('columns_n'),
                                                                    request.form.get('adjust'))
            return redirect('/data_explorer')

    @app.route('/simple_evolve', methods=["GET", "POST"])
    def simple_evolve():
        if request.method == "GET" or request.method == "POST":
            selected_dataset = request.form['datasets']
            core_balance = request.form["cores_balance"]
            gmc_interface.start_evolution_simple(request.form.get('datasets'),
                                                 core_balance)
            global_control.last_selections['dataset'] = selected_dataset
            return redirect('/simple')

    @app.route('/custom_gmc_run', methods=["GET", "POST"])
    def custom_gmc_run():
        if request.method == "GET" or request.method == "POST":
            selected_dataset = request.form["datasets"]
            global_control.last_selections['dataset'] = selected_dataset
            selected_grid = request.form["grids"]
            global_control.last_selections['grid'] = selected_grid
            validation_size = int(request.form['validation_size'])
            n_jobs = int(request.form['n_jobs'])
            pop_size = int(request.form['pop_size'])
            generations = int(request.form['gen_size'])
            elitism = int(request.form['elitism'])
            cv = int(request.form['cv'])
            early_stop = int(request.form['early_stop'])
            cross_chance = int(request.form['cross_chance'])
            cross_method = int(request.form['cross_method'])
            random_state = int(request.form['random_state'])
            global_control.last_selections_gmc['random_state'] = random_state
            if random_state == 0:
                random_state = None
            selection_type = int(request.form['selection_method'])
            mutation = int(request.form['mutation'])
            mutation_power = int(request.form['mutation_power'])
            pipeline_time_limit = int(request.form['pipe_time']) * 60
            grid = request.form['grids']
            partial_explore = int(request.form['partial_explore'])

            global_control.last_selections_gmc['pop_size'] = pop_size
            global_control.last_selections_gmc['gen_size'] = generations
            global_control.last_selections_gmc['elitism'] = elitism
            global_control.last_selections_gmc['cv'] = cv
            global_control.last_selections_gmc['early_stop'] = early_stop
            global_control.last_selections_gmc['cross_chance'] = cross_chance
            global_control.last_selections_gmc['cross_method'] = cross_method
            global_control.last_selections_gmc['selection_method'] = selection_type
            global_control.last_selections_gmc['mutation'] = mutation
            global_control.last_selections_gmc['mutation_power'] = mutation_power
            global_control.last_selections_gmc['pipe_time'] = request.form['pipe_time']
            global_control.last_selections_gmc['validation_part'] = validation_size
            global_control.last_selections['partial_explore'] = partial_explore
            try:
                preselection = request.form["preselection"]
            except KeyError:
                preselection = False
            try:
                fresh_blood = request.form["fresh_blood"]
            except KeyError:
                fresh_blood = False

            global_control.last_selections_gmc['preselection'] = preselection
            global_control.last_selections_gmc['fresh_genes'] = fresh_blood

            # if n_jobs == 0:
            #     global_control.status['status'] += '<br/><date>' + datetime.now().strftime(
            #         "%d.%m.%Y|%H-%M-%S") + ':</date> Zero cores = zero work'
            #     return redirect('/gmc')
            gmc_interface.run_gmc_thread(selected_dataset, n_jobs=n_jobs, population=pop_size, generations=generations,
                                         validation_size=validation_size, cv=cv, early_stop=early_stop,
                                         crossover_rate=cross_chance, cross_method=cross_method,
                                         selection_type=selection_type, mutation=mutation,
                                         mutation_power=mutation_power, pipeline_time_limit=pipeline_time_limit,
                                         preselection=preselection, elitism=elitism, random_state=random_state,
                                         grid=grid, fresh_blood=fresh_blood, partial_explore=partial_explore)
            return redirect('/gmc')

    @app.route('/custom_tpot_run', methods=["GET", "POST"])
    def custom_tpot_run():
        if request.method == "GET" or request.method == "POST":
            selected_dataset = request.form["datasets"]
            global_control.last_selections['dataset'] = selected_dataset
            validation_size = int(request.form['validation_size'])
            n_jobs = int(request.form['n_jobs'])
            pop_size = int(request.form['pop_size'])
            offspring_size = int(request.form['offspring_size'])
            generations = int(request.form['gen_size'])
            cv = int(request.form['cv'])
            early_stop = int(request.form['early_stop'])
            cross_chance = int(request.form['cross_chance'])
            random_state = int(request.form['random_state'])
            global_control.last_selections_tpot['random_state'] = random_state
            if random_state == 0:
                random_state = None
            mutation = int(request.form['mutation'])
            pipeline_time_limit = int(request.form['pipe_time']) * 60

            global_control.last_selections_tpot['pop_size'] = pop_size
            global_control.last_selections_tpot['offspring_size'] = offspring_size
            global_control.last_selections_tpot['gen_size'] = generations
            global_control.last_selections_tpot['cv'] = cv
            global_control.last_selections_tpot['early_stop'] = early_stop
            global_control.last_selections_tpot['cross_chance'] = cross_chance
            global_control.last_selections_tpot['mutation'] = mutation
            global_control.last_selections_tpot['pipe_time'] = request.form['pipe_time']
            global_control.last_selections_tpot['validation_part'] = validation_size

            # if n_jobs == 0:
            #     global_control.tpot_status['status'] += '<br/><date>' + datetime.now().strftime(
            #         "%d.%m.%Y|%H-%M-%S") + ':</date> Zero cores = zero work'
            #     return redirect('/tpot')
            gmc_interface.run_tpot_thread(file_name=selected_dataset, validation_size=validation_size, n_jobs=n_jobs,
                                          population=pop_size, offspring=offspring_size, generations=generations, cv=cv,
                                          early_stop=early_stop, crossover_rate=cross_chance, random_state=random_state,
                                          mutation=mutation, pipeline_time_limit=pipeline_time_limit)
            return redirect('/tpot')

    @app.route('/add_pipeline', methods=["GET", "POST"])
    def add_pipeline():
        if request.method == "GET" or request.method == "POST":
            selected_pipeline = request.form.get('pipelines')
            log.debug(f'{selected_pipeline=}')
            global_control.TEST_PIPELINES.add(selected_pipeline)
            return redirect('/test')

    @app.route('/remove_pipeline', methods=["GET", "POST"])
    def remove_pipeline():
        if request.method == "GET" or request.method == "POST":
            selected_pipeline = request.form['test_pipelines']
            global_control.TEST_PIPELINES.remove(selected_pipeline)
            return redirect('/test')

    @app.route('/test_pipelines', methods=["GET", "POST"])
    def test_pipelines():
        if request.method == "GET" or request.method == "POST":
            selected_dataset = request.form.get('datasets')
            n_jobs = request.form.get('n_jobs')
            cv = request.form.get('cv')
            random_state = int(request.form.get('random_state'))
            global_control.last_selections_gmc['random_state'] = random_state
            if random_state == 0:
                random_state = None
            try:
                show_roc = request.form["show_roc"]
            except KeyError:
                show_roc = False
            try:
                t_test = request.form["t_test"]
            except KeyError:
                t_test = False
            test_size = request.form.get('validation_size')

            global_control.last_selections['dataset'] = selected_dataset
            global_control.last_selections_gmc['validation_part'] = test_size
            global_control.last_selections_gmc['cv'] = cv

            gmc_interface.start_test_thread(global_control.TEST_PIPELINES, selected_dataset, int(n_jobs), cv,
                                            random_state, show_roc, t_test, test_size)

            return redirect('/test')

    ''' =========================================================================================================
                                                AJAX UPDATES
        ========================================================================================================= '''

    @app.route('/gmc_status')
    def gmc_status():
        if 'status' in global_control.status:
            return global_control.status['status']
        return 'GMC status unknown'

    @app.route('/gmc_score')
    def gmc_score():
        if 'best_score' in global_control.status:
            return str(global_control.status['best_score'])
        return 'No results yet.'

    @app.route('/gmc_plot')
    def gmc_plot():
        if 'plot' in global_control.status:
            return '<img src="' + global_control.status['plot'] + '"/>'
        return 'Not enough data to plot'

    @app.route('/gmc_pipe')
    def gmc_pipe():
        if 'pipeline' in global_control.status:
            return str(global_control.status['pipeline'])
        return 'No results yet.'

    @app.route('/tpot_status')
    def tpot_status():
        return global_control.tpot_status['status']

    @app.route('/tpot_score')
    def tpot_score():
        if 'best_score' in global_control.tpot_status:
            return str(global_control.tpot_status['best_score'])
        return 'No results yet.'

    @app.route('/tpot_plot')
    def tpot_plot():
        if 'plot' in global_control.tpot_status:
            return '<img src="' + global_control.tpot_status['plot'] + '"/>'
        return 'Not enough data to plot'

    @app.route('/tpot_pipe')
    def tpot_pipe():
        if 'pipeline' in global_control.tpot_status:
            return global_control.tpot_status['pipeline']
        return 'No results yet.'

    @app.route('/cls')
    def cls():
        global_control.status['status'] = ''
        return redirect(request.referrer)

    @app.route('/cls_tpot')
    def cls_tpot():
        global_control.tpot_status['status'] = ''
        return redirect(request.referrer)

    @app.route('/stop_gmc')
    def stop_gmc():
        global_control.stop_threads = True
        global_control.status['status'] += '<br/><date>' + datetime.now().strftime(
            "%d.%m.%Y|%H-%M-%S") + ':</date> Gentle stop requested'
        return redirect(request.referrer)

    @app.route('/stop_tpot')
    def stop_tpot():
        global_control.tpot_status['status'] += '<br/><date>' + datetime.now().strftime(
            "%d.%m.%Y|%H-%M-%S") + ':</date> Gentle stop requested. TPOT got 1 last minute to finish'
        global_control.stop_tpot = True
        return redirect(request.referrer)

    @app.route('/debug_on')
    def debug_on():
        log.setLevel(logging.DEBUG)
        log.debug("this will get printed")
        log.info("this will get printed - info")
        log.warning("this will get printed - warning")
        log.error("this will get printed - error")
        log.critical("this will get printed - critical")
        global_control.status['status'] += '<br/><date>' + datetime.now().strftime(
            "%d.%m.%Y|%H-%M-%S") + ':</date> Debug mode:ON'
        return redirect(request.referrer)

    @app.route('/debug_off')
    def debug_off():
        log.setLevel(logging.ERROR)
        log.debug("this will get printed")
        log.info("this will get printed - info")
        log.warning("this will get printed - warning")
        log.error("this will get printed - error")
        log.critical("this will get printed - critical")
        global_control.status['status'] += '<br/><date>' + datetime.now().strftime(
            "%d.%m.%Y|%H-%M-%S") + ':</date> Debug mode:OFF'
        return redirect(request.referrer)

    @app.route('/test_status')
    def test_status():
        return global_control.TEST_STATUS['status']

    # host='0.0.0.0', port=5000, <- access in local network
    # app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=False)


if __name__ == '__main__':
    main()
