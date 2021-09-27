import platform
# from functools import cache

import pandas as pd
import psutil
import cpuinfo
import base64
import io
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


global status
global tpot_status
global status_thread
global stop_threads
global stop_tpot
global tpot
global cached_data
global last_selections
global current_cpu_usage
global cpu_history
global mem_history
global tpot_thread

global threads
global machine_info

global plots

global queue

global PIPELINES
global TEST_PIPELINES
global TEST_STATUS


def init_pipelines():
    global PIPELINES
    global TEST_PIPELINES
    global TEST_STATUS
    PIPELINES = []
    TEST_PIPELINES = set()
    TEST_STATUS = dict()
    TEST_STATUS['status'] = '<br/><date>' + datetime.now().strftime(
        "%d.%m.%Y|%H-%M-%S") + ':</date> Nothing analysed yet.'


def init_queue():
    global queue
    queue = []


def init_tpot_thread():
    global tpot_thread
    tpot_thread = None


def init_stop_tpot():
    global stop_tpot
    stop_tpot = False


def init_plots():
    global plots
    plots = dict()


def init_cpu_history():
    global cpu_history
    cpu_history = []


def init_mem_history():
    global mem_history
    mem_history = []


def init_selections():
    global last_selections
    last_selections = dict()
    last_selections['rows_option'] = 20
    last_selections['columns_option'] = 10
    last_selections['rows'] = 'No data displayed.'
    last_selections['columns'] = 'No data displayed.'
    last_selections['classes'] = 'No data displayed.'


from datetime import datetime


def initialize_status():
    global status
    #         [status, best_score, pipeline, time, bar, plot]
    status = dict()
    status['status'] = '<br/><date>' + datetime.now().strftime("%d.%m.%Y|%H-%M-%S") + ':</date> GMC not working'
    # status = ['Idle', None, None, None, None, None]


def initialize_tpot_status():
    global tpot_status
    #            [status, best_score, pipeline, time, bar, plot]
    tpot_status = dict()
    tpot_status['status'] = '<br/><date>' + datetime.now().strftime("%d.%m.%Y|%H-%M-%S") + ':</date> TPOT not working'
    # tpot_status = ['TPOT not working', None, None, None, None, None, []]


def init_stop_threads():
    global stop_threads
    stop_threads = False


def init_tpot():
    global tpot
    tpot = None


def init_threads():
    global threads
    global machine_info
    available_threads = psutil.cpu_count(logical=True)
    machine_info['free_threads'] = available_threads
    # threads = (list(),) * available_threads
    threads = [None] * available_threads


def get_size(bytes, suffix="B"):
    """
    Credits: Abdou Rockikz
    Source: https://www.thepythoncode.com/article/get-hardware-system-information-python

    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def init_machine_info():
    global machine_info
    machine_info = {'cpu_name': cpuinfo.get_cpu_info()['brand_raw'], 'cpu': platform.processor(),
                    'logical_cores': psutil.cpu_count(logical=True),
                    'physical_cores': psutil.cpu_count(logical=False),
                    'max_non-turbo_frequency': psutil.cpu_freq().max, 'memory': get_size(psutil.virtual_memory().total),
                    'available_memory': get_size(psutil.virtual_memory().available),
                    'memory_usage': psutil.virtual_memory().percent}


def update_usage():
    global current_cpu_usage
    current_cpu_usage = dict()
    for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=1)):
        current_cpu_usage[f'core {i}'] = percentage
    current_cpu_usage['total_usage'] = psutil.cpu_percent()
    machine_info['memory_usage'] = psutil.virtual_memory().percent
    machine_info['available_memory'] = get_size(psutil.virtual_memory().available)

    update_usage_plots(current_cpu_usage['total_usage'], machine_info['memory_usage'])


''' keeps the full history, limit plot to 100 cycles '''


# @cache
def update_usage_plots(cpu_usage, mem_usage):
    global cpu_history
    global mem_history
    cpu_history.append(cpu_usage)
    mem_history.append(mem_usage)
    if len(cpu_history) < 2:
        return
    if len(mem_history) < 2:
        return

    if len(cpu_history) > 99:
        cpu_history.pop(0)
        # optionally keep max 100 in history
        cpu_history = cpu_history
    if len(mem_history) > 99:
        mem_history.pop(0)
        # optionally keep max 100 in history
        mem_history = mem_history

    series = pd.Series(cpu_history)
    series.plot(use_index=False)
    fig = Figure()

    fig.set_dpi(200.)
    # beware that some Figure's parameters are in inches
    fig.set_figheight(2.4)
    fig.set_figwidth(4.)

    # fig.set_rasterized(True)
    axis = fig.add_subplot(1, 1, 1)
    # axis.set_title("SYSTEM USAGE")

    axis.set_xlabel("CYCLE")
    axis.set_ylabel("PERCENTAGE")
    axis.grid()

    axis.plot(series)
    series.update(mem_history)
    axis.plot(series)
    fig.legend(['CPU@' + str(cpu_usage) + '%', 'RAM@' + str(mem_usage) + '%'], loc='upper center',
               bbox_to_anchor=(0.5, 1.02),
               ncol=2, fancybox=True, shadow=True)

    # Convert plot to PNG image
    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)

    # Encode PNG image to base64 string
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')

    plots['cpumem'] = pngImageB64String


# def update_memory_usage():
#     machine_info['memory_usage'] = psutil.virtual_memory().percent
#     machine_info['available_memory'] = get_size(psutil.virtual_memory().available)


def init_cached_data():
    global cached_data
    cached_data = ''


def init_control():
    initialize_status()
    initialize_tpot_status()
    init_tpot()
    init_machine_info()
    init_cpu_history()
    init_threads()
    init_tpot_thread()
    init_stop_tpot()
    init_cached_data()
    init_selections()
    init_plots()
    init_mem_history()
    init_queue()
    init_pipelines()

    update_usage()
    # update_memory_usage()
