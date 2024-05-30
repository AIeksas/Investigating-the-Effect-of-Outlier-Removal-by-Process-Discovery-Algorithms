import pandas
import pm4py
import split_miner

from upsetplot import from_contents, plot
from matplotlib import pyplot
from utils import get_log, get_worst_performing_set
from config import hiring_logs

def get_only_protected(log, worst_performing_set):    
    merged_df = pandas.merge(log, worst_performing_set, left_on='case:concept:name', right_on='case_id', how='inner')
    protected_set = merged_df[merged_df['case:protected']]
    return protected_set

def get_set(log, discover_function):
    net, im, fm = discover_function(log)
    tbr_diagnostics = pm4py.conformance_diagnostics_token_based_replay(log, net, im, fm, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp', return_diagnostics_dataframe=True)
    set_result = get_only_protected(log, get_worst_performing_set(tbr_diagnostics))
    return set_result['case_id'].unique().tolist()

def get_split_miner_set(log, log_path, epsilon):
    net, im, fm = split_miner.discover_split_miner(log_path, epsilon)
    tbr_diagnostics = pm4py.conformance_diagnostics_token_based_replay(log, net, im, fm, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp', return_diagnostics_dataframe=True)
    set_result = get_only_protected(log, get_worst_performing_set(tbr_diagnostics))
    return set_result['case_id'].unique().tolist()

def get_log_skeleton_set(log, noise_threshold=0.1):
    log_skeleton = pm4py.discover_log_skeleton(log, noise_threshold=noise_threshold, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')
    conformance_lsk = pm4py.conformance_log_skeleton(log, log_skeleton, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp', return_diagnostics_dataframe=True)
    conformance_lsk['is_fit'] = conformance_lsk['dev_fitness'] == 1
    set_result = get_only_protected(log, get_worst_performing_set(conformance_lsk, 'dev_fitness'))
    return set_result['case_id'].unique().tolist()

def get_declare_set(log, min_confidence_ratio = 0, min_support_ratio = 0):
    declare_model = pm4py.discover_declare(log, min_confidence_ratio=min_confidence_ratio, min_support_ratio=min_support_ratio)
    conf_result = pm4py.conformance_declare(log, declare_model, return_diagnostics_dataframe=True)
    conf_result['is_fit'] = conf_result['dev_fitness'] == 1
    set_result = get_only_protected(log, get_worst_performing_set(conf_result, 'dev_fitness'))
    return set_result['case_id'].unique().tolist()


def run(log_path):
    log = get_log(log_path)
    results = {}

    discover_function = lambda x: pm4py.discover_petri_net_alpha(x) #Alpha Miner
    results["Alpha Miner"] = get_set(log, discover_function)

    discover_function = lambda x: pm4py.discovery.discover_petri_net_inductive(x, noise_threshold = 0.18) #Inductive Miner
    results["Inductive Miner"] = get_set(log, discover_function)
     
    discover_function = lambda x: pm4py.discovery.discover_petri_net_heuristics(x, dependency_threshold = 1) #Heuristics Miner
    results["Heuristics Miner"] = get_set(log, discover_function)

    discover_function = lambda x: pm4py.discovery.discover_petri_net_ilp(x, alpha=0.73) #ILP Miner
    results["ILP Miner"] = get_set(log, discover_function)

    results["Split Miner"] = get_split_miner_set(log, log_path, 0.1)

    results["Log Skeleton"] = get_log_skeleton_set(log, 0.1)

    results["Declare"] = get_declare_set(log, min_confidence_ratio = 0.8, min_support_ratio=0.8)

    
    for_plot = from_contents(results)
    # print(for_plot)
    plot(for_plot, sort_by='cardinality', show_counts=True, sort_categories_by='input')
    title = log_path.split('/')[-1].split('.')[0] + " protected"
    pyplot.title(title)
    pyplot.savefig(title)

if __name__ == "__main__":
    for log_path in hiring_logs:
        run(log_path)