import pandas
import pm4py
import numpy as np
import split_miner

from matplotlib import pyplot
from utils import get_log, get_worst_performing_set
from config import hiring_logs

def get_worst_performing_set(diagnostics_result, key='trace_fitness'):
    percent = 0.2
    num_traces = int(len(diagnostics_result) * percent)
    indices_lowest = diagnostics_result[key].nsmallest(num_traces).index

    result = diagnostics_result.loc[indices_lowest]
    df_output = result[['case_id', 'is_fit']]
    df_set = df_output[~df_output['is_fit']]
    return df_set

def get_split_counts(log, discover_function, key='case:protected'):
    net, im, fm = discover_function(log)
    tbr_diagnostics = pm4py.conformance_diagnostics_token_based_replay(log, net, im, fm, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp', return_diagnostics_dataframe=True)
    worst_performing_set = get_worst_performing_set(tbr_diagnostics)
    return calculate_outlier_percentage(log, worst_performing_set, key)
    
def calculate_outlier_percentage(log, worst_performing_set, key='case:protected'):
    merged_df = pandas.merge(log, worst_performing_set, left_on='case:concept:name', right_on='case_id', how='inner')
    outlier_protected = merged_df[merged_df[key]]['case_id'].nunique()
    outlier_size = merged_df['case_id'].nunique()
        
    return get_percentage(outlier_protected, outlier_size)

def get_baseline(log):
    dataset_protected = log[log['case:protected']]['case:concept:name'].nunique()
    dataset_size = log['case:concept:name'].nunique()

    return get_percentage(dataset_protected, dataset_size)

def get_percentage(a, b):
    return round(100 * (a / b), 2)

def get_split_miner_split_counts(log, log_path, epsilon):
    net, im, fm = split_miner.discover_split_miner(log_path, epsilon)
    tbr_diagnostics = pm4py.conformance_diagnostics_token_based_replay(log, net, im, fm, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp', return_diagnostics_dataframe=True)
    worst_performing_set = get_worst_performing_set(tbr_diagnostics)
    return calculate_outlier_percentage(log, worst_performing_set)

def get_log_skeleton_split_counts(log, noise_threshold=0.1):
    log_skeleton = pm4py.discover_log_skeleton(log, noise_threshold=noise_threshold, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')
    conformance_lsk = pm4py.conformance_log_skeleton(log, log_skeleton, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp', return_diagnostics_dataframe=True)
    conformance_lsk['is_fit'] = conformance_lsk['dev_fitness'] == 1
    worst_performing_set = get_worst_performing_set(conformance_lsk, 'dev_fitness')
    return calculate_outlier_percentage(log, worst_performing_set)

def get_declare_split_counts(log, min_confidence_ratio = 0, min_support_ratio = 0):
    declare_model = pm4py.discover_declare(log, min_confidence_ratio=min_confidence_ratio, min_support_ratio=min_support_ratio)
    conf_result = pm4py.conformance_declare(log, declare_model, return_diagnostics_dataframe=True)
    conf_result['is_fit'] = conf_result['dev_fitness'] == 1
    worst_performing_set = get_worst_performing_set(conf_result, 'dev_fitness')
    return calculate_outlier_percentage(log, worst_performing_set)

def run(log_path):
    log = get_log(log_path)
    
    run_results = {}
    run_results["Dataset"] = get_baseline(log)

    discover_function = lambda x: pm4py.discover_petri_net_alpha(x) #Alpha Miner
    run_results["Alpha Miner"] = get_split_counts(log, discover_function)
    
    discover_function = lambda x: pm4py.discovery.discover_petri_net_inductive(x, noise_threshold = 0.18) #Inductive Miner
    run_results["Inductive Miner"] = get_split_counts(log, discover_function)
    
    discover_function = lambda x: pm4py.discovery.discover_petri_net_heuristics(x, dependency_threshold = 1) #Heuristics Miner
    run_results["Heuristics Miner"] = get_split_counts(log, discover_function)
    
    discover_function = lambda x: pm4py.discovery.discover_petri_net_ilp(x, alpha=0.73) #ILP Miner
    run_results["ILP Miner"] = get_split_counts(log, discover_function)

    run_results["Split Miner"] = get_split_miner_split_counts(log, log_path, 0.1)

    run_results["Log Skeleton"] = get_log_skeleton_split_counts(log, 0.1)

    run_results["Declare"] = get_declare_split_counts(log, min_confidence_ratio = 0.8, min_support_ratio=0.8)

    return run_results

def plot(data):
    print(data)

    x_values = np.arange(len(data))

    fig, ax = pyplot.subplots()
    width = 0.1
    multiplier = -4

    labels = ["Dataset", "Alpha Miner", "Inductive Miner", "Heuristics Miner", "ILP Miner", "Split Miner", "Log Skeleton", "Declare"]

    for label in labels:
        values = [item[label] for item in data]
        ax.bar(x_values + multiplier * width, values, width=width, label=label)
        multiplier = multiplier + 1

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Percentage, %')
    ax.set_title('Percentage of Protected Traces')
    ax.legend()

    ax.set_xticks(x_values)
    ax.set_xticklabels(['High', 'Medium', 'Low'])

    ax.hlines(y=data[0]["Dataset"], xmin=multiplier * width - 0.5*width, xmax=-1*multiplier * width - 0.5*width, linestyle='--', color='blue')
    ax.hlines(y=data[1]["Dataset"], xmin=1 + multiplier * width - 0.5*width, xmax=1+ -1*multiplier * width - 0.5*width, linestyle='--', color='blue')
    ax.hlines(y=data[2]["Dataset"], xmin=2 + multiplier * width - 0.5*width, xmax=2 +-1*multiplier * width - 0.5*width, linestyle='--', color='blue')

    # pyplot.show()
    pyplot.savefig('protected_traces')

if __name__ == "__main__":
    results = []

    for log_path in hiring_logs:
        results.append(run(log_path))
    
    plot(results)