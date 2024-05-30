import pandas
import pm4py
import numpy as np

def get_log(path):
    log = pm4py.read_xes(path)
    log['time:timestamp'] = pandas.to_datetime(log.index)
    return log

def get_worst_performing_set(diagnostics_result, key='trace_fitness'):
    percent = 0.2
    num_traces = int(len(diagnostics_result) * percent)
    indices_lowest = diagnostics_result[key].nsmallest(num_traces).index

    result = diagnostics_result.loc[indices_lowest]
    df_output = result[['case_id', 'is_fit']]
    df_set = df_output[~df_output['is_fit']]
    return df_set