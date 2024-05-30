import pandas
import pm4py
import subprocess

from config import split_miner_path, project_path

def get_log(path):
    log = pm4py.read_xes(path)
    log['time:timestamp'] = pandas.to_datetime(log.index)
    return log

def discover_split_miner(log_path, epsilon=0.1):
    command = f"java -jar {split_miner_path} -i {project_path}/{log_path} -di -p {epsilon}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    output, error = process.communicate()
    # print("Output:")
    # print(output.decode())

    bpmn = pm4py.read_bpmn('output')
    return pm4py.convert_to_petri_net(bpmn)
     
    
if __name__ == "__main__":
    hiring_log_high = 'event_logs/hiring_log_high-xes/hiring_log_high.xes'
    log = get_log(hiring_log_high)
    
    net, im, fm = discover_split_miner(hiring_log_high)
    fitness_tbr = pm4py.fitness_token_based_replay(log, net, im, fm)
    print("Result", fitness_tbr)