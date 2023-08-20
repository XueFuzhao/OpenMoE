import json
import subprocess
import os
import logging
import sys
import warnings
import pandas as pd

logging.basicConfig()
logger = logging.getLogger('result_retrieval_bigbenchlite')
logger.setLevel(logging.DEBUG)


#******************************************************************************************************************************************************************************************************************
#                                   *** organizations of the evaluation metric result files ***

#* The metric result files categorized by tasks, models, shots, and subtasks should be put under the same directory without hierarchy. 
#* File name should be organized like "bigbench:{task_name}.{mul/gen}.{some hyperparams}.{shot_num}.{model_name}.{subtask_name}-metrics.jsonl".
#* e.g. bigbench:bbq_lite_json.mul.t5_default_vocab.3_shot.all_examples.bbq_lite_json_age_ambig-metrics.jsonl
#* Note: .{subtask_name} in the file name is optional. For some tasks, it doesn't have subtasks. For a certain task, the correspnding result files shouldn't contain both subtask files and non-subtask files.
#******************************************************************************************************************************************************************************************************************


# The tasks to be retrieved in BIG-bench. If a task is not found, the corresponding row with be Nones.
_tasks = ["auto_debugging", "bbq_lite_json", "code_line_description", 
         "conceptual_combinations", "conlang_translation", "emoji_movie", 
         "formal_fallacies_syllogisms_negation", "hindu_knowledge", 
         "known_unknowns", "language_identification", "linguistics_puzzles", 
         "logic_grid_puzzle", "logical_deduction", "misconceptions_russian", 
         "novel_concepts", "operators", "parsinlu_reading_comprehension", 
         "play_dialog_same_or_different", "repeat_copy_logic", "strange_stories", 
         "strategyqa", "symbol_interpretation", "vitaminc_fact_verification", "winowhy"]

# The models whose results to be retrieved for each task in BIG-bench. If a model is not found, the corresponding column with be Nones.
_models_repo = [
"BIG-G_2m_T=0",
"BIG-G_2m_T=1",
"BIG-G_16m_T=0",
"BIG-G_16m_T=1",
"BIG-G_53m_T=0",
"BIG-G_53m_T=1",
"BIG-G_125m_T=0",
"BIG-G_125m_T=1",
"BIG-G_244m_T=0",
"BIG-G_244m_T=1",
"BIG-G_422m_T=0",
"BIG-G_422m_T=1",
"BIG-G_1b_T=0",
"BIG-G_1b_T=1",
"BIG-G_2b_T=0",
"BIG-G_2b_T=1",
"BIG-G_4b_T=0",
"BIG-G_4b_T=1",
"BIG-G_8b_T=0",
"BIG-G_8b_T=1",
"BIG-G_27b_T=0",
"BIG-G_27b_T=1",
"BIG-G_128b_T=0",
"BIG-G_128b_T=1",
"BIG-G-sparse_2m",
"BIG-G-sparse_16m",
"BIG-G-sparse_53m",
"BIG-G-sparse_125m",
"BIG-G-sparse_244m",
"BIG-G-sparse_422m",
"BIG-G-sparse_1b",
"BIG-G-sparse_2b",
"BIG-G-sparse_4b",
"BIG-G-sparse_8b",
"GPT-3-3B", 
"GPT-3-6B", 
"GPT-3-13B", 
"GPT-3-200B",
"GPT-3-Small",
"GPT-3-Medium",
"GPT-3-Large",
"GPT-3-XL",
"PaLM_8b",
"PaLM_64b",
"PaLM_535b"
]

# Our models whose results to be retrieved for each task.
_models_ours = [
    "all_examples"
]

# n-shot results wanted. For example, if 0-shot and 3-shot is wanted, 
# each cell of the retrived results will be "score_0 score_3", which denotes the concatenated scores for 0-shot and 3-shot results
_number_of_shots = [0,1,2,3,4,5]



# The path to the results directory
_results_dir = '/data/personal/nus-njj/OpenMoE1/results/BIG-bench-Lite-013'

# The path to the BIG-bench tasks directory
_repo_dir = './BIG-bench/bigbench/benchmark_tasks'

# The path to the retrieved results
_file_to_save = f"{_results_dir}/retrieved_results.csv"



def find_repo_task_dir(repo_dir: str, tasks: list):
    task_dirs = []
    existing_tasks = []
    repo_tasks_dir = os.path.abspath(repo_dir)
    for task in tasks:
        for i, dir_name in enumerate(os.listdir(repo_tasks_dir)):
            if task == dir_name:
                task_dirs.append(os.path.join(repo_tasks_dir, dir_name))
                existing_tasks.append(dir_name)
                break
            elif i == len(os.listdir(repo_tasks_dir))-1:
                warnings.warn(f"Task '{task}' not found in the repo task dir {os.path.join(repo_tasks_dir, dir_name)}.")
                task_dirs.append(None)
                existing_tasks.append(None)
    assert len(task_dirs) == len(existing_tasks)
    return task_dirs, existing_tasks

def update_task_metainfo(task_metric_meta, model_json):
    with open(model_json, 'r') as json_file:
        results_dict = json.load(json_file)
    scores_dict = {f"{D['subtask_description']}---{D['number_of_shots']}_shot":D for D in results_dict['scores']}
    for key in scores_dict.keys():
        if key not in task_metric_meta.keys():
            task_metric_meta[key] = scores_dict[key]
    return task_metric_meta
 
def find_repo_task_model_json(task_dir: str, models: list):
    model_jsons = []
    task_metric_meta = {}
    repo_task_result_dir = os.path.abspath(f"{task_dir}/results")
    for model in models:
        if task_dir is not None:
            for i, file_name in enumerate(os.listdir(repo_task_result_dir)):
                if file_name.startswith('scores') and file_name.endswith('.json') and model in file_name:
                    model_jsons.append(os.path.join(repo_task_result_dir, file_name))
                    task_metric_meta = update_task_metainfo(task_metric_meta, os.path.join(repo_task_result_dir, file_name)) # update the task meta info regarding that in the repo
                    break
                elif i == len(os.listdir(repo_task_result_dir))-1:
                    warnings.warn(f"Result of model '{model}' not found in the repo task results dir {os.path.join(repo_task_result_dir, 'results')}.")
                    model_jsons.append(None)
        else:
            model_jsons.append(None)
    assert len(models) == len(model_jsons)
    
    return model_jsons, task_metric_meta

def min_max_normalize(score, min_score, max_score):
    normalized_score = (score - min_score) / (max_score - min_score)
    return normalized_score

def compute_scores_repo(models: list, model_jsons: list, number_of_shots: list):
    '''
    Returns a list of normalized scores whose length is equal to the model number, each score is a string concatenating the scores of n-shot.
    '''
    normalized_scores_allmodel = {}
    for model_json, model in zip(model_jsons, models):
        if model_json is not None:
            with open(model_json, 'r') as json_file:
                results_dict = json.load(json_file)
            scores_dict = {f"{D['subtask_description']}---{D['number_of_shots']}_shot":D for D in results_dict['scores']}
            normalized_scores = [{} for _ in number_of_shots] # the computed n-shot scores for this model in this task, n corresponds to the n-th digit in this list
            normalized_scores_existing = [0]*len(number_of_shots) # the computed n-shot scores for this model in this task, n corresponds to the n-th digit in this list
            for si, shot_num in enumerate(number_of_shots):
                for subtask, score_meta in scores_dict.items():
                    if score_meta['number_of_shots'] == shot_num:
                        raw_score = score_meta['score_dict'][score_meta['preferred_score']]
                        if len(subtask.split(':')) == 2:
                            high_score = score_meta['high_score']
                            low_score = score_meta['low_score']
                            normalized_scores[si][subtask] = raw_score
                            # normalized_scores[si][subtask] = raw_score
                        else:
                            high_score = score_meta['high_score']
                            low_score = score_meta['low_score']
                            normalized_scores_existing[si] += min_max_normalize(raw_score, low_score, high_score)
                            normalized_scores[si]['high_score'] = score_meta['high_score']
                            normalized_scores[si]['low_score'] = score_meta['low_score']
                            if len({k:v for k, v in normalized_scores[si].items() if k!='high_score' and k!='low_score'}.keys()) == 0:
                                normalized_scores[si][subtask] = raw_score
                                
            
            normalized_scores = [min_max_normalize(float(f"{sum(list({k:v for k, v in subtask_normscores_shot.items() if k!='high_score' and k!='low_score'}.values()))/len(list({k:v for k, v in subtask_normscores_shot.items() if k!='high_score' and k!='low_score'}.values())):.{4}f}"), subtask_normscores_shot['low_score'], subtask_normscores_shot['high_score']) 
                                 if len(list({k:v for k, v in subtask_normscores_shot.items() if k!='high_score' and k!='low_score'}.values()))>0 else 0.0 
                                 for subtask_normscores_shot in normalized_scores]
            normalized_scores_allmodel[model] = ' '.join([str(f"{100 * float(num):.{2}f}") for num in normalized_scores_existing])
        else:
            normalized_scores_allmodel[model] = 'None'
    return normalized_scores_allmodel
    
def retrieve_and_write_repo(file_to_save:str, repo_dir:list, 
                            models: list, tasks: list, 
                            number_of_shots: list = [3]):
    task_dirs, _ = find_repo_task_dir(repo_dir, tasks)
    normalized_scores = {task: [] for task in tasks}
    task_metric_metainfos = {task: [] for task in tasks}
    for task_dir, task in zip(task_dirs, tasks):
        model_jsons, task_metric_meta = find_repo_task_model_json(task_dir, models)
        normalized_scores[task] = compute_scores_repo(models, model_jsons, number_of_shots)
        task_metric_metainfos[task] = task_metric_meta

    def custom_sort_key(item):
        return tasks.index(item[0])
    normalized_scores = dict(sorted(normalized_scores.items(), key=custom_sort_key))
    Header = [['model']+tasks]
    score_mat = [models] + [inner_dict.values() for outer_key, inner_dict in normalized_scores.items()]
    df1 = pd.DataFrame(Header)
    df2 = pd.DataFrame(score_mat).transpose()
    df = pd.concat([df1, df2], ignore_index=True)
    df.to_csv(file_to_save, header=False, index=False)
    return df, task_metric_metainfos

def find_result_task_jsonsubsets(results_dir: str, tasks: list):
    tasks_origin = tasks
    tasks = [f'{task}.mul' for task in tasks] + [f'{task}.gen' for task in tasks]
    task_dir = {task:[] for task in tasks}
    task_dir_origin = {task:[] for task in tasks_origin}
    for task in tasks:
        for i, result_filename in enumerate(os.listdir(results_dir)):
            if task in result_filename:
                task_dir[task].append(os.path.join(results_dir, result_filename))
        if task_dir[task]==[]:
            task_dir.pop(task)

    for task in tasks_origin:
        for i, result_filename in enumerate(os.listdir(results_dir)):
            if task in result_filename:
                task_dir_origin[task].append('_')
        if task_dir_origin[task]==[]:
            warnings.warn(f"Task '{task}' not found in the result dir {results_dir}.")
    return task_dir
    

def find_result_taskmodel_json(task, task_jsons: list, models: list, number_of_shots: list):
    model_jsons = {model:{shot: [] for shot in number_of_shots} for model in models}

    if task_jsons != []:
        for model in models:
            model_found = False
            for i, file_name in enumerate(task_jsons):
                for si, number_of_shot in enumerate(number_of_shots):
                    if file_name.endswith('metrics.jsonl') and model in file_name and f"{number_of_shot}_shot" in file_name:
                            model_found = True
                            model_jsons[model][number_of_shot].append(file_name)
            if not model_found:
                warnings.warn(f"Result of model '{model}' not found in the {task} subset.")
            
    assert len(models) == len(model_jsons.keys())
    
    return model_jsons

def compute_scores_result(task: str, models: list, model_jsons: list, number_of_shots: list, task_metric_metainfos: dict):
    assert len(models) == len(model_jsons.keys())
    
    normalized_scores_allmodel = {}
    for model in models:
        model_found = False
        normalized_scores = [{} for _ in number_of_shots]
        for i, number_of_shot in enumerate(number_of_shots):
            if model_jsons[model][number_of_shot] != []:
                model_found = True
                for subtask_file in model_jsons[model][number_of_shot]:
                    subtask_name = subtask_file.split('.')[-2].replace('-metrics', '')
                    if subtask_name != model:
                        key_name = f"{task.replace('.mul', '').replace('.gen', '')}:{subtask_name}---{number_of_shot}_shot".replace('atikamp__', 'atikamp?_')
                    else:
                        key_name = f"{task.replace('.mul', '').replace('.gen', '')}---{number_of_shot}_shot".replace('atikamp__', 'atikamp?_')
                    key_name_task = f"{task.replace('.mul', '').replace('.gen', '')}---{number_of_shot}_shot".replace('atikamp__', 'atikamp?_')
                    preferred_score = task_metric_metainfos[task.replace('.mul', '').replace('.gen', '')][key_name]['preferred_score']
                    high_score = task_metric_metainfos[task.replace('.mul', '').replace('.gen', '')][key_name_task]['high_score']
                    low_score = task_metric_metainfos[task.replace('.mul', '').replace('.gen', '')][key_name_task]['low_score']
                    with open(subtask_file, 'r') as json_file:
                        results_dict = json.load(json_file)
                    if preferred_score in results_dict.keys():
                        score = results_dict[preferred_score]
                    else:
                        warnings.warn(f"Task '{task}' do not have the {preferred_score} metric.")
                        return None
                    normalized_scores[i][key_name] = score
                    normalized_scores[i]['low_score'] = low_score
                    normalized_scores[i]['high_score'] = high_score
                    # if 'logical_deduction' in subtask_file:
                    #     print(key_name, score, low_score, high_score)
        for shot_dir in normalized_scores:
            for subtask in {k:v for k, v in shot_dir.items() if k!='high_score' and k!='low_score'}.keys():
                if ':' not in subtask:
                    assert len({k:v for k, v in shot_dir.items() if k!='high_score' and k!='low_score'}.keys()) == 1, f"For task {task} that doesn't has subtask, there shouldn't be any other subtask result files."
        #  for the existing shots, average the scores corresponding to that shot and normalize. If a shot does not exist, set the corresponding score as 0.
        normalized_scores = [min_max_normalize(float(f"{sum(list({k:v for k, v in subtask_normscores_shot.items() if k!='high_score' and k!='low_score'}.values()))/len(list({k:v for k, v in subtask_normscores_shot.items() if k!='high_score' and k!='low_score'}.values())):.{4}f}"), subtask_normscores_shot['low_score'], subtask_normscores_shot['high_score']) 
                                 if len(list({k:v for k, v in subtask_normscores_shot.items() if k!='high_score' and k!='low_score'}.values()))>0 else 0.0 
                                 for subtask_normscores_shot in normalized_scores]
        if model_found:
            normalized_scores_allmodel[model] = ' '.join([str(f"{100 * float(num):.{2}f}") for num in normalized_scores])
        else:
            normalized_scores_allmodel[model] = 'None'
    return normalized_scores_allmodel

def retrieve_and_write_results(file_to_save, results_dir, models, tasks, number_of_shots, task_metric_metainfos):
    task_dir = find_result_task_jsonsubsets(results_dir, tasks)
    normalized_scores = {task: {} for task in task_dir.keys()}
    for task, task_jsons in task_dir.items():
        model_jsons = find_result_taskmodel_json(task, task_jsons, models, number_of_shots) # a dict consisting of hierarchically organized json files by the models, shots, and subtasks
        _r = compute_scores_result(task, models, model_jsons, number_of_shots, task_metric_metainfos)
        if _r is not None:
            normalized_scores[task] = _r
    normalized_scores = {k.replace('.mul', '').replace('.gen', ''):v for k, v in normalized_scores.items() if v!={}}
    def custom_sort_key(item):
        return tasks.index(item[0])
    normalized_scores = dict(sorted(normalized_scores.items(), key=custom_sort_key))

    assert len(normalized_scores.keys()) == len(tasks), f"len(normalized_scores.keys()), len(tasks): {len(normalized_scores.keys())}, {len(tasks)}"
    score_mat = [models] + [inner_dict.values() for outer_key, inner_dict in normalized_scores.items()]
    df = pd.DataFrame(score_mat).transpose()
    df.to_csv(file_to_save, header=False, index=False, mode='a')
    
    return df
    

def result_retrieval_bigbenchlite(file_to_save:str, results_dir: str, repo_dir:list, 
                                  models_repo: list, models_ours: list, tasks: list, 
                                  number_of_shots: list = [3]):
    # retrieve and write from the repository results
    subprocess.run("git clone git@github.com:google/BIG-bench.git".split())
    _, task_metric_metainfos = retrieve_and_write_repo(file_to_save, repo_dir, models_repo, tasks, number_of_shots)
    retrieve_and_write_results(file_to_save, results_dir, models_ours, tasks, number_of_shots, task_metric_metainfos)
    
            




if __name__ == "__main__":
    result_retrieval_bigbenchlite(_file_to_save, 
                                  _results_dir,
                                  _repo_dir,
                                  _models_repo, 
                                  _models_ours, 
                                  _tasks,
                                  _number_of_shots)