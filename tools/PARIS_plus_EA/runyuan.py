import argparse
import os
import numpy as np
import json
import copy
import pickle
import time
import subprocess
import argparse

# 处理关系对齐
import pandas as pd

def load_relation_pairs(file_path):
    return pd.read_csv(file_path, sep='\t', header=None, names=['rel1', 'rel2', 'score'])

def filter_relations(df, threshold=0.6):
    filtered = df[df['score'] >= threshold].copy()
    filtered['is_reverse'] = filtered['rel2'].str.endswith('-')
    filtered['base_rel2'] = filtered['rel2'].str.rstrip('-')
    best_pairs = filtered.sort_values('score', ascending=False).drop_duplicates(['rel1', 'base_rel2'])
    return best_pairs[['rel1', 'rel2', 'score']]

def build_alignment_matrix(pairs_df):
    alignment = {}
    for _, row in pairs_df.iterrows():
        rel1, rel2, score = row['rel1'], row['rel2'], row['score']
        if rel1 not in alignment:
            alignment[rel1] = []
        alignment[rel1].append((rel2, score))
    return alignment

def find_dbo_non_dbo_pairs(alignment_matrix):
    dbo_non_dbo_pairs = []
    for rel1, matches in alignment_matrix.items():
        for rel2, score in matches:
            # 检查rel1和rel2是否满足一个含dbo前缀、另一个不含的条件
            rel1_has_dbo = rel1.startswith('dbo/')
            rel2_has_dbo = rel2.startswith('dbo/')
            if rel1_has_dbo != rel2_has_dbo:  # 异或逻辑
                dbo_non_dbo_pairs.append((rel1, rel2, score))
    return dbo_non_dbo_pairs

# PARIS
def read_triples(file_path):
    """
    read relation / attribute triples from file
    :param file_path: relation / attribute triples file path
    :return: relation / attribute triples
    """
    #triples = set()
    triples = list()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip('\n').split('\t')
            #triples.add((line[0], line[1], line[2]))
            triples.append((line[0], line[1], line[2]))
    file.close()
    return triples

def turn_and_write(rel_triples, attr_triples, seeds_triples, out_path):
    file = open(out_path, 'w', encoding='utf-8')
    for (s, p, o) in rel_triples:
        file.write('<' + s + '> <' + p + '> <' + o + '> .\n')
    for (s, p, o) in attr_triples:
        # FIX literal without quotes
        if not o.startswith('"'):
            mod_o = '"' + o + '"'
        else:
            mod_o = o
        file.write('<' + s + '> <' + p + '> ' + mod_o + ' .\n')
    for (s, p, l) in seeds_triples:
        file.write('<' + s + '> <' + p + '> ' + l + ' .\n')
    file.close()


def seed_triples(seed_aligns_name):
    """
    Returns two lists of triples, with the seed for PARIS well formatted into
    ({resource}, {#label}, {label})
    Parameters
    ----------
    folder
    dataset_division
    fold_num

    Returns
    -------
    seed_triples_1
    seed_triples_2
    """
    seed_triples_1 = []
    seed_triples_2 = []

    with open(seed_aligns_name, encoding="utf-8") as f:
    #with open("./data/CrossLPData_name/DBP-FB/cross/known_shared_entities.txt", encoding="utf-8") as f:
        for line in f:
            e1, e2 = line.strip("\n").split("\t")
            label = e1.split("/")[-1]
            #label = e2
            # Use label definition for N-Triples from W3C RDF recommendation. See https://www.w3.org/TR/n-triples/
            # label_str = '{resource} EntityMatchers:label "{label}"'

            # # Add the label
            # seed_triples_1.append((e1, "EntityMatchers:label", '"{}"'.format(label)))
            # seed_triples_2.append((e2, "EntityMatchers:label", '"{}"'.format(label)))

            label_str = '{resource} align "{label}"'

            # Add the label
            seed_triples_1.append((e1, "align", '"{}"'.format(label)))
            seed_triples_2.append((e2, "align", '"{}"'.format(label)))

    return seed_triples_1, seed_triples_2


    # 加入 KGC 预测出来的 .\pyclause_output\predicted_align.txt

def create_nt(folder, kg1: str, kg2: str, seed_aligns_path=None):

    if not os.path.exists(f"{folder}/PARIS_plus_temp_files"):
        os.makedirs(f"{folder}/PARIS_plus_temp_files")

    entid2name = dict()
    name2entid = dict()  # 添加反向映射，用于将实体名称转换回ID
    relid2name = dict()
    with open(f"{folder}/ent_ids_1", "r", encoding="utf-8") as f1, open(f"{folder}/ent_ids_2", "r", encoding="utf-8") as f2:
        for line in f1.readlines() + f2.readlines():
            idx, name = line.strip().split('\t')
            entid2name[idx] = name
            name2entid[name] = idx  # 构建名称到ID的映射
    with open(f"{folder}/rel_ids_1", "r", encoding="utf-8") as f1, open(f"{folder}/rel_ids_2", "r", encoding="utf-8") as f2:
        for line in f1.readlines() + f2.readlines():
            idx, name = line.strip().split('\t')
            relid2name[idx] = name

    with open(f"{folder}/rel_triples_1", "r", encoding="utf-8") as f1, open(f"{folder}/PARIS_plus_temp_files/rel_triples_1_name", "w", encoding="utf-8") as f2:
        for line in f1.readlines():
            e1, r, e2 = line.strip().split('\t')
            f2.write(f"{entid2name[e1]}\t{relid2name[r]}\t{entid2name[e2]}\n")
    with open(f"{folder}/rel_triples_2", "r", encoding="utf-8") as f1, open(f"{folder}/PARIS_plus_temp_files/rel_triples_2_name", "w", encoding="utf-8") as f2:
        for line in f1.readlines():
            e1, r, e2 = line.strip().split('\t')
            f2.write(f"{entid2name[e1]}\t{relid2name[r]}\t{entid2name[e2]}\n")

    try:
        with open(f"{folder}/attr_triples_1", "r", encoding="utf-8") as f1, open(f"{folder}/PARIS_plus_temp_files/attr_triples_1_name", "w", encoding="utf-8") as f2:
            for line in f1.readlines():
                e1, r, e2 = line.strip('\n').split('\t')
                f2.write(f"{entid2name[e1]}\t{r}\t{e2}\n")
        with open(f"{folder}/attr_triples_2", "r", encoding="utf-8") as f1, open(f"{folder}/PARIS_plus_temp_files/attr_triples_2_name", "w", encoding="utf-8") as f2:
            for line in f1.readlines():
                e1, r, e2 = line.strip('\n').split('\t')
                f2.write(f"{entid2name[e1]}\t{r}\t{e2}\n")
    except Exception as e:
        print("Attribute triples not found, skipping attribute triples processing. Error: ", e)

    # 如果没有指定种子对齐路径，使用默认路径并生成seed_aligns_name文件
    if not seed_aligns_path:
        seed_aligns_path = f"{folder}/seed_aligns"
    with open(seed_aligns_path, "r", encoding="utf-8") as f1, open(f"{folder}/PARIS_plus_temp_files/seed_aligns_name", "w", encoding="utf-8") as f2:
        for line in f1.readlines():
            e1, e2 = line.strip().split('\t')
            f2.write(f"{entid2name[e1]}\t{entid2name[e2]}\n")
    
    with open(f"{folder}/test_aligns", "r", encoding="utf-8") as f1, open(f"{folder}/PARIS_plus_temp_files/test_aligns_name", "w", encoding="utf-8") as f2:
        for line in f1.readlines():
            e1, e2 = line.strip().split('\t')
            f2.write(f"{entid2name[e1]}\t{entid2name[e2]}\n")


    rel_triples_1 = read_triples(f"{folder}/PARIS_plus_temp_files/rel_triples_1_name")
    rel_triples_2 = read_triples(f"{folder}/PARIS_plus_temp_files/rel_triples_2_name")
    try:
        attr_triples_1 = read_triples(f"{folder}/PARIS_plus_temp_files/attr_triples_1_name")
        attr_triples_2 = read_triples(f"{folder}/PARIS_plus_temp_files/attr_triples_2_name")
    except Exception as e:
        attr_triples_1 = []
        attr_triples_2 = []
        print("Attribute triples not found, skipping attribute triples processing. Error: ", e)
        
    seed_triples_1, seed_triples_2 = seed_triples(f"{folder}/PARIS_plus_temp_files/seed_aligns_name")

    turn_and_write(rel_triples_1, attr_triples_1, seed_triples_1, kg1)
    turn_and_write(rel_triples_2, attr_triples_2, seed_triples_2, kg2)
    
    return name2entid  # 返回名称到ID的映射


def run_paris(data_dir, out_folder, ontology1, ontology2):
    current_time = time.localtime()
    ontology1 = os.path.abspath(ontology1)
    ontology2 = os.path.abspath(ontology2)

    name = data_dir.split('/')[-2]

    task_name = '%s_%02d%02d_%02d%02d%02d' % (name, current_time.tm_mon, current_time.tm_mday,
                                              current_time.tm_hour, current_time.tm_min, current_time.tm_sec)

    task_name =  f"{out_folder}/{task_name}"

    os.mkdir(task_name)
    os.mkdir('%s/output' % task_name)
    os.mkdir('%s/log' % task_name)

    with open(task_name + '/paris.ini', 'w', encoding="utf-8") as ini_file:
        ini_file.write('resultTSV = %s/output\n' % task_name)
        ini_file.write('factstore1 = %s\n' % ontology1)
        ini_file.write('factstore2 = %s\n' % ontology2)
        ini_file.write('home = %s/log\n' % task_name)

    #_ = subprocess.call(['java', '-Xmx26000m', '-jar', './paris.jar', task_name + '/paris.ini'])
    import psutil
    total_memory_gb = psutil.virtual_memory().total // (1024**3)
    print(total_memory_gb)
    max_memory_mb = int(total_memory_gb * 0.8 * 1024)  # 80%的系统内存
    print(max_memory_mb)
    _ = subprocess.call(['java', '-Dfile.encoding=UTF-8', f'-Xmx{max_memory_mb}m', '-jar', './tools/PARIS_plus_EA/paris.jar', task_name + '/paris.ini'])
    return task_name


def compute_prec_rec_f1(aligns, truth_links):
    """
    Note that aligns should have been pruned from the alignments already present in the
    seed.
    Truth link, hence, must contain only test and valid links.
    Parameters
    ----------
    aligns
    truth_links

    Returns
    -------

    """
    aligns = set(aligns)
    truth_links = set(truth_links)
    num_correct = len(aligns.intersection(truth_links))
    if num_correct == 0 or len(aligns) == 0:
        print("Got 0, 0, 0 in evaluation!!!")
        return 0, 0, 0
    precision = num_correct / len(aligns)
    recall = num_correct / len(truth_links)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluate_paris(paris_out_folder, data_dir, threshold, name2entid, results_file):
    run_in = 0
    res_list = []
    while True:
        full_path = paris_out_folder + "/output/{run_in}_eqv.tsv".format(run_in=run_in)
        if os.path.exists(full_path):
            # PARIS create an empty file at the last_iter+1. If we encountered it, we can break
            if os.stat(full_path).st_size == 0:
                break
        run_in += 1
    full_path = paris_out_folder + "/output/{run_in}_eqv.tsv".format(run_in=run_in - 1)
    # Get PARIS result from the .tsv and elaborate it a bit to be compared with the same_list
    with open(full_path, encoding="utf-8") as f:
        for line in f:
            (e1, e2, confidence) = line.strip().split("\t")
            #print(e1, e2, confidence)
            if "dbp:" in e1:
                e1 = e1.replace('dbp:', 'http://dbpedia.org/')
            if "dbp:" in e2:
                e2 = e2.replace('dbp:', 'http://dbpedia.org/')
            if "y2:" in e2:
                e2 = e2.replace('y2:', "")
            #print(e1, e2, confidence)
            res_list.append((e1, e2, confidence))

    set_train = set()
    #with open(data_dir + "/aligned_entities.txt", encoding="utf-8") as f:
    with open(f"{data_dir}/PARIS_plus_temp_files/seed_aligns_name", encoding="utf-8") as f:
        for line in f:
            (e1, e2) = line.strip().split("\t")
            set_train.add((e1, e2))

    # 保留置信度高的，写入results.txt
    res_no_train = []
    with open(f"{data_dir}/PARIS_plus_output/predicted_align.txt", "w", encoding="utf-8") as f, open(results_file, "w", encoding="utf-8") as results_f:
        for align_with_confidence in res_list:
            align = (align_with_confidence[0], align_with_confidence[1])
            if align not in set_train and float(align_with_confidence[2]) > threshold:
                # Evaluate only the alignments that were not present in the training data (seed).
                res_no_train.append(align)
                f.write(f"{align_with_confidence[0]}\t{align_with_confidence[1]}\t{align_with_confidence[2]}\n")
                results_f.write(f"{name2entid[align_with_confidence[0]]}\t{name2entid[align_with_confidence[1]]}\t{align_with_confidence[2]}\n")

    # 测试，无验证
    test_links = []
    valid_links = []

    with open(f"{data_dir}/PARIS_plus_temp_files/test_aligns_name", encoding="utf-8") as f:
        for line in f:
            (e1, e2) = line.strip().split("\t")
            if (e1, e2) not in set_train:
                test_links.append((e1, e2))
    
    return compute_prec_rec_f1(res_no_train, test_links + valid_links)


def run_paris_experiment(data_dir, out_folder, threshold, seed_aligns_path=None, results_file="./results.txt"):
    precisions = []
    recalls = []
    f1s = []
    train_times = []
    test_times = []


    kg1_file = f"{data_dir}/PARIS_plus_output/kg1.nt"
    kg2_file = f"{data_dir}/PARIS_plus_output/kg2.nt"
    start_time = time.time()
    name2entid = create_nt(data_dir, kg1_file, kg2_file, seed_aligns_path)
    paris_out_folder = run_paris(data_dir, out_folder, kg1_file, kg2_file)
    train_time = time.time() - start_time
    start_time = time.time()
    precision, recall, f1 = evaluate_paris(paris_out_folder, data_dir, threshold, name2entid, results_file)
    test_time = time.time() - start_time

    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    train_times.append(train_time)
    test_times.append(test_time)
    # os.system("rm kg1.txt")
    # os.system("rm kg2.txt")

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1s = np.array(f1s)
    train_times = np.array(train_times)
    test_times = np.array(test_times)

    print("KGA results:")
    print("precisions: ", precisions)
    print("recalls: ", recalls)
    print("f1s: ", f1s)
    print("Train times: ", train_times)
    print("Test times: ", test_times)

    with open(f"{data_dir}/PARIS_plus_output/results.txt", "w", encoding="utf-8") as f:
        print("KGA results:", file=f)
        print("precisions: ", precisions, file=f)
        print("recalls: ", recalls, file=f)
        print("f1s: ", f1s, file=f)
        print("Train times: ", train_times, file=f)
        print("Test times: ", test_times, file=f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/kgbnu3/dgw/AgentEA/data/EA/dbp15k_zh_en")
    parser.add_argument("--threshold", type=float, default=0)
    parser.add_argument("--results_file", type=str, default="./tools/PARIS_plus_EA/results.txt")
    parser.add_argument("--seed_aligns", type=str, default=None, help="Path to seed alignments file")
    args = parser.parse_args()

    out_folder = f"{args.data_dir}/PARIS_plus_output"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    run_paris_experiment(args.data_dir, out_folder, args.threshold, args.seed_aligns, args.results_file)