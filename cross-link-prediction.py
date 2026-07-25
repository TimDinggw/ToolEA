import os
import shutil
import argparse
from collections import defaultdict


def load_triples(file_path):
    """Load triples from a file"""
    triples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                h, r, t = parts[0], parts[1], parts[2]
                triples.append((h, r, t))
    return triples


def load_alignment_pairs(file_path):
    """Load alignment pairs from a file"""
    pairs = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                e1, e2 = parts[0], parts[1]
                pairs[e2] = e1  # e2 maps to e1
    return pairs


def apply_alignment_to_triples(triples, alignment_pairs):
    """Apply alignment mapping to triples"""
    aligned_triples = []
    for h, r, t in triples:
        new_h = alignment_pairs.get(h, h)
        new_t = alignment_pairs.get(t, t)
        aligned_triples.append((new_h, r, new_t))
    return aligned_triples


def create_combined_training_data(rel_triples_1_path, rel_triples_2_path, seed_aligns_path, output_dir):
    """Create combined training data using alignment information"""
    # Load data
    triples_1 = load_triples(rel_triples_1_path)
    triples_2 = load_triples(rel_triples_2_path)
    alignment_pairs = load_alignment_pairs(seed_aligns_path)
    
    # Align triples from graph 2
    aligned_triples_2 = apply_alignment_to_triples(triples_2, alignment_pairs)
    
    # Create output directory if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Write combined training data
    combined_path = os.path.join(output_dir, "cross_train.txt")
    with open(combined_path, 'w', encoding='utf-8') as f:
        # Write triples from graph 1
        for h, r, t in triples_1:
            f.write(f"{h}\t{r}\t{t}\n")
        # Write aligned triples from graph 2
        for h, r, t in aligned_triples_2:
            f.write(f"{h}\t{r}\t{t}\n")
    
    print(f"Combined training data saved to {combined_path}")
    print(f"Total triples in combined training set: {len(triples_1) + len(triples_2)}")
    
    return combined_path


def prepare_test_data(cross_test_path, seed_aligns_path, output_dir):
    """Prepare test data using alignment information"""
    # Load test data and alignment pairs
    test_triples = load_triples(cross_test_path)
    alignment_pairs = load_alignment_pairs(seed_aligns_path)
    
    # Apply alignment to test data
    aligned_test_triples = apply_alignment_to_triples(test_triples, alignment_pairs)
    
    # Save aligned test data
    test_path = os.path.join(output_dir, "cross_test.txt")
    with open(test_path, 'w', encoding='utf-8') as f:
        for h, r, t in aligned_test_triples:
            f.write(f"{h}\t{r}\t{t}\n")
    
    print(f"Test data saved to {test_path}")
    print(f"Total triples in test set: {len(test_triples)}")
    
    return test_path


def run_rule_learning(train_path, output_dir):
    """Run rule learning on the training data"""
    # Import clause modules
    try:
        from clause import Learner, Options
        from clause.util.utils import get_base_dir
        from c_clause import Loader
    except ImportError as e:
        print(f"Error importing clause modules: {e}")
        print("Please install the required packages for rule learning.")
        return None
    
    # Create output directory for rules
    rules_output_path = os.path.join(output_dir, "pyclause_output", "cross-rules.txt")
    rules_dir = os.path.dirname(rules_output_path)
    if not os.path.exists(rules_dir):
        os.makedirs(rules_dir)
    
    # Set up options for rule learning
    options = Options()
    options.set("learner.mode", "anyburl")
    options.set("learner.anyburl.time", 120)
    options.set("learner.anyburl.raw.MAX_LENGTH_CYCLIC", 5)
    options.set("learner.anyburl.raw.WORKER_THREADS", 10)
    options.set("learner.anyburl.java_options", ["-Dfile.encoding=UTF-8"])
    
    # Learn rules
    learner = Learner(options=options.get("learner"))
    learner.learn_rules(path_data=train_path, path_output=rules_output_path)
    
    print(f"Rules learned and saved to {rules_output_path}")
    
    # Load rules into c_clause
    options.set("loader.c_max_length", 5)
    loader = Loader(options.get("loader"))
    loader.load_data(data=train_path)
    loader.load_rules(rules=rules_output_path)
    
    return rules_output_path


def run_evaluation(train_path, test_path, rules_path, output_dir):
    """Run evaluation on test data using learned rules"""
    try:
        from c_clause import RankingHandler, Loader, PredictionHandler
        from clause import Options, Ranking, TripleSet
    except ImportError as e:
        print(f"Error importing evaluation modules: {e}")
        print("Please install the required packages for evaluation.")
        return
    
    # Prepare paths for evaluation
    filterset_path = os.path.join(output_dir, "filterset.txt")
    ranking_file = os.path.join(output_dir, "pyclause_output", "cross-ranking.txt")
    explanations_file = os.path.join(output_dir, "pyclause_output", "cross-explanations.txt")
    
    # Create filter set (training + validation data)
    with open(train_path, "r", encoding="utf-8") as f_train, \
         open(filterset_path, "w", encoding="utf-8") as f_filter:
        f_filter.writelines(f_train.readlines())
    
    # Set up options for evaluation
    options = Options()
    options.set("ranking_handler.aggregation_function", "maxplus")
    options.set("ranking_handler.topk", 200)
    options.set("loader.load_u_d_rules", True)
    options.set("loader.load_u_xxc_rules", False)
    options.set("loader.load_u_xxd_rules", False)
    options.set("ranking_handler.num_threads", 64)
    
    # Load data and rules
    loader = Loader(options=options.get("loader"))
    loader.load_data(data=train_path, filter=filterset_path, target=test_path)
    loader.load_rules(rules=rules_path)
    
    # Run ranking evaluation
    ranker = RankingHandler(options=options.get("ranking_handler"))
    ranker.calculate_ranking(loader=loader)
    head_ranking = ranker.get_ranking(direction="head", as_string=True)
    tail_ranking = ranker.get_ranking(direction="tail", as_string=True)
    
    # Evaluate results
    testset = TripleSet(test_path, encod="utf-8")
    ranking = Ranking(k=200)
    ranking.convert_handler_ranking(head_ranking, tail_ranking, testset)
    ranking.compute_scores(testset.triples)
    
    # Print results
    print("\n*** CROSS-LINK PREDICTION EVALUATION RESULTS ***")
    print("Num triples: " + str(len(testset.triples)))
    print("MRR     " + '{0:.6f}'.format(ranking.hits.get_mrr()))
    print("hits@1  " + '{0:.6f}'.format(ranking.hits.get_hits_at_k(1)))
    print("hits@3  " + '{0:.6f}'.format(ranking.hits.get_hits_at_k(3)))
    print("hits@5  " + '{0:.6f}'.format(ranking.hits.get_hits_at_k(5)))
    print("hits@10 " + '{0:.6f}'.format(ranking.hits.get_hits_at_k(10)))
    
    # Save results
    results_path = os.path.join(output_dir, "pyclause_output", "cross-results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("*** CROSS-LINK PREDICTION EVALUATION RESULTS ***\n")
        f.write("Num triples: " + str(len(testset.triples)) + "\n")
        f.write("MRR     " + '{0:.6f}'.format(ranking.hits.get_mrr()) + "\n")
        f.write("hits@1  " + '{0:.6f}'.format(ranking.hits.get_hits_at_k(1)) + "\n")
        f.write("hits@3  " + '{0:.6f}'.format(ranking.hits.get_hits_at_k(3)) + "\n")
        f.write("hits@5  " + '{0:.6f}'.format(ranking.hits.get_hits_at_k(5)) + "\n")
        f.write("hits@10 " + '{0:.6f}'.format(ranking.hits.get_hits_at_k(10)) + "\n")
    
    # Write ranking results
    ranker.write_ranking(path=ranking_file, loader=loader)
    
    print(f"Evaluation results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Process DBP-FB dataset for cross-link prediction")
    parser.add_argument("--data_dir", type=str, default="/kgbnu3/dgw/AgentEA/data/dbp-fb", 
                        help="Directory containing the dbp-fb dataset")
    parser.add_argument("--output_dir", type=str, default="/kgbnu3/dgw/AgentEA/output/dbp-fb-cross-prediction",
                        help="Output directory for processed data and results")
    
    args = parser.parse_args()
    
    # Define paths
    rel_triples_1_path = os.path.join(args.data_dir, "rel_triples_1")
    rel_triples_2_path = os.path.join(args.data_dir, "rel_triples_2")
    



    #seed_aligns_path = os.path.join(args.data_dir, "seed_aligns")
    seed_aligns_path = "/kgbnu3/dgw/AgentEA/output/round_5_alignments_all.txt"




    cross_test_path = os.path.join(args.data_dir, "cross_test")
    
    # Validate input files exist
    for path in [rel_triples_1_path, rel_triples_2_path, seed_aligns_path, cross_test_path]:
        if not os.path.exists(path):
            print(f"Error: File does not exist: {path}")
            return
    
    print("Starting cross-link prediction processing for DBP-FB dataset...")
    print(f"Input data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Step 1: Create combined training data
    print("\nStep 1: Creating combined training data...")
    train_path = create_combined_training_data(rel_triples_1_path, rel_triples_2_path, 
                                               seed_aligns_path, args.output_dir)
    
    # Step 2: Prepare test data
    print("\nStep 2: Preparing test data...")
    test_path = prepare_test_data(cross_test_path, seed_aligns_path, args.output_dir)
    
    # Step 3: Learn rules
    print("\nStep 3: Learning rules from training data...")
    rules_path = run_rule_learning(train_path, args.output_dir)
    
    if rules_path is not None:
        # Step 4: Evaluate on test data
        print("\nStep 4: Evaluating on test data...")
        run_evaluation(train_path, test_path, rules_path, args.output_dir)
        print(f"\nProcessing completed! Results saved in: {args.output_dir}")
    else:
        print("\nSkipping evaluation due to rule learning failure.")


if __name__ == "__main__":
    main()