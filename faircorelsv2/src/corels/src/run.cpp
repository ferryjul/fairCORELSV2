#include <stdio.h>
#include <iostream>
#include <set>

#include "queue.hh"
#include "run.hh"

#define BUFSZ 512

NullLogger* logger = nullptr;
static PermutationMap* g_pmap = nullptr;
static CacheTree* g_tree = nullptr;
static Queue* g_queue = nullptr;
double g_init = 0.0;
std::set<std::string> g_verbosity;
rule_t * Grules;
rule_t * Glabels;
rule_t * Gmeta;

int callsNB = 1000;
int initCallsNB = 1000;
double accUpperBound;
rule_t* Gmaj_vect;
rule_t* Gmin_vect;
bool debugRun = false; // for printing more info while running/exploring
rule_t* Gincons_minerrs_vecs;
int upper_bound_filteringG;
int filteringModeGlob = -1;

int run_corels_begin(double c, char* vstring, int curiosity_policy,
                  int map_type, int ablation, int calculate_size, int nrules, int nlabels,
                  int nsamples, rule_t* rules, rule_t* labels, rule_t* meta, int freq, 
                  char* log_fname, int BFSmode, int seed, rule_t* maj_v, int nmaj_v,
                  rule_t* min_v, int nmin_v, double accuracy_upper_bound, int max_calls, 
                  rule_t* incons_minerrs_vecs, int nbInconsErrs, int upper_bound_filtering, int pruning_memoisation, int filteringMode)
{  
    Gincons_minerrs_vecs =  incons_minerrs_vecs;
    upper_bound_filteringG = upper_bound_filtering;
    filteringModeGlob = filteringMode;
    callsNB = max_calls;
    initCallsNB = max_calls;
    if(debugRun) {
        printf("Will explore at most %d nodes.\n", callsNB);
        printf("Provided accuracy upper bound = %lf\n", accuracy_upper_bound);
    }
    
    accUpperBound = accuracy_upper_bound;
    int nbInconsErrsMes = count_ones_vector(incons_minerrs_vecs[0].truthtable, nmaj_v);
    if(nbInconsErrsMes != nbInconsErrs){
        printf("#Incons mismatch: %d ones in vector, expected %d.\n", nbInconsErrsMes, nbInconsErrs);
        exit(-1);
    }
    // Check correctness
    if(nmaj_v != nmin_v){
        printf("Incorrect argument : nmaj and nmin should be equal\n");
        exit(-1);
    }
    int nbMaj = count_ones_vector(maj_v[1].truthtable, nmaj_v);
    int nbMin = count_ones_vector(min_v[1].truthtable, nmin_v);
    //printf("Maj vector : captures %d/%d instances.\n", nbMaj, nmaj_v);
    //printf("Min vector : captures %d/%d instances.\n", nbMin, nmin_v);
    if((nbMaj + nbMin) > nsamples){
        printf("Incorrect argument : majority and minority groups intersection should be empty.\n");
    }
    Gmaj_vect =  maj_v;
    Gmin_vect =  min_v;

    int nbMajG = count_ones_vector(Gmaj_vect[1].truthtable, nmaj_v);
    int nbMinG = count_ones_vector(Gmin_vect[1].truthtable, nmin_v);
    if((nbMajG + nbMinG) > nsamples){
        printf("Error : Internal error in vector copy \n");
    }
    //printf("Maj vector G : captures %d/%d instances.\n", nbMajG, nmaj_v);
    //printf("Min vector G : captures %d/%d instances.\n", nbMinG, nmin_v);

    Grules = rules;
    Glabels = labels;
    Gmeta = meta;
    //printf("seed = %d\n", seed);
    srand(seed);
    // Check arguments
    if(BFSmode < 0 || BFSmode > 4) {
        printf("Error : BFSmode should be in {1, 2, 3, 4}\n");
        exit(-1);
    }
    g_verbosity.clear();

    const char *voptions = "rule|label|minor|samples|progress|pruning|loud";

    char *vopt = NULL;
    char *vcopy = m_strdup(vstring);
    char *vcopy_begin = vcopy;
    while ((vopt = m_strsep(&vcopy, ',')) != NULL) {
        if (!strstr(voptions, vopt)) {
            fprintf(stderr, "verbosity options must be one or more of (%s)\n", voptions);
            return -1;
        }
        g_verbosity.insert(vopt);
    }
    free(vcopy_begin);

    if (g_verbosity.count("loud")) {
        g_verbosity.insert("progress");
        g_verbosity.insert("label");
        g_verbosity.insert("rule");
        g_verbosity.insert("minor");
        g_verbosity.insert("pruning");
    }

#ifndef GMP
    if (g_verbosity.count("progress"))
        printf("**Not using GMP library**\n");
#endif

    if (g_verbosity.count("rule")) {
        printf("%d rules %d samples\n\n", nrules, nsamples);
        rule_print_all(rules, nrules, nsamples, g_verbosity.count("samples"));
        printf("\n\n");
    }

    if (g_verbosity.count("label")) {
        printf("Labels (%d) for %d samples\n\n", nlabels, nsamples);
        rule_print_all(labels, nlabels, nsamples, g_verbosity.count("samples"));
        printf("\n\n");
    }

    if (g_verbosity.count("minor") && meta) {
        printf("Minority bound for %d samples\n\n", nsamples);
        rule_print_all(meta, 1, nsamples, g_verbosity.count("samples"));
        printf("\n\n");
    }

    if (g_verbosity.count("pruning")) {
        
        switch(filteringMode){
            case 0:
            {
                printf("Will perform no advanced pruning.\n");
                break;
            }
            case 1:
            {
                printf("Will perform Lazy pruning using the Mistral solver.\n");
                break;
            }
            case 2:
            {
                printf("Will perform Eager pruning using the Mistral solver.\n");
                break;
            }
            case 3:
            {
                printf("Will perform jointly Eager and Lazy pruning using the Mistral solver.\n");
                printf("This option is not recommended. It performs the highest possible pruning, with many possibly redundant calls to the solver.\n");
                break;
            }
            case 4:
            {
                printf("Will perform Eager pruning using the CPLEX solver.\n");
                printf("MILP Objective will be used to order the priority queue to break ties within a BFS strategy.\n");
                break;
            }
            case 5:
            {
                printf("Will perform Eager pruning using the CPLEX solver.\n");
                printf("MILP Objective will be used to order the priority queue.\n");
                break;
            }
            case 6:
            {
                printf("Will perform Eager pruning using the CPLEX solver.\n");
                break;
            }
            case 7:
            {
                printf("Will perform Eager pruning using the CPLEX solver.\n");
                break;
            }
            case 8:
            {
                printf("Will perform jointly Eager and Lazy pruning using the CPLEX solver.\n");
                printf("This option is not recommended. It performs the highest possible pruning, with many possibly redundant calls to the solver.\n");
                break;
            }   
            default:
                printf("Should never happen. Exiting.\n");
        }
        if(filteringMode > 0){
            if(upper_bound_filtering == 0){
                printf("Upper bound computation is not tight.\n");
            }
            if(upper_bound_filtering == 1){
                printf("Upper bound computation is tight.\n");
            }
        }
    }

    if (g_tree)
        delete g_tree;
    g_tree = nullptr;

    if (g_queue)
        delete g_queue;
    g_queue = nullptr;

    if (g_pmap)
        delete g_pmap;
    g_pmap = nullptr;

    int v = 0;
    if (g_verbosity.count("loud"))
        v = 1000;
    else if (g_verbosity.count("progress"))
        v = 1;

    if(!logger) {
        if(log_fname)
            logger = new Logger(c, nrules, v, log_fname, freq);
        else {
            logger = new PyLogger();
            logger->setVerbosity(v);
        }
    }

    g_init = timestamp();
    char run_type[BUFSZ];
    strcpy(run_type, "LEARNING RULE LIST via ");
    char const *type = "node";
    if(filteringModeGlob == 4){ // BFS-MILP guided priority queue ordering
        strcat(run_type, "MILP guided BFS");
        g_queue = new Queue(filtering_based_bfs_cmp, run_type);
    }
    else if(filteringModeGlob == 5){ // MILP guided priority queue ordering
        strcat(run_type, "MILP guided");
        g_queue = new Queue(filtering_based_cmp, run_type);
    }
    else if (curiosity_policy == 1) { // else we use the policy argument to define the priority queue ordering
        strcat(run_type, "CURIOUS");
        g_queue = new Queue(curious_cmp, run_type);
        type = "curious";
    } else if (curiosity_policy == 2) {
        strcat(run_type, "LOWER BOUND");
        g_queue = new Queue(lb_cmp, run_type);
    } else if (curiosity_policy == 3) {
        strcat(run_type, "OBJECTIVE");
        g_queue = new Queue(objective_cmp, run_type);
    } else if (curiosity_policy == 4) {
        strcat(run_type, "DFS");
        g_queue = new Queue(dfs_cmp, run_type);
    } else {
        strcat(run_type, "BFS");
        switch(BFSmode) {
            case 0:
                g_queue = new Queue(base_cmp, run_type);
                break;
            case 1:
                g_queue = new Queue(base_cmp_fifo, run_type);
                break;
            case 2:
                g_queue = new Queue(base_cmp_obj, run_type);
               // printf("Using objective-aware BFS");
                break;
            case 3:
                g_queue = new Queue(base_cmp_lb, run_type);
                break;
            case 4:
                g_queue = new Queue(base_cmp_random, run_type);
                break;
        }
    }

    
    //std::cout << run_type << std::endl;
    if (map_type == 1) {
        strcat(run_type, " Prefix Map\n");
        PrefixPermutationMap* prefix_pmap = new PrefixPermutationMap;
        g_pmap = (PermutationMap*) prefix_pmap;
    } else if (map_type == 2) {
        strcat(run_type, " Captured Symmetry Map\n");
        CapturedPermutationMap* cap_pmap = new CapturedPermutationMap;
        g_pmap = (PermutationMap*) cap_pmap;
    } else {
        strcat(run_type, " No Permutation Map\n");
        NullPermutationMap* null_pmap = new NullPermutationMap;
        g_pmap = (PermutationMap*) null_pmap;
    }

    g_tree = new CacheTree(nsamples, nrules, c, rules, labels, meta, ablation, calculate_size, type);
    if (g_verbosity.count("progress"))
        printf("%s", run_type);
    bbound_begin(g_tree, g_queue, Gincons_minerrs_vecs, upper_bound_filteringG, pruning_memoisation);
    return 0;
}

int run_corels_loop(size_t max_num_nodes, double beta, int fairness, int mode,
                        double min_fairness_acceptable, int kBest) {


    // Normal run (no restart)
    // if:
    // - not reached the max #nodes in trie
    // - there are nodes remaining in queue 
    // - not reached max #calls
    // then keep exploring
    if((g_tree->num_nodes() < max_num_nodes) && !g_queue->empty() && (callsNB > 0)) {
        bbound_loop(g_tree, g_queue, g_pmap, beta, fairness, Gmaj_vect, Gmin_vect, mode, filteringModeGlob,
                        min_fairness_acceptable, kBest, accUpperBound);
        callsNB--;
        return 0;
    }
    
    // Now check terminal conditions
    if(max_num_nodes <= g_tree->num_nodes()){
        if(g_verbosity.count("pruning")){
            printf("Exiting because max #nodes in the trie was reached : %d/%d\n", max_num_nodes, g_tree->num_nodes());
        }
        return 1;
    } else if(callsNB <= 0){
        if(g_verbosity.count("pruning")){
            printf("Performed max allowed #calls to bbound_loop (%d)\n", initCallsNB);
        }
        return 2;
    } else if(g_queue->empty()){
        if(g_verbosity.count("pruning")){
            printf("Optimum found and proved!\n");
        }
        return 3;
    }
    return -1;
}

double run_corels_end(int** rulelist, int* rulelist_size, int** classes, double** confScores, int early, int latex_out, rule_t* rules, rule_t* labels, char* opt_fname, unsigned long** runStats, int exitCode, char* fileName, int fileNameLen)
{
    if(debugRun){
        printf("Performed %d calls to bbound_loop.\n", initCallsNB - callsNB);
    }

    std::string fileNameString(fileName, fileNameLen);
    std::vector<unsigned long> vals = bbound_end(g_tree, g_queue, g_pmap, early, Grules, Glabels, exitCode, fileNameString, g_verbosity.count("pruning"));
    const tracking_vector<unsigned short, DataStruct::Tree>& r_list = g_tree->opt_rulelist();
    const tracking_vector<bool, DataStruct::Tree>& preds = g_tree->opt_predictions();
    const vector<double> scores = g_tree->getConfScores();
    *runStats = (unsigned long*)malloc(sizeof(unsigned long) * 2); // Confidence scores
    if(debugRun){
        printf("nb explored = %lu, nb cache = %lu\n", vals[0], vals[1]);
    }
    (*runStats)[0] = vals[0];
    (*runStats)[1] = vals[1];
    //double accuracy = 1.0 - g_tree->min_objective() + g_tree->c() * r_list.size();
    double accuracy = g_tree->getFinalAcc();
    *rulelist = (int*)malloc(sizeof(int) * r_list.size()); // Antecedents
    *classes = (int*)malloc(sizeof(int) * (1 + r_list.size())); // Consequents
    *confScores = (double*)malloc(sizeof(double) * (1 + r_list.size())); // Confidence scores
    *rulelist_size = r_list.size();
    for(size_t i = 0; i < r_list.size(); i++) {
        (*rulelist)[i] = r_list[i]; // Condition i
        (*confScores)[i] = scores[i]; // Confidence score for rule i
        (*classes)[i] = preds[i]; // Pred i
    }
    (*confScores)[r_list.size()] = (scores)[r_list.size()];
    (*classes)[r_list.size()] = preds.back(); // Default prediction
    if (g_verbosity.count("progress")) {
        printf("final num_nodes: %zu\n", g_tree->num_nodes());
        printf("final num_evaluated: %zu\n", g_tree->num_evaluated());
        printf("final min_objective: %1.5f\n", g_tree->min_objective());
        printf("final accuracy: %1.5f\n", accuracy);
        printf("final total time: %f\n", time_diff(g_init));
    }
    if(opt_fname) {
        print_final_rulelist(r_list, g_tree->opt_predictions(), latex_out, Grules, Glabels, opt_fname, g_tree->getConfScores());
        logger->dumpState();
        logger->closeFile();
    }
    if(g_tree)
        delete g_tree;
    g_tree = nullptr;
    if(g_pmap)
        delete g_pmap;
    g_pmap = nullptr;
    if(g_queue)
        delete g_queue;
    g_queue = nullptr;

    return accuracy;
}
