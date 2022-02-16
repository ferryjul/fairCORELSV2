#include "queue.hh"
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <math.h>
//#include "filtering_algorithms.cpp"
//#include "milp_pruning_cplex.hh"
#include "pruning_interfaces.hh"
//#include <time.h> // for cp filtering running time measures
#include <ctime> // for solver RT measurements
#include <numeric> // for solver RT measurements

Queue::Queue(std::function<bool(Node*, Node*)> cmp, char const *type)
    : q_(new q (cmp)), type_(type) {}

Queue::~Queue() {
    if(q_)
        delete q_;
}

/* Computes confusion matrices for both groups */
bool debug = false; // for printing more info while running/exploring
int pushingTicket = 0;
int pruningCnt = 0;
int permBound = 0;
int arriveHere = 0;
unsigned long exploredNodes = 0;
bool firstPass = true;
bool firstPass2 = true;
unsigned long nodesBeforeBest = 0;
unsigned long cacheBeforeBest = 0;
int filtering_modeG = -1;
// Global variables for improved pruning
int nb_sp_plus;
int nb_sp_minus;
int nb_su_plus;
int nb_su_minus;
bool firstCall = true;
int improvedPruningCnt = 0;
int improvedPruningCntTot = 0;
double longestfilteringrun = -1.0;
int best_rl_length = 0;
double total_solver_calls = 0;
long double total_solving_time = 0.0;
solver_args args_longest_run;
int max_depth = 0;
int timeoutCnt = 0;
// for new upper bound (pruning)
rule_t* incons_min_errs;
VECTOR incons_remaining;
int Gupper_bound_filtering = 0;
int Gpruning_memoisation = 0;
int U_improved = 0;
std::vector<double> timesTot; 
std::clock_t total_start;
std::clock_t total_end;

// -----------------------------
typedef std::tuple<double, std::string, int, int> timestamp_sol_upd_t; // tuple for advanced memoisation (stores (L, U, res))
//                 obj f., Running time + unit, # nodes, cache size
std::vector<timestamp_sol_upd_t> listOfSolUpd;

double max2El(double e1, double e2) {
    if(e1 < e2) {
        return e2;
    } else {
        return e1;
    }
}

double min2El(double e1, double e2) {
    if(e1 > e2) {
        return e2;
    } else {
        return e1;
    }
}

// reduced version, working slightly faster, for prefix only (used for PPC filtering variables init)
confusion_matrix_groups compute_confusion_matrix_prefix(VECTOR parent_prefix_predictions,
                                                CacheTree* tree,
                                                VECTOR parent_not_captured, 
                                                rule_t* maj_v,
                                                rule_t* min_v,
                                                bool prediction, 
                                                bool default_prediction){

    // datastructures to store the results
    confusion_matrix_groups cmg;
    confusion_matrix cm_minority;
    confusion_matrix cm_majority;
    int nsamples = tree->nsamples();
    int pm;

    // true positives, false negatives, true negatives, and false positives
    VECTOR TP, FP, FN, TN;
    rule_vinit(nsamples, &TP);
    rule_vinit(nsamples, &FP);
    rule_vinit(nsamples, &FN);
    rule_vinit(nsamples, &TN);

    rule_vand(TP, parent_prefix_predictions, tree->label(1).truthtable, nsamples, &pm);
    rule_vand(FP, parent_prefix_predictions, tree->label(0).truthtable, nsamples, &pm);
    rule_vandnot(FN, tree->label(1).truthtable, parent_prefix_predictions, nsamples, &pm);
    rule_vandnot(TN, tree->label(0).truthtable, parent_prefix_predictions, nsamples, &pm);

    // restrict to instances captured by prefix
    rule_vandnot(TP, TP, parent_not_captured, nsamples, &pm);
    rule_vandnot(FP, FP, parent_not_captured, nsamples, &pm);
    rule_vandnot(FN, FN, parent_not_captured, nsamples, &pm);
    rule_vandnot(TN, TN, parent_not_captured, nsamples, &pm);

    // true positives, false negatives, true negatives, and false positives for majority group
    VECTOR TP_maj, FP_maj, FN_maj, TN_maj;
    rule_vinit(tree->nsamples(), &TP_maj);
    rule_vinit(tree->nsamples(), &FP_maj);
    rule_vinit(tree->nsamples(), &FN_maj);
    rule_vinit(tree->nsamples(), &TN_maj);

    int nTP_maj, nFP_maj, nFN_maj, nTN_maj;
    rule_vand(TP_maj, TP, maj_v[1].truthtable, nsamples, &nTP_maj);
    rule_vand(FP_maj, FP, maj_v[1].truthtable, nsamples, &nFP_maj);
    rule_vand(FN_maj, FN, maj_v[1].truthtable, nsamples, &nFN_maj);
    rule_vand(TN_maj, TN, maj_v[1].truthtable, nsamples, &nTN_maj);
    
    
    // true positives, false negatives, true negatives, and false positives for minority group
    VECTOR TP_min, FP_min, FN_min, TN_min;
    rule_vinit(nsamples, &TP_min);
    rule_vinit(nsamples, &FP_min);
    rule_vinit(nsamples, &FN_min);
    rule_vinit(nsamples, &TN_min);

    int nTP_min, nFP_min, nFN_min, nTN_min;
    rule_vand(TP_min, TP, min_v[1].truthtable, nsamples, &nTP_min);
    rule_vand(FP_min, FP, min_v[1].truthtable, nsamples, &nFP_min);
    rule_vand(FN_min, FN, min_v[1].truthtable, nsamples, &nFN_min);
    rule_vand(TN_min, TN, min_v[1].truthtable, nsamples, &nTN_min);


    cmg.minority.nTP = nTP_min;
    cmg.majority.nTP = nTP_maj;
    cmg.minority.nFP = nFP_min;
    cmg.majority.nFP = nFP_maj;
    cmg.minority.nTN = nTN_min;
    cmg.majority.nTN = nTN_maj;
    cmg.minority.nFN = nFN_min;
    cmg.majority.nFN = nFN_maj;
    rule_vfree(&TP);
    rule_vfree(&FP);
    rule_vfree(&FN);
    rule_vfree(&TN);
    rule_vfree(&TP_maj);
    rule_vfree(&FP_maj);
    rule_vfree(&FN_maj);
    rule_vfree(&TN_maj);
    rule_vfree(&TP_min);
    rule_vfree(&FP_min);
    rule_vfree(&FN_min);
    rule_vfree(&TN_min);

    return cmg;
}

int countUncapturedIncons(VECTOR parent_not_captured, int nsamples){
    int ret = 0;

    rule_vand(incons_remaining, parent_not_captured, incons_min_errs[0].truthtable, nsamples, &ret);
    // the if block below is for verification purposes only and should remain commented except for debug
    //if(count_ones_vector(incons_remaining, nsamples)!=ret){
    //    printf("count_ones_vector(incons_remaining, nsamples)=%d, ret=%d\n", count_ones_vector(incons_remaining, nsamples), ret);
    //    printf("abnormal, exiting\n");
    //    exit(-1);
    //}
    return ret;
}

confusion_matrix_groups compute_confusion_matrix(VECTOR parent_prefix_predictions,
                                                CacheTree* tree,
                                                VECTOR parent_not_captured, 
                                                VECTOR captured,  
                                                rule_t* maj_v,
                                                rule_t* min_v,
                                                bool prediction, 
                                                bool default_prediction){

    // datastructures to store the results
    confusion_matrix_groups cmg;
    confusion_matrix cm_minority;
    confusion_matrix cm_majority;


    int nsamples = tree->nsamples();
    int pm, num_not_captured;
    VECTOR preds_prefix, not_captured;

    rule_vinit(nsamples, &not_captured);
    rule_vinit(nsamples, &preds_prefix);

    rule_vandnot(not_captured, parent_not_captured, captured, nsamples, &num_not_captured);

    rule_copy(preds_prefix, parent_prefix_predictions, nsamples);
    
    if(default_prediction) { // else it is already OK
        rule_vor(preds_prefix, preds_prefix, not_captured, nsamples, &pm);
    }

    if(prediction) { // else it is already OK
        rule_vor(preds_prefix, preds_prefix, captured, nsamples, &pm);
    }

    // true positives, false negatives, true negatives, and false positives
    VECTOR TP, FP, FN, TN;
    rule_vinit(nsamples, &TP);
    rule_vinit(nsamples, &FP);
    rule_vinit(nsamples, &FN);
    rule_vinit(nsamples, &TN);

    rule_vand(TP, preds_prefix, tree->label(1).truthtable, nsamples, &pm);
    rule_vand(FP, preds_prefix, tree->label(0).truthtable, nsamples, &pm);
    rule_vandnot(FN, tree->label(1).truthtable, preds_prefix, nsamples, &pm);
    rule_vandnot(TN, tree->label(0).truthtable, preds_prefix, nsamples, &pm);

    // true positives, false negatives, true negatives, and false positives for majority group
    VECTOR TP_maj, FP_maj, FN_maj, TN_maj;
    rule_vinit(tree->nsamples(), &TP_maj);
    rule_vinit(tree->nsamples(), &FP_maj);
    rule_vinit(tree->nsamples(), &FN_maj);
    rule_vinit(tree->nsamples(), &TN_maj);

    int nTP_maj, nFP_maj, nFN_maj, nTN_maj;
    rule_vand(TP_maj, TP, maj_v[1].truthtable, nsamples, &nTP_maj);
    rule_vand(FP_maj, FP, maj_v[1].truthtable, nsamples, &nFP_maj);
    rule_vand(FN_maj, FN, maj_v[1].truthtable, nsamples, &nFN_maj);
    rule_vand(TN_maj, TN, maj_v[1].truthtable, nsamples, &nTN_maj);
    
    
    // true positives, false negatives, true negatives, and false positives for minority group
    VECTOR TP_min, FP_min, FN_min, TN_min;
    rule_vinit(nsamples, &TP_min);
    rule_vinit(nsamples, &FP_min);
    rule_vinit(nsamples, &FN_min);
    rule_vinit(nsamples, &TN_min);

    int nTP_min, nFP_min, nFN_min, nTN_min;
    rule_vand(TP_min, TP, min_v[1].truthtable, nsamples, &nTP_min);
    rule_vand(FP_min, FP, min_v[1].truthtable, nsamples, &nFP_min);
    rule_vand(FN_min, FN, min_v[1].truthtable, nsamples, &nFN_min);
    rule_vand(TN_min, TN, min_v[1].truthtable, nsamples, &nTN_min);

    // stats for majority
    double nPPV_maj = (double) nTP_maj / max((nTP_maj + nFP_maj), 1);
    double nTPR_maj = (double) nTP_maj / max((nTP_maj + nFN_maj), 1);

    double nFDR_maj = (double) nFP_maj / max((nFP_maj + nTP_maj), 1);
    double nFPR_maj = (double) nFP_maj / max((nFP_maj + nTN_maj), 1);

    double nFOR_maj = (double) nFN_maj / max((nFN_maj + nTN_maj), 1);
    double nFNR_maj = (double) nFN_maj / max((nFN_maj + nTP_maj), 1);

    double nNPV_maj = (double) nTN_maj / max((nTN_maj + nFN_maj), 1);
    double nTNR_maj = (double) nTN_maj / max((nTN_maj + nFP_maj), 1);

    cm_majority.nTP = nTP_maj;
    cm_majority.nFP = nFP_maj;
    cm_majority.nFN = nFN_maj;
    cm_majority.nTN = nTN_maj;

    cm_majority.nPPV = nPPV_maj;
    cm_majority.nTPR = nTPR_maj;
    cm_majority.nFDR = nFDR_maj;
    cm_majority.nFPR = nFPR_maj;
    cm_majority.nFOR = nFOR_maj;
    cm_majority.nFNR = nFNR_maj;
    cm_majority.nNPV = nNPV_maj;
    cm_majority.nTNR = nTNR_maj;


    // stats for minority
    double nPPV_min = (double) nTP_min / max((nTP_min + nFP_min), 1);
    double nTPR_min = (double) nTP_min / max((nTP_min + nFN_min), 1);
    double nFDR_min = (double) nFP_min / max((nFP_min + nTP_min), 1);
    double nFPR_min = (double) nFP_min / max((nFP_min + nTN_min), 1);
    double nFOR_min = (double) nFN_min / max((nFN_min + nTN_min), 1);
    double nFNR_min = (double) nFN_min / max((nFN_min + nTP_min), 1);
    double nNPV_min = (double) nTN_min / max((nTN_min + nFN_min), 1);
    double nTNR_min = (double) nTN_min / max((nTN_min + nFP_min), 1);

    cm_minority.nTP = nTP_min;
    cm_minority.nFP = nFP_min;
    cm_minority.nFN = nFN_min;
    cm_minority.nTN = nTN_min;

    cm_minority.nPPV = nPPV_min;
    cm_minority.nTPR = nTPR_min;
    cm_minority.nFDR = nFDR_min;
    cm_minority.nFPR = nFPR_min;
    cm_minority.nFOR = nFOR_min;
    cm_minority.nFNR = nFNR_min;
    cm_minority.nNPV = nNPV_min;
    cm_minority.nTNR = nTNR_min;


    // restrict to instances captured by prefix
    int nminTP_min, nminFP_min, nminFN_min, nminTN_min;
    rule_vandnot(TP_min, TP_min, not_captured, nsamples, &nminTP_min);
    rule_vandnot(FP_min, FP_min, not_captured, nsamples, &nminFP_min);
    rule_vandnot(FN_min, FN_min, not_captured, nsamples, &nminFN_min);
    rule_vandnot(TN_min, TN_min, not_captured, nsamples, &nminTN_min);

    cm_minority.nminTP = nminTP_min;
    cm_minority.nminFP = nminFP_min;
    cm_minority.nminFN = nminFN_min;
    cm_minority.nminTN = nminTN_min;

    int nminTP_maj, nminFP_maj, nminFN_maj, nminTN_maj;
    rule_vandnot(TP_maj, TP_maj, not_captured, nsamples, &nminTP_maj);
    rule_vandnot(FP_maj, FP_maj, not_captured, nsamples, &nminFP_maj);
    rule_vandnot(FN_maj, FN_maj, not_captured, nsamples, &nminFN_maj);
    rule_vandnot(TN_maj, TN_maj, not_captured, nsamples, &nminTN_maj);

    cm_majority.nminTP = nminTP_maj;
    cm_majority.nminFP = nminFP_maj;
    cm_majority.nminFN = nminFN_maj;
    cm_majority.nminTN = nminTN_maj;

    cmg.majority = cm_majority;
    cmg.minority = cm_minority;

    rule_vfree(&not_captured);
    rule_vfree(&preds_prefix);
    rule_vfree(&TP);
    rule_vfree(&FP);
    rule_vfree(&FN);
    rule_vfree(&TN);
    rule_vfree(&TP_maj);
    rule_vfree(&FP_maj);
    rule_vfree(&FN_maj);
    rule_vfree(&TN_maj);
    rule_vfree(&TP_min);
    rule_vfree(&FP_min);
    rule_vfree(&FN_min);
    rule_vfree(&TN_min);
    return cmg;
}

/* Computes fairness metrics given confusion matrices of both groups */
fairness_metrics compute_fairness_metrics(confusion_matrix_groups cmg){
    fairness_metrics metrics;
    
    // statistical_parity
    double statistical_parity_maj = (double) (cmg.majority.nTP + cmg.majority.nFP) / 
                                                max((cmg.majority.nTP + cmg.majority.nFP + cmg.majority.nFN + cmg.majority.nTN),1);
                                
    double statistical_parity_min = (double) (cmg.minority.nTP + cmg.minority.nFP) / 
                                                max((cmg.minority.nTP + cmg.minority.nFP + cmg.minority.nFN + cmg.minority.nTN),1);
                                
                                
    metrics.statistical_parity =  fabs(statistical_parity_maj - statistical_parity_min);

    // predictive parity
    metrics.predictive_parity = fabs(cmg.majority.nPPV - cmg.minority.nPPV);

    // predictive equality
    metrics.predictive_equality = fabs(cmg.majority.nFPR - cmg.minority.nFPR);

    // equal opportunity
    metrics.equal_opportunity = fabs(cmg.majority.nFNR - cmg.minority.nFNR);

    // equalized_odds
    //metrics.equalized_odds = fabs(cmg.majority.nFNR - cmg.minority.nFNR) + fabs(cmg.majority.nFPR - cmg.minority.nFPR);
    metrics.equalized_odds = max(fabs(cmg.majority.nFNR - cmg.minority.nFNR), fabs(cmg.majority.nFPR - cmg.minority.nFPR));

    // cond_use_acc_equality
    metrics.cond_use_acc_equality = max(fabs(cmg.majority.nPPV - cmg.minority.nPPV), fabs(cmg.majority.nNPV - cmg.minority.nNPV));

    return metrics;
}


/*
 * Performs incremental computation on a node, evaluating the bounds and inserting into the cache,
 * queue, and permutation map if appropriate.
 * This is the function that contains the majority of the logic of the algorithm.
 *
 * parent -- the node that is going to have all of its children evaluated.
 * parent_not_captured -- the vector representing data points NOT captured by the parent.
 */
void evaluate_children(CacheTree* tree, 
                        Node* parent, 
                        tracking_vector<unsigned short, DataStruct::Tree> parent_prefix,
                        VECTOR parent_not_captured, 
                        Queue* q, 
                        PermutationMap* p,
                        double beta,
                        int fairness,
                        rule_t* maj_v,
                        rule_t* min_v,
                        int mode,
                        int filteringMode,
                        double min_fairness_acceptable,
                        double accuracyUpperBound){

    int i, len_prefix;
    len_prefix = parent->depth() + 1;

    if(best_rl_length>0){
        if(filteringMode == 5){ // Early stopping possible if priority queue is ordered by the MILP objective!
            if(parent->get_opt_bound() >= tree->min_objective()){
                return;
            }
        }
    }
    if(firstCall){
        filtering_modeG = filteringMode;
        firstCall = false;
        VECTOR captured_it;
        rule_vinit(tree->nsamples(), &captured_it);
        rule_vand(captured_it, min_v[1].truthtable, tree->label(1).truthtable, tree->nsamples(), &nb_sp_plus);
        rule_vand(captured_it, min_v[1].truthtable, tree->label(0).truthtable, tree->nsamples(), &nb_sp_minus);
        rule_vand(captured_it, maj_v[1].truthtable, tree->label(1).truthtable, tree->nsamples(), &nb_su_plus);
        rule_vand(captured_it, maj_v[1].truthtable, tree->label(0).truthtable, tree->nsamples(), &nb_su_minus);
        if(debug) {
            printf("Initializing cardinalities for SP improved pruning : \n");
            printf("Got %d protected positives, %d protected negatives, %d unprotected positives, %d unprotected negatives.\n", nb_sp_plus, nb_sp_minus, nb_su_plus, nb_su_minus);
        }
        rule_vfree(&captured_it);
        int U = accuracyUpperBound * (tree->nsamples());
        if(0) {
            printf("U is %d/%d.\n", U, tree->nsamples());
        }
        // init model for filtering with guiding if using CPLEX C++
        if(filtering_modeG == 4 || filtering_modeG == 5){
            build_model(fairness,
                nb_sp_plus,
                nb_sp_minus,
                nb_su_plus,
                nb_su_minus,
                0.5*(tree->nsamples()),
                U,
                1.0 - min_fairness_acceptable,
                Gpruning_memoisation,
                true);
        } else if(filtering_modeG == 1 || filtering_modeG == 2 || filtering_modeG == 3){ // init model for filtering if using Mistral
            mistral_init_memo(Gpruning_memoisation);
        } else if(filtering_modeG == 6 || filtering_modeG == 7 || filtering_modeG == 8){ // init model for filtering if using CPLEX C++
            build_model(fairness,
                nb_sp_plus,
                nb_sp_minus,
                nb_su_plus,
                nb_su_minus,
                0.5*(tree->nsamples()),
                U,
                1.0 - min_fairness_acceptable,
                Gpruning_memoisation,
                false);
        }

        if(debug) {
            
            if(fairness == 1 && filteringMode)
                printf("will perform improved SP pruning\n");
            else if(fairness == 2 && filteringMode)
                printf("will perform improved PP pruning\n");
            else if(fairness == 3 && filteringMode)
                printf("will perform improved PE pruning\n");
            else if(fairness == 4 && filteringMode)
                printf("will perform improved EO pruning\n");
            else if(fairness == 5 && filteringMode)
                printf("will perform improved EOdds pruning\n");
        }
        longestfilteringrun = -1;
        nodesBeforeBest = 0;
        cacheBeforeBest = 0;      
    }
    
    VECTOR captured, captured_zeros, not_captured, not_captured_zeros, not_captured_equivalent;
    int num_captured, c0, c1, captured_correct;
    int num_not_captured, d0, d1, default_correct, num_not_captured_equivalent;
    num_not_captured_equivalent = 0;
    bool prediction, default_prediction;
    double lower_bound, objective, parent_lower_bound, lookahead_bound;
    double parent_equivalent_minority;
    double equivalent_minority = 0.;
    int nsamples = tree->nsamples();
    int nrules = tree->nrules();
    double c = tree->c();
    double threshold = c * nsamples;
    rule_vinit(nsamples, &captured);
    rule_vinit(nsamples, &captured_zeros);
    rule_vinit(nsamples, &not_captured);
    rule_vinit(nsamples, &not_captured_zeros);
    rule_vinit(nsamples, &not_captured_equivalent);
    
    parent_lower_bound = parent->lower_bound();
    parent_equivalent_minority = parent->equivalent_minority();
    //int verbosity = logger->getVerbosity();
    double t0 = timestamp();

    // Compute prefix's predictions
    VECTOR captured_it, not_captured_yet, captured_zeros_j, preds_prefix, captured_prefix;

    int nb, nb2, pm;

    rule_vinit(tree->nsamples(), &captured_it);
    rule_vinit(tree->nsamples(), &not_captured_yet);
    rule_vinit(tree->nsamples(), &preds_prefix);
    rule_vinit(tree->nsamples(), &captured_zeros_j);
    //rule_vinit(tree->nsamples(), &captured_prefix);
    // Initially not_captured_yet is full of ones
    rule_vor(not_captured_yet, tree->label(0).truthtable, tree->label(1).truthtable, tree->nsamples(),&nb);

    // Initially preds_prefix is full of zeros
    rule_vclear(tree->nsamples(), preds_prefix);
    //rule_vclear(tree->nsamples(), captured_prefix);

    int depth = len_prefix;
    tracking_vector<unsigned short, DataStruct::Tree>::iterator it;

    for (it = parent_prefix.begin(); it != parent_prefix.end(); it++) {
        rule_vand(captured_it, not_captured_yet, tree->rule(*it).truthtable, tree->nsamples(), &nb);
        rule_vandnot(not_captured_yet, not_captured_yet, captured_it, tree->nsamples(), &pm);
        rule_vand(captured_zeros_j, captured_it, tree->label(0).truthtable, tree->nsamples(), &nb2);
        if(nb2 <= (nb - nb2)) { //then prediction is 1
            rule_vor(preds_prefix, preds_prefix, captured_it, tree->nsamples(), &nb);
        }
    }
    // Here occurs the Lazy pruning -------------------------------------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------------------------------------------------------
    bool prefixPassedCP = true;
    if((filteringMode == 1 || filteringMode == 3 || filteringMode == 6 || filteringMode == 8) && best_rl_length > 0 && (fairness == 1 || fairness == 3 || fairness == 4 || fairness == 5)){  // Here occurs the PPC Filtering
			int L = (1 - (tree->min_objective()  - (len_prefix*c) ) )*tree->nsamples();			
	
            float fairness_tolerence = 1-min_fairness_acceptable; // equiv max unfairness acceptable

            confusion_matrix_groups cmg = compute_confusion_matrix_prefix(preds_prefix, tree, parent_not_captured,
                                                                                 maj_v, min_v, prediction, default_prediction);

			int TPp = cmg.minority.nTP;
			int FPp = cmg.minority.nFP;
			int TNp = cmg.minority.nTN;
			int FNp = cmg.minority.nFN;
			int TPu = cmg.majority.nTP;
			int FPu = cmg.majority.nFP;
			int TNu = cmg.majority.nTN;
			int FNu = cmg.majority.nFN;
            int U;
            if(Gupper_bound_filtering==0){
                U = accuracyUpperBound * (tree->nsamples());
            }else{
                int remainingInconsErrors = countUncapturedIncons(parent_not_captured, tree->nsamples());
                // this if block is for verification purposes only and should remain commented
                /*if(remainingInconsErrors != countUncapturedIncons(parent_not_captured, tree->nsamples())){
                    std::cout << "remainingInconsErrors = " << remainingInconsErrors << ", countUncapturedIncons(parent_not_captured, tree->nsamples()) = " << countUncapturedIncons(parent_not_captured, tree->nsamples())<< std::endl;
                }*/
                U =   tree->nsamples() - (remainingInconsErrors + FNp + FPp + FNu + FPu); // minimum nb of errors that it may make
                int oldU = accuracyUpperBound * (tree->nsamples());
                if(oldU>U){
                    U_improved++;
                }
                // this if block is for verification purposes only and should remain commented
                /*else if(oldU<U){
                    std::cout << "oldU = " << oldU << ", new U = " << U << std::endl;
                    exit(-1);
                }*/
                
            }
            
            if(filteringMode == 1 || filteringMode == 3){
                std::clock_t start = std::clock();
                int config = 0;
                if(fairness == 1){
                    config = 8;
                } else if(fairness == 4){
                    config = 2;
                }
                double maxSolvingTime = 5*10e9; // <- 5 seconds is already a lot, it simply helps avoiding to get stuck
                Mistral::Outcome res = runFiltering(fairness, //metric
                                    config, //solver config
                                    nb_sp_plus,nb_sp_minus, 
                                    nb_su_plus, nb_su_minus, 
                                    L,U , 
                                    fairness_tolerence, 
                                    TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu,
                                    maxSolvingTime //timeout (nanoseconds, or -1 for no timeout)
                                    );

                if(res == UNSAT){ // no solution => the fairness constraint can never be satisfied using the current prefix -> we skip its evaluation without adding it to the queue
                    improvedPruningCnt++;
                    prefixPassedCP = false;
                }   
                std::clock_t end = std::clock();
                double cpu_time_used_microsecs = ((double) (end - start) * 1000000) / CLOCKS_PER_SEC;
                timesTot.push_back(cpu_time_used_microsecs);
            }
            else if(filteringMode == 6 || filteringMode == 8){
                std::clock_t start = std::clock();
                int res_opt = prune_opt(L, U, TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu, false);
                if(res_opt < 0){ // no solution => the fairness constraint can never be satisfied using the current prefix -> we skip its evaluation without adding it to the queue
                    improvedPruningCnt++;
                    prefixPassedCP = false;
                }   
                std::clock_t end = std::clock();
                double cpu_time_used_microsecs = ((double) (end - start) * 1000000) / CLOCKS_PER_SEC;
                timesTot.push_back(cpu_time_used_microsecs);
            }
            
    }
    // ------------------------------------------------------------------------------------------------------------------------------------------------

    // begin evaluating children
    for (i = 1; prefixPassedCP && i < nrules; i++) {
        //printf("rule : %d/%d, %s\n", i, nrules, tree->rule(i).features);
        double t1 = timestamp();
        // check if this rule is already in the prefix
        if (std::find(parent_prefix.begin(), parent_prefix.end(), i) != parent_prefix.end())
            continue;
        exploredNodes++; // we consider node explored here as we below compute the preds & captured instances
        // captured represents data captured by the new rule
        rule_vand(captured, parent_not_captured, tree->rule(i).truthtable, nsamples, &num_captured);
        // lower bound on antecedent support
        if ((tree->ablation() != 1 && tree->ablation() != 3) && (num_captured < threshold))
            continue;
        rule_vand(captured_zeros, captured, tree->label(0).truthtable, nsamples, &c0);
        c1 = num_captured - c0;
        if (c0 > c1) {
            prediction = 0;
            captured_correct = c0;
        } else {
            prediction = 1;
            captured_correct = c1;
        }
        // lower bound on accurate antecedent support
        if ((tree->ablation() != 1 && tree->ablation() != 3) && (captured_correct < threshold))
            continue;
        // subtract off parent equivalent points bound because we want to use pure lower bound from parent
        lower_bound = parent_lower_bound - parent_equivalent_minority + (double)(num_captured - captured_correct) / nsamples + c;
        logger->addToLowerBoundTime(time_diff(t1));
        logger->incLowerBoundNum();
        if (lower_bound >= tree->min_objective()) // hierarchical objective lower bound
	        continue;
        double t2 = timestamp();
        rule_vandnot(not_captured, parent_not_captured, captured, nsamples, &num_not_captured);
        rule_vand(not_captured_zeros, not_captured, tree->label(0).truthtable, nsamples, &d0);
        d1 = num_not_captured - d0;
        if (d0 > d1) {
            default_prediction = 0;
            default_correct = d0;
        } else {
            default_prediction = 1;
            default_correct = d1;
        }

        //double misc = (double)(num_not_captured - default_correct) / nsamples;
        double unfairness = 0.0;

        confusion_matrix_groups cmg = compute_confusion_matrix(preds_prefix, tree, parent_not_captured, captured,
                                                                                 maj_v, min_v, prediction, default_prediction);

        improvedPruningCntTot++;
        // for debug purposes
        if(depth > max_depth && debug){
            max_depth = depth;
            printf("Now working at depth %d.\n", max_depth);
        }    

        bool filteringOK = true;
        int res_opt = tree->nsamples();

        int TPp = cmg.minority.nminTP;
        int FPp = cmg.minority.nminFP;
        int TNp = cmg.minority.nminTN;
        int FNp = cmg.minority.nminFN;
        int TPu = cmg.majority.nminTP;
        int FPu = cmg.majority.nminFP;
        int TNu = cmg.majority.nminTN;
        int FNu = cmg.majority.nminFN;
        confusion_matrix_data* node_confusion_matrix_groups = new confusion_matrix_data{TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu};// = std::make_tuple(TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu);

        fairness_metrics fm = compute_fairness_metrics(cmg);
        
        switch (fairness)
        {
            case 1:
                unfairness = fm.statistical_parity;
                //cmg.unfairnessLB = 0; // cancels the effect of the simple bound, now useless as we use improved cp filtering
                break;
            case 2:
                unfairness = fm.predictive_parity;
                //cmg.unfairnessLB = cmg.predparityLB;
                break;
            case 3:
                unfairness = fm.predictive_equality;
                //cmg.unfairnessLB = cmg.predequalityLB;
                break;
            case 4:
                unfairness = fm.equal_opportunity;
                //cmg.unfairnessLB = 0; // cancels the effect of the simple bound, now useless as we use improved cp filtering
                break;
            case 5:
                unfairness = fm.equalized_odds;
                //cmg.unfairnessLB = 0; 
                break;
            case 6:
                unfairness = fm.cond_use_acc_equality;
                //cmg.unfairnessLB = 0;
                break;
            default:
                break;
        }

        /* --- compute the objective function --- */
        if(mode == 2) { // Max fairness
            objective = unfairness + lower_bound;
        } else if(mode == 3) { // Espilon Constraint mode
            double misc = (double)(num_not_captured - default_correct) / nsamples;
            objective =  misc + lower_bound;
        } else if(mode == 4) { // Max accuracy 
            double misc = (double)(num_not_captured - default_correct) / nsamples;
            objective =  misc + lower_bound;
        } else { // Regularized mode
            double misc = (double)(num_not_captured - default_correct) / nsamples;
            /* Distance-to-a-reference objective function */
            /*double unfairnessObjective = 0.0;
            double accuracyObjective = 1.0;
            double distance = sqrt((beta*squareCalc(unfairness - unfairnessObjective)) + ((1-beta)*squareCalc((1 - misc) - accuracyObjective)));
            objective = distance + lower_bound;*/
            /* Weighted sum of objective functions */
            //objective = distance + lower_bound;
            objective =  (1-beta)*misc + beta*unfairness + lower_bound;
        }
        logger->addToObjTime(time_diff(t2));
        logger->incObjNum();
        if (objective < tree->min_objective()) {
            if(mode == 3) { // if mode 3 we check if the constraint on fairness is satisfied
                if((1-unfairness) > min_fairness_acceptable) {
                    std::clock_t now = std::clock();
                    std::string elapsed_str_ms = std::to_string((1000*((double) (now - total_start)) / CLOCKS_PER_SEC));

                    best_rl_length = len_prefix;
                    listOfSolUpd.push_back(std::make_tuple(objective, elapsed_str_ms, exploredNodes, tree->num_nodes()));

                    nodesBeforeBest = exploredNodes;
                    cacheBeforeBest = tree->num_nodes();      
                    logger->setTreeMinObj(objective);
                    tree->update_min_objective(objective);
                    tree->update_opt_rulelist(parent_prefix, i);
                    tree->update_opt_predictions(parent, prediction, default_prediction);
                    logger->dumpState();      
                }
            } else {                
                logger->setTreeMinObj(objective);
                tree->update_min_objective(objective);
                tree->update_opt_rulelist(parent_prefix, i);
                tree->update_opt_predictions(parent, prediction, default_prediction);
                // dump state when min objective is updated
                logger->dumpState();
            }
        } // && best_rl_length > 0 <- Was also in condition of the 'if' below !
        // calculate equivalent points bound to capture the fact that the minority points can never be captured correctly
        if (tree->has_minority()) {
            rule_vand(not_captured_equivalent, not_captured, tree->minority(0).truthtable, nsamples, &num_not_captured_equivalent);
            equivalent_minority = (double)(num_not_captured_equivalent) / nsamples;
            lower_bound += equivalent_minority;
        }
        if (tree->ablation() != 2 && tree->ablation() != 3)
            lookahead_bound = lower_bound + c;
        else
            lookahead_bound = lower_bound;
        
        bool toodeep = false;
        if(depth >= 8){
            toodeep = true;
        }

        if(lookahead_bound < tree->min_objective() && !toodeep){
            // Here occurs the Eager pruning -------------------------------------------------------------------------------------------------------------------
            // ------------------------------------------------------------------------------------------------------------------------------------------------
            if((filteringMode == 2 || filteringMode == 3 || filteringMode == 4 || filteringMode == 5 || filteringMode == 7 || filteringMode == 8)  && (fairness == 1 || fairness == 3 || fairness == 4 || fairness == 5)){  // Here occurs the PPC Filtering
                double l_required=(1 - (tree->min_objective()  - ((len_prefix+1)*c) ) )*tree->nsamples();
                int L = ceil(l_required);
                if((L-l_required) == 0){
                    L++;
                }
                // check L's calibration:
                if(((len_prefix+1)*c+((double)(tree->nsamples()-L)/(double)tree->nsamples()))>=tree->min_objective()){
                    // this line is for verification purposes only and should remain commented
                    //std::cout << "tree->min_objective()=" << tree->min_objective() << ", " << "old needed = " << ((len_prefix+1)*c+((double)(tree->nsamples()-L)/(double)tree->nsamples())) << ", new needed :" << ((len_prefix+1)*c+((double)(tree->nsamples()-L-1)/(double)tree->nsamples())) << std::endl;
                    L++;
                }

                float fairness_tolerence = 1-min_fairness_acceptable; // equiv max unfairness acceptable


                int U;
                if(Gupper_bound_filtering==0){
                    U = accuracyUpperBound * (tree->nsamples());
                }else{
                    // this block is for verification purposes only and should remain commented
                    //int remainingInconsErrors = num_not_captured_equivalent;//countUncapturedIncons(not_captured, tree->nsamples());
                    /*if(num_not_captured_equivalent != remainingInconsErrors){
                        std::cout << "num_not_captured_equivalent = " << num_not_captured_equivalent << ", remainingInconsErrors = " << remainingInconsErrors << std::endl;
                    }*/
                    U =   tree->nsamples() - (num_not_captured_equivalent + FNp + FPp + FNu + FPu);
                    int oldU = accuracyUpperBound * (tree->nsamples());
                    if(oldU>U){
                        U_improved++;
                    }
                    // this if block is for verification purposes only and should remain commented
                    /*else if(oldU<U){
                        std::cout << "oldU = " << oldU << ", new U = " << U << std::endl;
                        exit(-1);
                    }*/
                    
                }
                if(filteringMode == 2 || filteringMode == 3){
                    std::clock_t start = std::clock();
                    int config = 0;
                    if(fairness == 1){
                        config = 8;
                    } else if(fairness == 4){
                        config = 2;
                    }
                    double maxSolvingTime = 5*10e9; // <- 5 seconds is already a lot, it simply helps avoiding to get stuck
                    Mistral::Outcome res = runFiltering(fairness, //metric
                                        config, //solver config
                                        nb_sp_plus,nb_sp_minus, 
                                        nb_su_plus, nb_su_minus, 
                                        L,U , 
                                        fairness_tolerence, 
                                        TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu,
                                        maxSolvingTime //timeout (nanoseconds, or -1 for no timeout)
                                        );

                    if(res == UNSAT){ // no solution => the fairness constraint can never be satisfied using the current prefix -> we skip its evaluation without adding it to the queue
                        improvedPruningCnt++;
                        filteringOK = false;
                    }   
                    std::clock_t end = std::clock();
                    double cpu_time_used_microsecs = ((double) (end - start) * 1000000) / CLOCKS_PER_SEC;
                    timesTot.push_back(cpu_time_used_microsecs);
                } else if(filteringMode == 4 || filteringMode == 5 || filteringMode == 7 || filteringMode == 8){
                    std::clock_t start = std::clock();
                    //res_opt = compute_pruning_opt(fairness, nb_sp_plus, nb_sp_minus, nb_su_plus, nb_su_minus, L, U, fairness_tolerence, TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu, false);
                    res_opt = prune_opt(L, U, TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu, false);
                    if(res_opt < 0){ // no solution => the fairness constraint can never be satisfied using the current prefix -> we skip its evaluation without adding it to the queue
                        improvedPruningCnt++;
                        filteringOK = false;
                    }   
                    std::clock_t end = std::clock();
                    double cpu_time_used_microsecs = ((double) (end - start) * 1000000) / CLOCKS_PER_SEC;
                    timesTot.push_back(cpu_time_used_microsecs);
                }
            }
        }
        // ------------------------------------------------------------------------------------------------------------------------------------------------
        

        // only add node to our datastructures if its children will be viable
        double opt_bound = ((len_prefix+1)*c)+(((double)tree->nsamples() - (double)res_opt)/(double)tree->nsamples());
        if ((filteringOK && !toodeep) && (lookahead_bound < tree->min_objective()) && ((filtering_modeG != 4 && filtering_modeG != 5) || (opt_bound <= tree->min_objective()))) { //&& (fairnesslb>=min_fairness_acceptable)
            arriveHere++;
            double t3 = timestamp();
            // check permutation bound
            Node* n = p->insert(i, nrules, prediction, default_prediction,
                                   lower_bound, objective, parent, num_not_captured, nsamples,
                                   len_prefix, c, equivalent_minority, tree, not_captured, parent_prefix, node_confusion_matrix_groups);
            delete node_confusion_matrix_groups;
            logger->addToPermMapInsertionTime(time_diff(t3));
            // n is NULL if this rule fails the permutaiton bound
            if (n) {
                pushingTicket++;
                n->set_num(pushingTicket);
                n->set_unfairness(unfairness);
                // opt bound for queue ordering
                if(filtering_modeG == 4 || filtering_modeG == 5){
                    if(opt_bound>tree->min_objective()){
                        std::cout << "Unexpected situation: opt_bound=" << opt_bound << ", and tree->min_objective()=" << tree->min_objective() << std::endl;
                        std::cout << "res_opt=" << res_opt << ", objective=" << objective << std::endl;
                        int L = (1 - (tree->min_objective()  - ((len_prefix+1)*c) ) )*tree->nsamples();
                        std::cout << "L given to CPLEX was " << L << ", tree->min_objective() = " << tree->min_objective() << ", tree->nsamples() = " << tree->nsamples() << ", len_prefix = " << len_prefix << ", c = " << c << std::endl;
                        std::cout << "min obj given to CPLEX was " << ((len_prefix+1)*c+((double)(tree->nsamples()-L)/(double)tree->nsamples())) << std::endl;
                        exit(-1);
                    }
                }
                n->set_opt_bound(opt_bound);
                double t4 = timestamp();
                tree->insert(n);
                logger->incTreeInsertionNum();
                logger->incPrefixLen(len_prefix);
                logger->addToTreeInsertionTime(time_diff(t4));
                double t5 = timestamp();
                q->push(n);
                logger->setQueueSize(q->size());
                if (tree->calculate_size())
                    logger->addQueueElement(len_prefix, lower_bound, false);
                logger->addToQueueInsertionTime(time_diff(t5));
            } else {
                permBound++;
            }
        } // else:  objective lower bound with one-step lookahead
    }
    rule_vfree(&captured_it);
    //rule_vfree(&captured_prefix);
    rule_vfree(&not_captured_yet);
    rule_vfree(&preds_prefix);
    rule_vfree(&captured_zeros_j);


    rule_vfree(&captured);
    rule_vfree(&captured_zeros);
    rule_vfree(&not_captured);
    rule_vfree(&not_captured_zeros);
    rule_vfree(&not_captured_equivalent);

    logger->addToRuleEvalTime(time_diff(t0));
    logger->incRuleEvalNum();
    logger->decPrefixLen(parent->depth());
    if (tree->calculate_size())
        logger->removeQueueElement(len_prefix - 1, parent_lower_bound, false);
    if (parent->num_children() == 0) {
        tree->prune_up(parent);
    } else {
        parent->set_done();
        tree->increment_num_evaluated();
    }
}

static size_t num_iter = 0;
static double min_objective = 0.0;
static VECTOR captured, not_captured;
static double start = 0.0;

/*
 * Explores the search space by using a queue to order the search process.
 * The queue can be ordered by DFS, BFS, or an alternative priority metric (e.g. lower bound).
 */
void bbound_begin(CacheTree* tree, Queue* q, rule_t* G_incons_min_errs, int upper_bound_filtering, int pruning_memoisation) {
    Gpruning_memoisation = pruning_memoisation;
    Gupper_bound_filtering = upper_bound_filtering;
    if(Gupper_bound_filtering > 0){
        rule_vinit(tree->nsamples(), &incons_remaining);
    }        
    total_start = std::clock();
    incons_min_errs = G_incons_min_errs;
    start = timestamp();
    num_iter = 0;
    rule_vinit(tree->nsamples(), &captured);
    rule_vinit(tree->nsamples(), &not_captured);

    logger->setInitialTime(start);
    logger->initializeState(tree->calculate_size());
    // initial log record
    logger->dumpState();         

    min_objective = 1.0;
    tree->insert_root();
    logger->incTreeInsertionNum();
    q->push(tree->root());
    logger->setQueueSize(q->size());
    logger->incPrefixLen(0);
    // log record for empty rule list
    logger->dumpState();
    listOfSolUpd.push_back(std::make_tuple(tree->min_objective(), "0", 0, 0));
}

void bbound_loop(CacheTree* tree, 
                Queue* q, 
                PermutationMap* p,
                double beta,
                int fairness,
                rule_t* maj_v,
                rule_t* min_v,
                int mode,
                int filteringMode,
                double min_fairness_acceptable,
                int kBest,
                double accuracyUpperBound){

    double t0 = timestamp();
    int verbosity = logger->getVerbosity();
    size_t queue_min_length = logger->getQueueMinLen();
    int cnt;
    std::pair<Node*, tracking_vector<unsigned short, DataStruct::Tree> > node_ordered = q->select(kBest, tree, captured);
    logger->addToNodeSelectTime(time_diff(t0));
    logger->incNodeSelectNum();
    if (node_ordered.first) {
        double t1 = timestamp();
        // not_captured = default rule truthtable & ~ captured
        rule_vandnot(not_captured,
                     tree->rule(0).truthtable, captured,
                     tree->nsamples(), &cnt);
        evaluate_children(tree, node_ordered.first, node_ordered.second, not_captured, q, p, beta, fairness, maj_v, min_v, mode, filteringMode,
                        min_fairness_acceptable, accuracyUpperBound);
        logger->addToEvalChildrenTime(time_diff(t1));
        logger->incEvalChildrenNum();

        if (tree->min_objective() < min_objective) {
            min_objective = tree->min_objective();
            if (verbosity >= 10)
                printf("before garbage_collect. num_nodes: %zu\n", tree->num_nodes());
            logger->dumpState();
            tree->garbage_collect();
            logger->dumpState();
            if (verbosity >= 10)
                printf("after garbage_collect. num_nodes: %zu\n", tree->num_nodes());
        }
    }
    logger->setQueueSize(q->size());
    if (queue_min_length < logger->getQueueMinLen()) {
        // garbage collect the permutation map: can be simplified for the case of BFS
        queue_min_length = logger->getQueueMinLen();
        //pmap_garbage_collect(p, queue_min_length);
    }
    ++num_iter;
    if ((num_iter % 10000) == 0) {
        if (verbosity >= 10){
            printf("iter: %zu, tree: %zu, queue: %zu, pmap: %zu, time elapsed: %f\n",
                   num_iter, tree->num_nodes(), q->size(), p->size(), time_diff(start));
            printf("using new filtering pruned %d/%d nodes, min_obj is %f.\n", improvedPruningCnt, improvedPruningCntTot, tree->min_objective());
            printf("res_opt is %f\n", node_ordered.first->get_opt_bound());
        }
    }
    if ((num_iter % logger->getFrequency()) == 0) {
        // want ~1000 records for detailed figures
        logger->dumpState();
    }
}
    
std::vector<unsigned long> bbound_end(CacheTree* tree, Queue* q, PermutationMap* p, bool early, rule_t* rules, rule_t* labels, int exitCode, std::string fileName, int verbose_pruning) {
    int verbosity = logger->getVerbosity();
    bool print_queue = 0;
    total_end = std::clock();
    // save solution updates to a file-
    std::ofstream myFile(fileName);
    myFile << "Objective, Running_Time(ms), #Explored, Cache_Size" << std::endl;

    for(int index = 0; index < listOfSolUpd.size(); index ++){
        myFile << std::get<0>(listOfSolUpd[index]) << "," << std::get<1>(listOfSolUpd[index]) << "," << std::get<2>(listOfSolUpd[index]) << "," << std::get<3>(listOfSolUpd[index]) << std::endl;
    }
    std::string exitCodeStr;
    switch(exitCode){
        case -1:{
            exitCodeStr="UNFITTED";
            break;
        }
        case 1:{
            exitCodeStr="TRIE_SIZE_OUT";
            break;
        }
        case 2:{
            exitCodeStr="EXPL_OUT";
            break;
        }
        case 3:{
            exitCodeStr="OPT";
            break;
        }
        case 4:{
            exitCodeStr="TIME_OUT";
            break;
        }
        case 5:{
            exitCodeStr="MEMORY_OUT";
            break;
        }
        default:{
            exitCodeStr="ERROR";
            break;
        }
    }
    std::string endingTime = std::to_string(1000*((double) (total_end - total_start)) / CLOCKS_PER_SEC);// + " s.";
    myFile << "Exit_Status," << endingTime << "," << exitCodeStr << ",endOfData" << std::endl;
    if(filtering_modeG >=1 && filtering_modeG <= 8){
       myFile << "Solver_Average_RT," << std::accumulate(timesTot.begin(), timesTot.end(), 0)/timesTot.size() << " microsecs,," << std::endl;
    }
    
    myFile.close();
    // --------------------------------
    if(verbose_pruning > 0){
        std::cout << "Total CPU time = " << ((double) (total_end - total_start)) / CLOCKS_PER_SEC << " seconds." << std::endl;
    }
    if(Gupper_bound_filtering > 0){
        //std::cout << "Improved upper bound " << U_improved << "/" << improvedPruningCntTot << " times." << std::endl;
        rule_vfree(&incons_remaining);
    }   

    if(filtering_modeG == 4 || filtering_modeG == 5 || filtering_modeG == 6 || filtering_modeG == 7 || filtering_modeG == 8){
        end_cplex(verbose_pruning);
    } else if(filtering_modeG == 1 || filtering_modeG == 2 || filtering_modeG == 3){
        mistral_clear_memo(verbose_pruning);
    }
    if(filtering_modeG >=1 && filtering_modeG <= 8){
        if(verbose_pruning > 0){
            std::cout << "Average solving time was " << std::accumulate(timesTot.begin(), timesTot.end(), 0)/timesTot.size() << " microsecs." << std::endl;        
        }
    }
    if(debug) {
        printf("explored %lu nodes.\n", exploredNodes);
        printf("using new filtering pruned %d/%d nodes.\n", improvedPruningCnt, improvedPruningCntTot);
        printf("Total solving time = %Lf s\n", total_solving_time/1000000.0);
        printf("Longest fitlering run took %f ms.\n", longestfilteringrun/1000.0);
        printf("Average time per solver call = %Lf ms\n", (total_solving_time/1000.0)/total_solver_calls);
        printf("%d/%f solver calls timed out.\n", timeoutCnt, total_solver_calls);
        printf("Number of nodes in the trie at exit : %d\n",  tree->num_nodes());
        printf("params : (%d,%d,%d,%d,%d,%d,%f,%d,%d,%d,%d,%d,%d,%d,%d)\n",args_longest_run.nb_sp_plus,
                    args_longest_run.nb_sp_minus,
                    args_longest_run.nb_su_plus,
                    args_longest_run.nb_su_minus,
                    args_longest_run.L,
                    args_longest_run.U,
                    args_longest_run.fairness_tolerence,
                    args_longest_run.TPp,
                    args_longest_run.FPp,
                    args_longest_run.TNp,
                    args_longest_run.FNp,
                    args_longest_run.TPu,
                    args_longest_run.FPu,
                    args_longest_run.TNu,
                    args_longest_run.FNu);
    }
    improvedPruningCnt = 0;
    improvedPruningCntTot = 0;
    longestfilteringrun = -1.0;
    total_solving_time = 0.0;
    total_solver_calls = 0.0;
    max_depth = 0;
    pushingTicket = 0;
    pruningCnt = 0;
    best_rl_length = 0;
    exploredNodes = 0;
    firstPass = true;
    firstPass2 = true;
    firstCall = true;
    logger->dumpState(); // second last log record (before queue elements deleted)
   // if(pruningCnt > 0)
        //printf("Pruned %d subtrees with unfairness lower bound.\n", pruningCnt);
    if (verbosity >= 5)
        printf("iter: %zu, tree: %zu, queue: %zu, pmap: %zu, time elapsed: %f\n",
               num_iter, tree->num_nodes(), q->size(), p->size(), time_diff(start));
    
    if (!early) {
        if (q->empty()) {
            if (verbosity >= 1) 
                printf("Exited because queue empty\n");
        }
        else if (verbosity >= 1)
            printf("Exited because max number of nodes in the tree was reached\n");
    }

    // Print out queue
    ofstream f;
    if (print_queue) {
        char fname[] = "queue.txt";
        if (verbosity >= 1) {
            printf("Writing queue elements to: %s\n", fname);
        }
        f.open(fname, ios::out | ios::trunc);
        f << "lower_bound objective length frac_captured rule_list\n";
    }

    // Clean up data structures
    if (verbosity >= 1) {
        printf("Deleting queue elements and corresponding nodes in the cache,"
            "since they may not be reachable by the tree's destructor\n");
        printf("\nminimum objective: %1.10f\n", tree->min_objective());
    }
    Node* node;
    double min_lower_bound = 1.0;
    double lb;
    size_t num = 0;
    while (!q->empty()) {
        node = q->front();
        q->pop();
        if (node->deleted()) {
            tree->decrement_num_nodes();
            logger->removeFromMemory(sizeof(*node), DataStruct::Tree);
            delete node;
        } else {
            lb = node->lower_bound() + tree->c();
            if (lb < min_lower_bound)
                min_lower_bound = lb;
            if (print_queue) {
                std::pair<tracking_vector<unsigned short, DataStruct::Tree>, tracking_vector<bool, DataStruct::Tree> > pp_pair = node->get_prefix_and_predictions();
                tracking_vector<unsigned short, DataStruct::Tree> prefix = std::move(pp_pair.first);
                tracking_vector<bool, DataStruct::Tree> predictions = std::move(pp_pair.second);
                f << node->lower_bound() << " " << node->objective() << " " << node->depth() << " "
                  << (double) node->num_captured() / (double) tree->nsamples() << " ";
                for(size_t i = 0; i < prefix.size(); ++i) {
                    f << tree->rule_features(prefix[i]) << "~"
                      << predictions[i] << ";";
                }
                f << "default~" << predictions.back() << "\n";
                num++;
            }
        }
    }
    
    /* Compute confidence scores and exact accuracy */
    compData res = computeFinalFairness(tree->nsamples(), 
                         tree->opt_rulelist(), 
                         tree->opt_predictions(),
                         rules,
                         labels);
    
    tree->setConfScores(res.conf_scores);
    tree->setFinalAcc(res.accuracy);
    if (verbosity >= 1)
        printf("minimum lower bound in queue: %1.10f\n\n", min_lower_bound);
    
    if (print_queue)
        f.close();
    // last log record (before cache deleted)
    logger->dumpState();

    rule_vfree(&captured);
    rule_vfree(&not_captured);
    std::vector<unsigned long> returnVal;
    returnVal.push_back(nodesBeforeBest);
    returnVal.push_back(cacheBeforeBest);

    return returnVal;
}
