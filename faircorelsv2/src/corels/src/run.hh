#ifndef RUN_H
#define RUN_H

#include "rule.hh"

#ifdef __cplusplus
extern "C" {
#endif

int run_corels_begin(double c, char* vstring, int curiosity_policy,
                  int map_type, int ablation, int calculate_size, int nrules, int nlabels,
                  int nsamples, rule_t* rules, rule_t* labels, rule_t* meta, int freq, 
                  char* log_fname, int BFSmode, int seed, rule_t* maj_v, 
                  int nmaj_v, rule_t* min_v, int nmin_v, double accuracy_upper_bound, int max_calls,
                  rule_t* incons_minerrs_vecs, int nbInconsErrs, int upper_bound_filtering, int pruning_memoisation, int filteringMode);

int run_corels_loop(size_t max_num_nodes, double beta, int fairness, int mode,
                        double min_fairness_acceptable, int kBest);

double run_corels_end(int** rulelist, int* rulelist_size, int** classes, double** confScores, int early, int latex_out, rule_t* rules, rule_t* labels, char* opt_fname, unsigned long** runStats, int exitCode, char* fileName, int fileNameLen);

#ifdef __cplusplus
}
#endif

#endif