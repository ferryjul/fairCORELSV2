#pragma once

#ifndef PRUNING_HH
#define PRUNING_HH

#include <mistral_solver.hpp>
#include <mistral_variable.hpp>
#include <mistral_search.hpp>
#include "memoisation_utils.hh"
/*
extern int compute_pruning_opt(int fairnessMetric,
            int nb_sp_plus,
			int nb_sp_minus,
			int nb_su_plus,
			int nb_su_minus,
			int L,
			int U,
			double fairness_tolerence,
			int TPp,
			int FPp,
			int TNp,
			int FNp,
			int TPu,
			int FPu,
			int TNu,
			int FNu,
            bool check_response);
*/

// CPLEX RELATED FUNCTIONS
extern int build_model(int fairnessMetric,
         int nb_sp_plus,
			int nb_sp_minus,
			int nb_su_plus,
			int nb_su_minus,
			int L,
			int U,
			double fairness_tolerence,
            int memoisation,
			bool optimization);

extern int prune_opt(int L,
			int U,
            int TPp,
			int FPp,
			int TNp,
			int FNp,
			int TPu,
			int FPu,
			int TNu,
			int FNu,
            bool check_response);

extern int end_cplex(int verbose);

// MISTRAL RELATED FUNCTIONS
/*struct runResult {
	
	Declared values for Outcome :
	#define SAT      1
	#define OPT      3
	#define UNSAT    0
	#define UNKNOWN  2
	#define LIMITOUT 4
	
	Mistral::Outcome result;
	unsigned long solvingTime;
} runResult;*/

extern int mistral_init_memo(int memoisation);

/*extern struct runResult runFiltering(
			int metric, 
			int solverConfig,
			int nb_sp_plus,
			int nb_sp_minus,
			int nb_su_plus,
			int nb_su_minus,
			int L,
			int U,
			float tolerence,
			int TPp,
			int FPp,
			int TNp,
			int FNp,
			int TPu,
			int FPu,
			int TNu,
			int FNu,
			double timeout
			);*/

extern Mistral::Outcome runFiltering(
			int metric, 
			int solverConfig,
			int nb_sp_plus,
			int nb_sp_minus,
			int nb_su_plus,
			int nb_su_minus,
			int L,
			int U,
			float tolerence,
			int TPp,
			int FPp,
			int TNp,
			int FNp,
			int TPu,
			int FPu,
			int TNu,
			int FNu,
			double timeout
			);

extern int mistral_clear_memo(int verbose);

#endif /* PRUNING_HH */

