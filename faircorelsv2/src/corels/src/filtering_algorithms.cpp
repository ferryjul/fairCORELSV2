//#include "memoisation_utils.hh"
#include "pruning_interfaces.hh"

using namespace std;

// Base class
class FilteringAlgorithm {

	public:
		/*	ub_sp_plus is the number of examples protected with a positive class
			ub_sp_minus is the number of examples protected with a negative class
			ub_su_plus is the number of examples unprotected with a positive class
			ub_su_minus is the number of examples unprotected with a negative class
			TPp is the number of True Positive protected instances (among instances captured by the prefix)
		*/
		
		FilteringAlgorithm(
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
			string metric_name):
			nb_protected( nb_sp_plus+ nb_sp_minus),
			nb_unprotected( nb_su_plus+ nb_su_minus),
			nb_protected_negative (nb_sp_minus),
			nb_unprotected_negative ( nb_su_minus),
			nb_protected_positive (nb_sp_plus),
			nb_unprotected_positive ( nb_su_plus),
			L(L),
			U(U),
			fairness_tolerence(fairness_tolerence),
			fairness_name(metric_name)
			{
				/* DECLARE MODEL BASE VARIABLES AND ACCURACY CONSTRAINTS */
				//scope[0] is sp_plus
				scope.add(Mistral::Variable(TPp, nb_sp_plus - FNp) );
				//scope[1] is sp_minus
				scope.add(Mistral::Variable(FPp, nb_sp_minus - TNp) );
				//scope[2] is su_plus
				scope.add(Mistral::Variable(TPu, nb_su_plus - FNu) );
				//scope[3] is su_minus
				scope.add(Mistral::Variable(FPu, nb_su_minus - TNu) );

				result =UNSAT;

				// only vars we need
				Mistral::VarArray accuracy_constraint_variables;
				accuracy_constraint_variables.add (scope[0]) ;
				accuracy_constraint_variables.add (scope[1]) ;
				accuracy_constraint_variables.add (scope[2]) ;
				accuracy_constraint_variables.add (scope[3]) ;

				std::vector<int> accuracy_coefficient ;
				// accuracy_coefficient represents the coefficients used in the accuracy.
				accuracy_coefficient.push_back(1);
				accuracy_coefficient.push_back(-1);
				accuracy_coefficient.push_back(1);
				accuracy_coefficient.push_back(-1);

				//Accuracy constraints
				int constant  = U  -  nb_sp_minus - nb_su_minus ;
				s.add( Sum(accuracy_constraint_variables, accuracy_coefficient) <= constant  ) ;

				constant  = L  -   nb_sp_minus - nb_su_minus ;

				s.add( Sum(accuracy_constraint_variables, accuracy_coefficient) >= constant  ) ;
			}

		friend std::ostream& operator<< (std::ostream &out, const FilteringAlgorithm &filtAlg);

		void set_timeout(double timeout){
			s.parameters.time_limit = timeout; // TODO add arg
		}

		void run(int verbosity, int config){
			s.parameters.verbosity = verbosity;
			//std::cout <<  s << std::endl;
			//s.rewrite();
			s.consolidate();
			if (!config)
				//use default solver strategy
				result =  s.solve();
			else
			{
				bool branch_on_decision_only = true;
				Mistral::RestartPolicy *_option_policy = NULL;
				Mistral::BranchingHeuristic *_option_heuristic ;


				switch(config) {

				case 1 :
					branch_on_decision_only= true;
					//This is used in the competition
					_option_policy = new Mistral::Luby();
					_option_heuristic = new Mistral::LastConflict < Mistral::GenericDVO < Mistral::MinDomainOverWeight, 2, Mistral::ConflictCountManager >,  Mistral::Guided< Mistral::MinValue >,  Mistral::Guided< Mistral::MinValue >, 1 > (&s);
					break;
				case 2 :
					branch_on_decision_only= true;
					_option_policy = new Mistral::Luby();
					_option_heuristic = new Mistral::LastConflict < Mistral::GenericDVO < Mistral::MinDomainOverWeight, 2, Mistral::ConflictCountManager >,  Mistral::Guided< Mistral::RandomMinMax >,  Mistral::Guided< Mistral::RandomMinMax >, 1 > (&s);
					break;

				case 3 :
					branch_on_decision_only= true;

					_option_policy = new Mistral::Luby();
					_option_heuristic = new Mistral::LastConflict < Mistral::GenericDVO < Mistral::MinDomainOverWeight, 2, Mistral::ConflictCountManager >,  Mistral::Guided< Mistral::MiddleValue >,  Mistral::Guided< Mistral::MiddleValue >, 1 > (&s);
					break;

				case 4 :
					branch_on_decision_only= true;
					_option_policy = new Mistral::Geometric();
					_option_heuristic = new Mistral::LastConflict < Mistral::GenericDVO < Mistral::MinDomainOverWeight, 2, Mistral::ConflictCountManager >,  Mistral::Guided< Mistral::MinValue >,  Mistral::Guided< Mistral::MinValue >, 1 > (&s);
					break;
				case 5 :
					branch_on_decision_only= true;
					_option_policy = new Mistral::Geometric();
					_option_heuristic = new Mistral::LastConflict < Mistral::GenericDVO < Mistral::MinDomainOverWeight, 2, Mistral::ConflictCountManager >,  Mistral::Guided< Mistral::RandomMinMax >,  Mistral::Guided< Mistral::RandomMinMax >, 1 > (&s);
					break;

				case 6 :
					branch_on_decision_only= true;
					_option_policy = new Mistral::Geometric();
					_option_heuristic = new Mistral::LastConflict < Mistral::GenericDVO < Mistral::MinDomainOverWeight, 2, Mistral::ConflictCountManager >,  Mistral::Guided< Mistral::MiddleValue >,  Mistral::Guided< Mistral::MiddleValue >, 1 > (&s);
					break;


				case 7 :
					branch_on_decision_only= true;
					_option_policy = new Mistral::NoRestart();
					_option_heuristic = new Mistral::LastConflict < Mistral::GenericDVO < Mistral::MinDomainOverWeight, 2, Mistral::ConflictCountManager >,  Mistral::Guided< Mistral::MinValue >,  Mistral::Guided< Mistral::MinValue >, 1 > (&s);
					break;
				case 8 :
					branch_on_decision_only= true;
					_option_policy = new Mistral::NoRestart();
					_option_heuristic = new Mistral::LastConflict < Mistral::GenericDVO < Mistral::MinDomainOverWeight, 2, Mistral::ConflictCountManager >,  Mistral::Guided< Mistral::RandomMinMax >,  Mistral::Guided< Mistral::RandomMinMax >, 1 > (&s);
					break;

				case 9 :
					branch_on_decision_only= true;
					_option_policy = new Mistral::NoRestart();
					_option_heuristic = new Mistral::LastConflict < Mistral::GenericDVO < Mistral::MinDomainOverWeight, 2, Mistral::ConflictCountManager >,  Mistral::Guided< Mistral::MiddleValue >,  Mistral::Guided< Mistral::MiddleValue >, 1 > (&s);
					break;

				case 10 :
					branch_on_decision_only= false;
					//This is used in the competition
					_option_policy = new Mistral::Luby();
					_option_heuristic = new Mistral::LastConflict < Mistral::GenericDVO < Mistral::MinDomainOverWeight, 2, Mistral::ConflictCountManager >,  Mistral::Guided< Mistral::MinValue >,  Mistral::Guided< Mistral::MinValue >, 1 > (&s);
					break;
				case 11 :
					branch_on_decision_only= false;
					_option_policy = new Mistral::Luby();
					_option_heuristic = new Mistral::LastConflict < Mistral::GenericDVO < Mistral::MinDomainOverWeight, 2, Mistral::ConflictCountManager >,  Mistral::Guided< Mistral::RandomMinMax >,  Mistral::Guided< Mistral::RandomMinMax >, 1 > (&s);
					break;

				case 12 :
					branch_on_decision_only= false;

					_option_policy = new Mistral::Luby();
					_option_heuristic = new Mistral::LastConflict < Mistral::GenericDVO < Mistral::MinDomainOverWeight, 2, Mistral::ConflictCountManager >,  Mistral::Guided< Mistral::MiddleValue >,  Mistral::Guided< Mistral::MiddleValue >, 1 > (&s);
					break;

				case 13 :
					branch_on_decision_only= false;
					_option_policy = new Mistral::Geometric();
					_option_heuristic = new Mistral::LastConflict < Mistral::GenericDVO < Mistral::MinDomainOverWeight, 2, Mistral::ConflictCountManager >,  Mistral::Guided< Mistral::MinValue >,  Mistral::Guided< Mistral::MinValue >, 1 > (&s);
					break;
				case 14 :
					branch_on_decision_only= false;
					_option_policy = new Mistral::Geometric();
					_option_heuristic = new Mistral::LastConflict < Mistral::GenericDVO < Mistral::MinDomainOverWeight, 2, Mistral::ConflictCountManager >,  Mistral::Guided< Mistral::RandomMinMax >,  Mistral::Guided< Mistral::RandomMinMax >, 1 > (&s);
					break;

				case 15 :
					branch_on_decision_only= false;
					_option_policy = new Mistral::Geometric();
					_option_heuristic = new Mistral::LastConflict < Mistral::GenericDVO < Mistral::MinDomainOverWeight, 2, Mistral::ConflictCountManager >,  Mistral::Guided< Mistral::MiddleValue >,  Mistral::Guided< Mistral::MiddleValue >, 1 > (&s);
					break;


				case 16 :
					branch_on_decision_only= false;
					_option_policy = new Mistral::NoRestart();
					_option_heuristic = new Mistral::LastConflict < Mistral::GenericDVO < Mistral::MinDomainOverWeight, 2, Mistral::ConflictCountManager >,  Mistral::Guided< Mistral::MinValue >,  Mistral::Guided< Mistral::MinValue >, 1 > (&s);
					break;
				case 17 :
					branch_on_decision_only= false;
					_option_policy = new Mistral::NoRestart();
					_option_heuristic = new Mistral::LastConflict < Mistral::GenericDVO < Mistral::MinDomainOverWeight, 2, Mistral::ConflictCountManager >,  Mistral::Guided< Mistral::RandomMinMax >,  Mistral::Guided< Mistral::RandomMinMax >, 1 > (&s);
					break;

				case 18 :
					branch_on_decision_only= false;
					_option_policy = new Mistral::NoRestart();
					_option_heuristic = new Mistral::LastConflict < Mistral::GenericDVO < Mistral::MinDomainOverWeight, 2, Mistral::ConflictCountManager >,  Mistral::Guided< Mistral::MiddleValue >,  Mistral::Guided< Mistral::MiddleValue >, 1 > (&s);
					break;
				default :
					std::cout << " c confid not used " << config << std::endl;

				}
				s.parameters.activity_decay = 0.95;
				s.parameters.seed = 10 ;
				if (branch_on_decision_only){
					//This one is used only when branching on decision variables
					Mistral::VarArray decision_variables;
					decision_variables.add (scope[0]) ;
					decision_variables.add (scope[1]) ;
					decision_variables.add (scope[2]) ;
					decision_variables.add (scope[3]) ;

					result = s.depth_first_search(decision_variables, _option_heuristic, _option_policy);
				}
				else
					result = s.depth_first_search(s.variables , _option_heuristic, _option_policy);

			}
		}

		double get_cpu_time(){
			return double(s.statistics.end_time - s.statistics.start_time) ;
		}

		void print_statistics(){
			s.statistics.print_full(std::cout);
		}

		bool isFeasible(){
			if (result )
			{
				return true;
			} else {
				return false;
			}
		}

		Mistral::Outcome get_outcome(){
			return result;
		}

		virtual double compute_fairness(int sp_plus, int sp_minus, int su_plus, int su_minus) = 0;

		void print_and_verify_solution(){

			if (result )
			{
				std::cout <<  " \n \n c Solution Found! "  << 	std::endl;

				//for ( unsigned int i= 0 ; i< scope.size ; ++i)
				//	std::cout <<  " c Solution value of scope[" << i <<  "]  is " << scope[i].get_solution_int_value() << std::endl;


				int sp_plus= scope[0].get_solution_int_value();
				int sp_minus= scope[1].get_solution_int_value();
				int su_plus= scope[2].get_solution_int_value();
				int su_minus= scope[3].get_solution_int_value();

				//int fairness_sum= scope[4].get_solution_int_value();

				std::cout <<  " c nb protected positive that are predicted positive: " <<  sp_plus << std::endl;
				std::cout <<  " c nb protected negative that are predicted positive: " <<  sp_minus << std::endl;
				std::cout <<  " c nb unprotected positive that are predicted positive: " <<  su_plus << std::endl;
				std::cout <<  " c nb unprotected negative that are predicted positive:  " <<  su_minus << std::endl;
				//std::cout <<  " c nb fairness_sum  " <<  fairness_sum << std::endl;
				printf(" c nb_protected_negative : %d\n", nb_protected_negative);
				printf(" c nb_unprotected_negative : %d\n", nb_unprotected_negative);
				int accuracy = sp_plus + su_plus -sp_minus - su_minus + nb_protected_negative + nb_unprotected_negative;

				double fairness = compute_fairness(sp_plus, sp_minus, su_plus, su_minus);
				std::cout <<  " c Accuracy ( Number of examamples well classified) is "  << accuracy  << " out of " << nb_protected+ nb_unprotected << 	std::endl;

				std::cout <<  " c Fairness " << fairness_name << " (as float) is "  << fairness << " (tolerence= " << fairness_tolerence << ")" << std::endl;

				assert (accuracy >= L   ) ;
				assert (accuracy <= U   ) ;
				assert (fairness <= fairness_tolerence    ) ;
				assert (fairness >= - fairness_tolerence    ) ;
				//int discrete_fairness= (nb_unprotected * (sp_plus+sp_minus)) - (nb_protected * (su_plus +su_minus) )  ;
				//std::cout <<  " c Fairness (discrete constraint) is "  << discrete_fairness << std::endl;
				std::cout <<  " c Solution Verified"  << 	std::endl;

			}
			else
				std::cout <<  " c No Solution! "  << 	std::endl;
		}


	protected:

		Mistral::VarArray scope;
		Mistral::Solver s;
		Mistral::Outcome result;
		string fairness_name;
		//
		int nb_protected, nb_unprotected, nb_protected_negative, nb_unprotected_negative, nb_protected_positive, nb_unprotected_positive ;

		//L is a lower bound for the accuracy
		int L ;

		//U is a lower bound for the accuracy
		int U;
		//fairness_tolerence is a float in [0,1] that represents the unfairness tolerated.
		//If this value is 0 then the model is 100% fair.
		double fairness_tolerence ;

};

/*ostream& operator<< (ostream &out, const FilteringAlgorithm &filtAlg){
	out << " c Instance data :  " <<  	std::endl
	<<  " c total number of examples: "  << filtAlg.nb_protected_positive + filtAlg.nb_protected_negative + filtAlg.nb_unprotected_positive + filtAlg.nb_unprotected_negative << 	std::endl
	<<  " c total number of protected examples: "  << filtAlg.nb_protected_positive + filtAlg.nb_protected_negative << 	std::endl
	<<  " c total number of unprotected examples: "  << filtAlg.nb_unprotected_positive + filtAlg.nb_unprotected_negative  << 	std::endl
	<<  " c total number of protected examples with positive class: "  << filtAlg.nb_protected_positive << 	std::endl
	<<  " c total number of protected examples with negative class: "  << filtAlg.nb_protected_negative << 	std::endl
	<<  " c total number of unprotected examples with positive class: "  << filtAlg.nb_unprotected_positive << 	std::endl
	<<  " c total number of unprotected examples with negative class: "  << filtAlg.nb_unprotected_negative << 	std::endl
	<<  " c Lower bound for accuracy: "  << filtAlg.L << 	std::endl
	<<  " c Upper bound for accuracy: "  << filtAlg.U << 	std::endl
	<<  " c Unfairness tolerance (should be in [0,1]): "  << filtAlg.fairness_tolerence << 	std::endl;
	return out;
}*/

// Models for particular metrics (extend base class)

class FilteringStatisticalParity: public FilteringAlgorithm {

public:
	/*	ub_sp_plus is the number of examples protected with a positive class
		ub_sp_minus is the number of examples protected with a negative class
		ub_su_plus is the number of examples unprotected with a positive class
		ub_su_minus is the number of examples unprotected with a negative class
		TPp is the number of True Positive protected instances (among instances captured by the prefix)
	 */
	FilteringStatisticalParity(
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
			int FNu
			 ) : FilteringAlgorithm(nb_sp_plus,
			nb_sp_minus,
			nb_su_plus,
			nb_su_minus,
			L,
			U,
			fairness_tolerence,
			TPp,
			FPp,
			TNp,
			FNp,
			TPu,
			FPu,
			TNu,
			FNu,
			"Statistical Parity")
	{

			//Fairness constraints
			std::vector<int> fairness_coefficient;

			fairness_coefficient.push_back(nb_unprotected);
			fairness_coefficient.push_back(nb_unprotected);
			fairness_coefficient.push_back(-nb_protected);
			fairness_coefficient.push_back(-nb_protected);

			int fairness_bound = (int ) ( (float) nb_protected *(float) nb_unprotected * fairness_tolerence );

			s.add( Sum(scope, fairness_coefficient) <= fairness_bound ) ;
			s.add( Sum(scope, fairness_coefficient) >= (-fairness_bound) ) ;

	}

	double compute_fairness(int sp_plus, int sp_minus, int su_plus, int su_minus){
		return ((double)(sp_plus+sp_minus) /(double) nb_protected )  - 	 ((double)(su_plus +su_minus) /(double) nb_unprotected );
	}

};

class FilteringPredictiveEquality : public FilteringAlgorithm {

public:
	/*	ub_sp_plus is the number of examples protected with a positive class
		ub_sp_minus is the number of examples protected with a negative class
		ub_su_plus is the number of examples unprotected with a positive class
		ub_su_minus is the number of examples unprotected with a negative class
		TPp is the number of True Positive protected instances (among instances captured by the prefix)
	 */
	FilteringPredictiveEquality(
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
			int FNu
			 ) : FilteringAlgorithm(nb_sp_plus,
			nb_sp_minus,
			nb_su_plus,
			nb_su_minus,
			L,
			U,
			fairness_tolerence,
			TPp,
			FPp,
			TNp,
			FNp,
			TPu,
			FPu,
			TNu,
			FNu,
			"Predictive Equality")
{
		//Fairness constraints
		std::vector<int> fairness_coefficient;
		fairness_coefficient.push_back(nb_unprotected_negative);
		fairness_coefficient.push_back(-nb_protected_negative);

		// only vars we need
		Mistral::VarArray fairness_constraint_variables;
		fairness_constraint_variables.add (scope[1]) ; // sp -
		fairness_constraint_variables.add (scope[3]) ; // su -

		int fairness_bound = (int ) ( (double) nb_unprotected_negative *(double) nb_protected_negative * fairness_tolerence );

		s.add( Sum(fairness_constraint_variables, fairness_coefficient) <= fairness_bound ) ;
		s.add( Sum(fairness_constraint_variables, fairness_coefficient) >= (-fairness_bound) ) ;
}


	double compute_fairness(int sp_plus, int sp_minus, int su_plus, int su_minus){
		return ((double)sp_minus/(double) (nb_protected_negative)  )  - 	 ((double)su_minus/(double) (nb_unprotected_negative)  );
	}



};

class FilteringEqualOpportunity : public FilteringAlgorithm{
	
public:
	/*	ub_sp_plus is the number of examples protected with a positive class
		ub_sp_minus is the number of examples protected with a negative class
		ub_su_plus is the number of examples unprotected with a positive class
		ub_su_minus is the number of examples unprotected with a negative class
		TPp is the number of True Positive protected instances (among instances captured by the prefix)
	 */
	FilteringEqualOpportunity(
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
			int FNu
			 ) : FilteringAlgorithm(nb_sp_plus,
			nb_sp_minus,
			nb_su_plus,
			nb_su_minus,
			L,
			U,
			fairness_tolerence,
			TPp,
			FPp,
			TNp,
			FNp,
			TPu,
			FPu,
			TNu,
			FNu,
			"Equal Opportunity")
{
		//std::cout << " c Solver internal representation " << s << std::endl;
		//Fairness constraints
		// only vars we need
		Mistral::VarArray fairness_constraint_variables;
		fairness_constraint_variables.add (scope[0]) ; // sp +
		fairness_constraint_variables.add (scope[2]) ; // su +

		std::vector<int> fairness_coefficient;
		fairness_coefficient.push_back(-nb_su_plus);
		fairness_coefficient.push_back(nb_sp_plus);

		int fairness_bound = (int ) ( (float) nb_su_plus *(float) nb_sp_plus * fairness_tolerence );

		s.add( Sum(fairness_constraint_variables, fairness_coefficient) <= fairness_bound ) ;
		s.add( Sum(fairness_constraint_variables, fairness_coefficient) >= (-fairness_bound) ) ;
}

	double compute_fairness(int sp_plus, int sp_minus, int su_plus, int su_minus){
		return ((double)((nb_protected_positive - sp_plus) /(double) (nb_protected_positive)  )  - 	 ((double)(nb_unprotected_positive - su_plus)/(double) (nb_unprotected_positive)  ));

	}

};

class FilteringEqualizedOdds : public FilteringAlgorithm {
	
public:
	/*	ub_sp_plus is the number of examples protected with a positive class
		ub_sp_minus is the number of examples protected with a negative class
		ub_su_plus is the number of examples unprotected with a positive class
		ub_su_minus is the number of examples unprotected with a negative class
		TPp is the number of True Positive protected instances (among instances captured by the prefix)
	 */
	FilteringEqualizedOdds(
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
			int FNu
			) : FilteringAlgorithm(nb_sp_plus,
			nb_sp_minus,
			nb_su_plus,
			nb_su_minus,
			L,
			U,
			fairness_tolerence,
			TPp,
			FPp,
			TNp,
			FNp,
			TPu,
			FPu,
			TNu,
			FNu,
			"Equalized Odds")
{

		//Fairness constraints
		// 1) For EO (FNR)
		// only vars we need
		Mistral::VarArray fairness_constraint_variables_1;
		fairness_constraint_variables_1.add (scope[0]) ; // sp +
		fairness_constraint_variables_1.add (scope[2]) ; // su +

		std::vector<int> fairness_coefficient_1;
		fairness_coefficient_1.push_back(-nb_su_plus);
		fairness_coefficient_1.push_back(nb_sp_plus);
	
		int fairness_bound_1 = (int ) ( (float) nb_su_plus *(float) nb_sp_plus * fairness_tolerence );

		s.add( Sum(fairness_constraint_variables_1, fairness_coefficient_1) <= fairness_bound_1 ) ;
		s.add( Sum(fairness_constraint_variables_1, fairness_coefficient_1) >= (-fairness_bound_1) ) ;

		// 2) For PE (FPR)
		// only vars we need
		Mistral::VarArray fairness_constraint_variables_2;
		fairness_constraint_variables_2.add (scope[1]) ; // sp -
		fairness_constraint_variables_2.add (scope[3]) ; // su -

		std::vector<int> fairness_coefficient_2;
		fairness_coefficient_2.push_back(nb_unprotected_negative);
		fairness_coefficient_2.push_back(-nb_protected_negative);

		int fairness_bound_2 = (int ) ( (float) nb_unprotected_negative *(float) nb_protected_negative * fairness_tolerence );

		s.add( Sum(fairness_constraint_variables_2, fairness_coefficient_2) <= fairness_bound_2 ) ;
		s.add( Sum(fairness_constraint_variables_2, fairness_coefficient_2) >= (-fairness_bound_2) ) ;

}

	double compute_fairness(int sp_plus, int sp_minus, int su_plus, int su_minus){
		double fairness_FNR = ((double)((nb_protected_positive - sp_plus) /(double) (nb_protected_positive)  )  - 	 ((double)(nb_unprotected_positive - su_plus)/(double) (nb_unprotected_positive)  ));
		double fairness_FPR = ((double)sp_minus/(double) (nb_protected_negative)  )  - 	 ((double)su_minus/(double) (nb_unprotected_negative)  );

		return max(fabs(fairness_FNR), fabs(fairness_FPR)); // only need this to CHECK the correctness of the solution
	}

};

// Data structure used to return solver results
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

//Global variables and types for memoisation
std::hash<std::string> myhasher_mistral;

struct key1_mistral_hash : public std::unary_function<mykey1_t, std::size_t>
{
   std::size_t operator()(const mykey1_t& k) const
   {
      std::string s = std::to_string(std::get<0>(k))
                     + std::to_string(std::get<1>(k)) + std::to_string(std::get<2>(k))
                     + std::to_string(std::get<3>(k)) + std::to_string(std::get<4>(k))
                     + std::to_string(std::get<5>(k)) + std::to_string(std::get<6>(k))
                     + std::to_string(std::get<7>(k)) + std::to_string(std::get<8>(k))
                     + std::to_string(std::get<9>(k));
      return myhasher_mistral(s);
   }
};

struct key2_mistral_hash : public std::unary_function<mykey2_t, std::size_t>
{
   std::size_t operator()(const mykey2_t& k) const
   {
      std::string s = std::to_string(std::get<0>(k))
                     + std::to_string(std::get<1>(k)) + std::to_string(std::get<2>(k))
                     + std::to_string(std::get<3>(k)) + std::to_string(std::get<4>(k))
                     + std::to_string(std::get<5>(k)) + std::to_string(std::get<6>(k))
                     + std::to_string(std::get<7>(k));
      return myhasher_mistral(s);
   }
};

typedef std::tuple<int, int, Mistral::Outcome> memo2_mistral_data_t; // tuple for advanced memoisation (stores (L, U, res))
typedef std::unordered_map<const mykey1_t,Mistral::Outcome,key1_mistral_hash,key1_equal> memo1_mistral_t;
typedef std::unordered_map<const mykey2_t,std::vector<memo2_mistral_data_t>,key2_mistral_hash,key2_equal> memo2_mistral_t;

memo1_mistral_t memo1_mistral;
memo2_mistral_t memo2_mistral;
int memoisation_mistral_g;
int memoCheck_mistral = 0;
int memoRead_mistral = 0;
int memoRead_mistral2 = 0;

/*
Creates the memo object according to the chosen policy
*/
int mistral_init_memo(int memoisation){
    memoisation_mistral_g = memoisation;
    memoCheck_mistral = 0;
    memoRead_mistral = 0;
    memoRead_mistral2 = 0;
    if(memoisation_mistral_g == 1){
      memo1_mistral_t m;
      memo1_mistral = m; 
   } else if(memoisation_mistral_g == 2){
      memo2_mistral_t m;
      memo2_mistral = m; 
   }
}

/*
Ends session and destroys the model.
*/
int mistral_clear_memo(int verbose){
   if(memoisation_mistral_g == 1){
		if(verbose > 0){
      		std::cout << "Memo avoided " << 100*(double)memoRead_mistral/(double)memoCheck_mistral << "% of solver calls." << std::endl;
		}
      	memo1_mistral.clear();
		if(verbose > 0){
      		std::cout << "Successfully cleared Mistral memo (1)." << std::endl;
		}
   } else if(memoisation_mistral_g == 2){
      if(verbose > 0){
		  std::cout << "Memo avoided " << 100*(double)(memoRead_mistral+memoRead_mistral2)/(double)memoCheck_mistral << "% of solver calls (" << 100*(double)memoRead_mistral2/(double)memoCheck_mistral << "% saved by advanced memoisation)." << std::endl;
	  }
	  memo2_mistral.clear();
      if(verbose > 0){
		  std::cout << "Successfully cleared Mistral memo (2)." << std::endl;
	  }
   }
   return 0;
}

/*
Runs the filtering algorithm corresponding to metric, with the provided parameters, and returns the measured running time (in nanoseconds)
The solver in ran with the provided configuration
*/
Mistral::Outcome runFiltering(
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
			){
        // We first check the memo...
        if(memoisation_mistral_g == 1){
            memo1_mistral_t::const_iterator check_element = memo1_mistral.find(std::make_tuple(TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu, L, U));
            memoCheck_mistral++;
            if(check_element != memo1_mistral.end()){
               memoRead_mistral++;
               return check_element->second;
            }
            //else
            //   std::cout << "Element not found in memo :(" << std::endl;
         } else if(memoisation_mistral_g == 2){
            //std::cout << "Memoisation 2 not implemented now." << std::endl;
            //exit(1);
            memo2_mistral_t::const_iterator check_element = memo2_mistral.find(std::make_tuple(TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu));
            memoCheck_mistral++;
            if(check_element != memo2_mistral.end()){ 
               int L_saved ;
               int U_saved ;
               Mistral::Outcome result_saved;
               for(int i = 0; i < (check_element->second).size(); i++){
                  L_saved = std::get<0>((check_element->second)[i]);
                  U_saved = std::get<1>((check_element->second)[i]);
                  result_saved = std::get<2>((check_element->second)[i]);
                  if(L_saved == L && U_saved == U){ // If exact same problem
                     memoRead_mistral++;
                     return result_saved;
                  } else if(result_saved == UNSAT && L_saved <= L && U_saved >= U){ // If similar problem with different accuracy bounds
                     memoRead_mistral2++;
                     return result_saved;
                  } // else refine objective bounds ? TODO later!?
               }
            }
         }
        // Nothing found in memo: run the solver!
		Mistral::Outcome res;
		switch(metric)
		{
			case 1:
			{
				FilteringStatisticalParity check_bounds(nb_sp_plus,nb_sp_minus, nb_su_plus, nb_su_minus, L,U , tolerence, TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu);
				if(timeout > 0){
					check_bounds.set_timeout(timeout);
				}
				//std::cout << check_bounds << std::endl;
				//std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
				check_bounds.run(0, solverConfig);
				//check_bounds.print_and_verify_solution();
				//std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
				res = check_bounds.get_outcome();
				//res.solvingTime =  check_bounds.get_cpu_time();//std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
				break;
			}
			case 2:
			{
				/*FilteringPredictiveParity check_bounds(nb_sp_plus,nb_sp_minus, nb_su_plus, nb_su_minus, L,U , tolerence, TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu);
				if(timeout > 0){
					check_bounds.set_timeout(timeout);
				}
				//std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
				check_bounds.run(0, solverConfig);
				//std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
				res.result = check_bounds.get_outcome();
				res.solvingTime = check_bounds.get_cpu_time();//std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();*/
				std::cout <<  "Filtering not supported for PP " << std::endl;
				exit(1);
			}
			case 3:
			{
				FilteringPredictiveEquality check_bounds(nb_sp_plus,nb_sp_minus, nb_su_plus, nb_su_minus, L,U , tolerence, TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu);
				if(timeout > 0){
					check_bounds.set_timeout(timeout);
				}
				//std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
				check_bounds.run(0, solverConfig);
				//std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
				res = check_bounds.get_outcome();
				//res.solvingTime = check_bounds.get_cpu_time();//std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
				break;
			}
			case 4:
			{
				FilteringEqualOpportunity check_bounds(nb_sp_plus,nb_sp_minus, nb_su_plus, nb_su_minus, L,U , tolerence, TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu);
				if(timeout > 0){
					check_bounds.set_timeout(timeout);
				}
				//std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
				check_bounds.run(0, solverConfig);
				//std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
				res = check_bounds.get_outcome();
				//res.solvingTime = check_bounds.get_cpu_time();//std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
				break;
			}
			case 5:
			{
				FilteringEqualizedOdds check_bounds(nb_sp_plus,nb_sp_minus, nb_su_plus, nb_su_minus, L,U , tolerence, TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu);
				if(timeout > 0){
					check_bounds.set_timeout(timeout);
				}
				//std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
				check_bounds.run(0, solverConfig);
				//std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
				res = check_bounds.get_outcome();
				//res.solvingTime = check_bounds.get_cpu_time();//std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
				break;
			}
			case 6:
			{
				/*FilteringConditionalUseAccuracyEquality check_bounds(nb_sp_plus,nb_sp_minus, nb_su_plus, nb_su_minus, L,U , tolerence, TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu);
				if(timeout > 0){
					check_bounds.set_timeout(timeout);
				}
				//std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
				check_bounds.run(0, solverConfig);
				//std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
				res.result = check_bounds.get_outcome();
				res.solvingTime = check_bounds.get_cpu_time();//std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();*/
				std::cout <<  "Filtering not supported for CUAE " << std::endl;
				exit(1);
			}
			default:
			{
				std::cout <<  "Error : metric should be an integer in [1,6] " << std::endl;
				exit(1);
			}
		}
        if(memoisation_mistral_g == 1){
            memo1_mistral[std::make_tuple(TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu, L, U)] = res;
        } else if(memoisation_mistral_g == 2){
            memo2_mistral_t::const_iterator check_element = memo2_mistral.find(std::make_tuple(TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu));
            if(check_element != memo2_mistral.end()){ 
                memo2_mistral[std::make_tuple(TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu)].push_back(std::make_tuple(L, U, res));
            } else {
                std::vector<memo2_mistral_data_t> new_vector;
                new_vector.push_back(std::make_tuple(L, U, res));
                memo2_mistral[std::make_tuple(TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu)] = new_vector;
            }
        }
		return res;
}