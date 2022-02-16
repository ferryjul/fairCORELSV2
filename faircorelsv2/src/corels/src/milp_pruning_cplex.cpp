// -------------------------------------------------------------- -*- C++ -*-
// File: blend.cpp
// Version 12.9.0  
// --------------------------------------------------------------------------
// Licensed Materials - Property of IBM
// 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
// Copyright IBM Corporation 2000, 2019. All Rights Reserved.
//
// US Government Users Restricted Rights - Use, duplication or
// disclosure restricted by GSA ADP Schedule Contract with
// IBM Corp.
// --------------------------------------------------------------------------
//
// blend.cpp -- A blending problem

/* ------------------------------------------------------------

Problem Description
-------------------

Goal is to blend four sources to produce an alloy: pure metal, raw
materials, scrap, and ingots.

Each source has a cost.
Each source is made up of elements in different proportions.
Alloy has minimum and maximum proportion of each element.

Minimize cost of producing a requested quantity of alloy.

------------------------------------------------------------ */

#include "pruning_interfaces.hh"
//#include "memoisation_utils.hh"

#ifndef CPLEX_SUPPORT // Compile without CPLEX

int build_model(int fairnessMetric,
         int nb_sp_plus,
			int nb_sp_minus,
			int nb_su_plus,
			int nb_su_minus,
			int L,
			int U,
			double fairness_tolerence,
         int memoisation,
			bool optimization){
            std::cout << "FairCORELS was compiled without CPLEX support. Please consult installation notes to re-compile FairCORELS with CPLEX." << std::endl;
            exit(1);
         }

int prune_opt(int L,
			int U,
         int TPp,
			int FPp,
			int TNp,
			int FNp,
			int TPu,
			int FPu,
			int TNu,
			int FNu,
         bool check_response){
            std::cout << "FairCORELS was compiled without CPLEX support. Please consult installation notes to re-compile FairCORELS with CPLEX." << std::endl;
            exit(1);
            return -1;
         }

int end_cplex(int verbose){
   std::cout << "FairCORELS was compiled without CPLEX support. Please consult installation notes to re-compile FairCORELS with CPLEX." << std::endl;
   exit(1);
   return -1;
}

#else // Compile with CPLEX

#include <ilcplex/ilocplex.h>

ILOSTLBEGIN

std::hash<std::string> myhasher;

struct key1_hash : public std::unary_function<mykey1_t, std::size_t>
{
   std::size_t operator()(const mykey1_t& k) const
   {
      std::string s = std::to_string(std::get<0>(k))
                     + std::to_string(std::get<1>(k)) + std::to_string(std::get<2>(k))
                     + std::to_string(std::get<3>(k)) + std::to_string(std::get<4>(k))
                     + std::to_string(std::get<5>(k)) + std::to_string(std::get<6>(k))
                     + std::to_string(std::get<7>(k)) + std::to_string(std::get<8>(k))
                     + std::to_string(std::get<9>(k));
      return myhasher(s);
   }
};

struct key2_hash : public std::unary_function<mykey2_t, std::size_t>
{
   std::size_t operator()(const mykey2_t& k) const
   {
      std::string s = std::to_string(std::get<0>(k))
                     + std::to_string(std::get<1>(k)) + std::to_string(std::get<2>(k))
                     + std::to_string(std::get<3>(k)) + std::to_string(std::get<4>(k))
                     + std::to_string(std::get<5>(k)) + std::to_string(std::get<6>(k))
                     + std::to_string(std::get<7>(k));
      return myhasher(s);
   }
};

typedef std::tuple<int, int, int> memo2_data_t; // tuple for advanced memoisation (stores (L, U, res))
typedef std::unordered_map<const mykey1_t,int,key1_hash,key1_equal> memo1_t;
typedef std::unordered_map<const mykey2_t,std::vector<memo2_data_t>,key2_hash,key2_equal> memo2_t;

bool optimization_g;
//Global variables for memoisation
memo1_t memo1;
memo2_t memo2;
int memoisation_g;
int memoCheck = 0;
int memoRead = 0;
int memoRead2 = 0;

// Global variables for CPLEX Structures
IloEnv env;
IloModel model;//(env); 
IloNumVarArray var;// (env); 
IloRangeArray con;//(env); 
IloCplex cplex_g;
IloObjective obj_g;

// Global variables for problem cardinalities (needed during model update!)
int nb_sp_plus_g;
int nb_sp_minus_g;
int nb_su_plus_g;
int nb_su_minus_g;

/*
Can be used to decide whether the optimisation (opt=True) 
or decision/satisfiability (opt=False) problem should be solved.
*/
void set_optimization(bool opt){
   if(opt){
      if(!optimization_g){ // from SAT to OPT
         optimization_g=true;
         IloNumArray new_coeffs;
         new_coeffs.add(1);
         new_coeffs.add(-1);
         new_coeffs.add(1);
         new_coeffs.add(-1);
         obj_g.setLinearCoefs(var, new_coeffs);//0));//)); //1.00 * x[0] -1.00 * x[1] + 1.00 * x[2] - 1.00 * x[3]
      }
   } else{
      if(optimization_g){ // from OPT to SAT
         optimization_g=false;
         IloNumArray new_coeffs;
         new_coeffs.add(0);
         new_coeffs.add(0);
         new_coeffs.add(0);
         new_coeffs.add(0);
         obj_g.setLinearCoefs(var, new_coeffs);
      }
   }
}

/*
Initializes CPLEX env., creates the model
bool optimization: if true then solves the OPT problem to order the priority queue, if false only checks feasibility
*/
int build_model(int fairnessMetric,
         int nb_sp_plus,
			int nb_sp_minus,
			int nb_su_plus,
			int nb_su_minus,
			int L,
			int U,
			double fairness_tolerence,
         int memoisation,
			bool optimization)

{
   IloEnv env_loc;
   IloModel model_loc(env_loc); 
   IloNumVarArray var_loc (env_loc); 
   IloRangeArray con_loc(env_loc);
   env = env_loc;
   model = model_loc;
   var = var_loc;
   con = con_loc;

   nb_sp_plus_g = nb_sp_plus;
   nb_sp_minus_g = nb_sp_minus;
   nb_su_plus_g = nb_su_plus;
   nb_su_minus_g = nb_su_minus;

   memoisation_g = memoisation;
   memoCheck = 0;
   memoRead = 0;
   memoRead2 = 0;
   
   optimization_g = optimization;
   try {
      // sp_plus is var[0]
      var.add(IloNumVar(env, 0, nb_sp_plus , ILOINT)); 
      // sp_minus is var[1]
      var.add(IloNumVar(env, 0, nb_sp_minus  , ILOINT)); 
      // su_plus is var[2]
      var.add(IloNumVar(env, 0, nb_su_plus  , ILOINT)); 
      // su_minus is var[3]
      var.add(IloNumVar(env, 0, nb_su_minus  , ILOINT)); 
      
      // Add objective 
      if(optimization){
         obj_g = IloMaximize(env, var[0] - var[1] + var[2] - var[3]); // <- the good one
         //0));//)); //1.00 * x[0] -1.00 * x[1] + 1.00 * x[2] - 1.00 * x[3]
      } else {
         obj_g = IloMaximize(env, 0);
      }
      model.add(obj_g);
   
      // Add constraints 
   
      // Accuracy constraint
      int Up_c = U - nb_sp_minus - nb_su_minus;
      int Low_c = L - nb_sp_minus - nb_su_minus;

      con.add( var[0] - var[1] + var[2] - var[3] <= Up_c); 
      con.add( var[0] - var[1] + var[2] - var[3] >= Low_c); 
      

      // Fairness constraint
      switch(fairnessMetric){
         case 1: {
            int tot_p = (nb_sp_plus + nb_sp_minus);
            int tot_u = (nb_su_plus + nb_su_minus);
            int const_fairness_1 = fairness_tolerence * tot_p * tot_u;
            con.add(tot_u * var[0] + tot_u * var[1] - tot_p * var[2] - tot_p * var[3] <= const_fairness_1);
            con.add(- const_fairness_1 <= tot_u * var[0] + tot_u * var[1] - tot_p * var[2] - tot_p * var[3]);
            break;
         }

         case 3: {
            int const_fairness_3 = fairness_tolerence * nb_su_minus * nb_sp_minus;
            con.add(nb_su_minus * var[1] - nb_sp_minus * var[3] <= const_fairness_3);
            con.add(- const_fairness_3 <= nb_su_minus * var[1] - nb_sp_minus * var[3]);
            break;
         }

         case 4: {
            int const_fairness_4 = fairness_tolerence * nb_su_plus * nb_sp_plus;
            con.add(nb_su_plus * var[0] - nb_sp_plus * var[2] <= const_fairness_4);
            con.add(- const_fairness_4 <= nb_su_plus * var[0] - nb_sp_plus * var[2]);
            break;
         }

         case 5: {
            int const_fairness_3 = fairness_tolerence * nb_su_minus * nb_sp_minus;
            con.add(nb_su_minus * var[1] - nb_sp_minus * var[3] <= const_fairness_3);
            con.add(- const_fairness_3 <= nb_su_minus * var[1] - nb_sp_minus * var[3]);
            int const_fairness_4 = fairness_tolerence * nb_su_plus * nb_sp_plus;
            con.add(nb_su_plus * var[0] - nb_sp_plus * var[2] <= const_fairness_4);
            con.add(- const_fairness_4 <= nb_su_plus * var[0] - nb_sp_plus * var[2]);
            break;
         }

         default: {
            std::cout << "Metric " << fairnessMetric << " unknown. Exiting.";
            exit(1);
         }
      }

      model.add(con); 

      // Optimize
      IloCplex cplex(model);
      cplex.setParam(IloCplex::Param::Threads, 1);
      //std::cout << cplex.getParam(IloCplex::Param::Threads) << std::endl;
      if(!optimization){
         cplex.setParam(IloCplex::Param::MIP::Limits::Solutions, 1); // to only check feasibility
      }
      cplex.setOut(env.getNullStream());
      cplex.setWarning(env.getNullStream());
      //cplex.solve();
      cplex_g = cplex;

      //std::cout << "CPLEX model created." << std::endl;
   }
   catch (IloException& ex) {
      cerr << "Error: " << ex << endl;
   }
   catch (...) {
      cerr << "Error" << endl;
   }

   if(memoisation_g == 1){
      memo1_t m;
      memo1 = m; 
   } else if(memoisation_g == 2){
      memo2_t m;
      memo2 = m; 
   }
   return 0;
}

/*
Updates variables' domains and accuracy constraint bounds.
Then solves the problem again, and returns the result.
*/
int prune_opt(int L,
			int U,
         int TPp,
			int FPp,
			int TNp,
			int FNp,
			int TPu,
			int FPu,
			int TNu,
			int FNu,
         bool check_response){
         
         if(memoisation_g == 1){
            memo1_t::const_iterator check_element = memo1.find(std::make_tuple(TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu, L, U));
            memoCheck++;
            if(check_element != memo1.end()){
               memoRead++;
               return check_element->second;
            }
            //else
            //   std::cout << "Element not found in memo :(" << std::endl;
         } else if(memoisation_g == 2){
            //std::cout << "Memoisation 2 not implemented now." << std::endl;
            //exit(1);
            memo2_t::const_iterator check_element = memo2.find(std::make_tuple(TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu));
            memoCheck++;
            if(check_element != memo2.end()){ 
               int L_saved ;
               int U_saved ;
               int result_saved;
               for(int i = 0; i < (check_element->second).size(); i++){
                  L_saved = std::get<0>((check_element->second)[i]);
                  U_saved = std::get<1>((check_element->second)[i]);
                  result_saved = std::get<2>((check_element->second)[i]);
                  if(L_saved == L && U_saved == U){ // If exact same problem
                     memoRead++;
                     return result_saved;
                  } else if(result_saved == -1 && L_saved <= L && U_saved >= U){ // If similar problem with different accuracy bounds
                     memoRead2++;
                     return result_saved;
                  } // else refine objective bounds ? TODO later!?
               }
            }
         }
         try{
                  
            /*  Recall of variables' declarations:
            // sp_plus is var[0]
            var.add(IloNumVar(env, TPp, nb_sp_plus - FNp, ILOINT)); 
            // sp_minus is var[1]
            var.add(IloNumVar(env, FPp, nb_sp_minus - TNp , ILOINT)); 
            // su_plus is var[2]
            var.add(IloNumVar(env, TPu, nb_su_plus - FNu , ILOINT)); 
            // su_minus is var[3]
            var.add(IloNumVar(env, FPu, nb_su_minus - TNu , ILOINT)); */

            // Update the model (variables' domains)
            if(var[0].getLB() != TPp){
               var[0].setLB(TPp); 
            }
            if(var[1].getLB() != FPp){
               var[1].setLB(FPp); 
            }
            if(var[2].getLB() != TPu){
               var[2].setLB(TPu); 
            }
            if(var[3].getLB() != FPu){
               var[3].setLB(FPu); 
            }

            if(var[0].getUB() != nb_sp_plus_g - FNp){
               var[0].setUB(nb_sp_plus_g - FNp); 
            }
            if(var[1].getUB() != nb_sp_minus_g - TNp){
               var[1].setUB(nb_sp_minus_g - TNp); 
            }
            if(var[2].getUB() != nb_su_plus_g - FNu){
               var[2].setUB(nb_su_plus_g - FNu); 
            }
            if(var[3].getUB() != nb_su_minus_g - TNu){
               var[3].setUB(nb_su_minus_g - TNu); 
            }

            // Update the model (Accuracy constraint)
            int Up_c = U - nb_sp_minus_g - nb_su_minus_g;
            int Low_c = L - nb_sp_minus_g - nb_su_minus_g;

            con[0].setUB(Up_c); 
            con[1].setLB(Low_c);

            // Solve it
            cplex_g.solve();
               
            // Display solutions
            if (cplex_g.getStatus() == IloAlgorithm::Infeasible){
               if(check_response)
                  env.out() << "No Solution" << endl;
               //cplex_g.clearModel();
               //cplex_g.end();
               if(memoisation_g == 1){
                  memo1[std::make_tuple(TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu, L, U)] = -1;
               } else if(memoisation_g == 2){
                  memo2_t::const_iterator check_element = memo2.find(std::make_tuple(TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu));
                  if(check_element != memo2.end()){ 
                     memo2[std::make_tuple(TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu)].push_back(std::make_tuple(L, U, -1));
                  } else {
                     std::vector<memo2_data_t> new_vector;
                     new_vector.push_back(std::make_tuple(L, U, -1));
                     memo2[std::make_tuple(TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu)] = new_vector;
                  }
               }
               return -1;
            } else {
               if(check_response){
                  env.out() << "Solution status: " << cplex_g.getStatus() << endl;
               }
               // Print results
               if(check_response){
                  int nb_well_classified = cplex_g.getValue(var[0]) - cplex_g.getValue(var[1]) + cplex_g.getValue(var[2]) - cplex_g.getValue(var[3]);
                  if(cplex_g.getObjValue() != nb_well_classified)
                     std::cout << "cplex_g.getValue(var[0]) - cplex_g.getValue(var[1]) + cplex_g.getValue(var[2]) - cplex_g.getValue(var[3]) = " << nb_well_classified << ", cplex_g.getObjValue() = " << cplex_g.getObjValue() << std::endl;
               }

               int ret = cplex_g.getObjValue() + nb_sp_minus_g + nb_su_minus_g;
               //cplex_g.clearModel();
               //cplex_g.end();
               if(memoisation_g == 1){
                  memo1[std::make_tuple(TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu, L, U)] = ret;
               } else if(memoisation_g == 2){
                  memo2_t::const_iterator check_element = memo2.find(std::make_tuple(TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu));
                  if(check_element != memo2.end()){ 
                     memo2[std::make_tuple(TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu)].push_back(std::make_tuple(L, U, ret));
                  } else {
                     std::vector<memo2_data_t> new_vector;
                     new_vector.push_back(std::make_tuple(L, U, ret));
                     memo2[std::make_tuple(TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu)] = new_vector;
                  }
               }
               return ret;
            }
         }
         catch (IloException& ex) {
            cerr << "Error: " << ex << endl;
            exit(0);
         }
         catch (...) {
            cerr << "Error" << endl;
            exit(0);
         }
      }

/*
Ends session and destroys the model.
*/
int end_cplex(int verbose){
   cplex_g.clearModel();
   cplex_g.end();
   env.end();
   if(memoisation_g == 1){
      if(verbose > 0){
         std::cout << "Memo avoided " << 100*(double)memoRead/(double)memoCheck << "% of solver calls." << std::endl;
      }
      memo1.clear();
      if(verbose > 0){
         std::cout << "Successfully cleared CPLEX memo (1)." << std::endl;
      }
   } else if(memoisation_g == 2){
      if(verbose > 0){
         std::cout << "Memo avoided " << 100*(double)(memoRead+memoRead2)/(double)memoCheck << "% of solver calls (" << 100*(double)memoRead2/(double)memoCheck << "% saved by advanced memoisation)." << std::endl;
      }
      memo2.clear();
      if(verbose > 0){
         std::cout << "Successfully cleared CPLEX memo (2)." << std::endl;
      }
   }
   if(verbose > 0){
      std::cout << "Successfully ended the CPLEX session." << std::endl;
   }
   return 0;
}


#endif 