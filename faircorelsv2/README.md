Welcome to FairCorelsV2, a Python library for learning fair and interpretable OPTIMAL models using the Certifiably Optimal RulE ListS (CORELS) algorithm! 

FairCORELSV2 uses Python, Numpy, GMP, and a C++ compiler.
It leverages CP/MILP solvers (either Mistral or CPLEX) to efficiently prune the search space, leveraging jointly the objective function and the fairness constraint.
GMP (GNU Multiple Precision library) is not required, but it is *highly recommended*, as it improves performance. 
If it is not installed, fairCORELS will run slower.

The `FairCORELSV2` module includes one classifier method : `FairCorelsClassifierV2`.
The `FairCorelsClassifierV2` class implements the fairCORELSV2 method. Details on the methods used within FairCORELSV2 are provided within our full paper accepted at the CPAIOR 2022 conference ([preprint](https://homepages.laas.fr/jferry/drupal/sites/homepages.laas.fr.jferry/files/u202/CPAIOR2022_paper.pdf)).
In a nutshell, `FairCORELSV2` brings several improvements to the original [`FairCORELS`](https://github.com/ferryjul/fairCORELS) algorithm: a novel pruning approach leveraging Mixed Integer Linear Programming, and a new prefix permutation map designed to provide optimality guarantees when learning fair rule lists.

The currently supported fairness notions are : statistical parity, predictive parity, predictive equality, equal opportunity, equalized odds, and conditional use accuracy equality.

# Detail of the parameters for FairCorelsClassifierV2:

## Constructor arguments :

### Exploration-related arguments

* `c` : float, optional (default=0.01)
    Regularization parameter. Higher values penalize longer rulelists.

* `n_iter` : int, optional (default=1000)
    Maximum number of nodes (rulelists) to search before exiting.

* `map_type` : str, optional (default="prefix")
    The type of prefix map to use. Supported maps are "none" for no map,
    "prefix" for a map that uses rule prefixes for keys, "captured" for
    a map with a prefix's captured vector as keys.
    **NOTE that "prefix" map corresponds to the implementation proposed in our paper at the CPAIOR 2022 conference.
    Indeed, the original CORELS' prefix permutation map failed to guarantee optimality when learning fair rule lists.**

* `policy` : str, optional (default="bfs")
    The search policy for traversing the tree (i.e. the criterion with which
    to order nodes in the queue). Supported criteria are "bfs", for breadth-first
    search; "curious", which attempts to find the most promising node; 
    "lower_bound" which is the objective function evaluated with that rulelist
    minus the default prediction error; "objective" for the objective function
    evaluated at that rulelist; and "dfs" for depth-first search.
    **WARNING if filteringMode is 4 or 5 (order priority queue by the MILP pruning objective) then this parameter is unused!**

* `verbosity` : list, optional (default=["rulelist"])
    The verbosity levels required. A list of strings, it can contain any
    subset of ["rulelist", "rule", "label", "minor", "samples", "progress", "mine", "pruning", "loud"].

    - "rulelist" prints the generated rulelist at the end.
    - "rule" prints a summary of each rule generated.
    - "label" prints a summary of the class labels.
    - "minor" prints a summary of the minority bound.
    - "samples" produces a complete dump of the rules, label, and/or minor data. You must also provide at least one of "rule", "label", or "minor" to specify which data you want to dump, or "loud" for all data. The "samples" option often spits out a lot of output.
    - "progress" prints periodic messages as faircorels runs.
    - "mine" prints debug information while mining rules, including each rule as it is generated.
    - "pruning" prints information about the pruning strategy and statistics. It also prints the total CPU time while ending exploration, and the exploration status.
    - "loud" is the equivalent of ["progress", "label", "rule", "mine", "minor"].

* `ablation` : int, optional (default=0)
    Specifies addition parameters for the bounds used while searching. Accepted
    values are 0 (all bounds), 1 (no antecedent support bound), and 2 (no
    lookahead bound).

* `max_card` : int, optional (default=2)
    Maximum cardinality allowed when mining rules. Can be any value greater than
    or equal to 1. For instance, a value of 2 would only allow rules that combine
    at most two features in their antecedents.

* `min_support` : float, optional (default=0.01)
    The fraction of samples that a rule must capture in order to be used. 1 minus
    this value is also the maximum fraction of samples a rule can capture.
    Can be any value between 0.0 and 0.5.

* `beta` : float, optional (default=0.0)
    The weight the unfairness in the objective function

* `mode` : int optional (default=3)
    Method used for the multi-ojective framework
    1: weigted sum, 2: maximum fairness, 3: epsilon-constraint, 4: maximum accuracy

* `bfs_mode` : int optinal (default=0)
    BFS strategy
    0: original CORELS, 1:FIFO, 2:objective_aware, 3:lower_bound, 4:random

* `random_state` : int optional (default=42)
    Random seed for randomized search

* `fileName` : str optional (default="faircorels_log_out")
    Name of csv file to save exploration progression
    Saves a summary of the performed exploration (each local best solution, the time and #nodes explored to reach it, etc.) along with the final status.

* `kbest` : int optional (default=1)
    Randomly use one of the k best objectives

### Fairness-related arguments

* `fairness` : int optional (default=1)
    The type of fairness metric used. 
    1 : statistical parity, 2 : predictive parity, 3 : predictive equality, 4 : equal opportunity, 5 : equalized odds, 6 : conditional use accuracy equality

* `epsilon` : float optional (default=0.05)
    max acceptable unfairness (fairness violation tolerence)

* `maj_pos` : int optional (default=-1)
    The position of the rule that defined the majority group
    If not specified, all individuals not in minority group are in majority group
    Not used if maj_vect is used

* `min_pos` : int optional (default=2)
    The position of the rule that defined the minority group
    Not used if min_vect is used

* `maj_vect` : int list optional (default=[-1])
    List of integers in {0,1} indicating if instances belong to the majority group
    If not specified, this vector is computed using maj_pos

* ` min_vect` : int list optional (default=[-1])
    List of integers in {0,1} indicating if instances belong to the minority group
    If not specified, this vector is computed using min_pos

### Enhanced pruning arguments

* `filteringMode`: int optional (default=0)
    Defines the mode for the performed pruning (following the method introduced in our CPAIOR 2022 full paper).
    - 0: no filtering
    - 1: SAT filtering, Lazy, using the Mistral solver (C++) (embedded within FairCORELS)
    - 2: SAT filtering, Eager, using the Mistral solver (C++) (embedded within FairCORELS)
    - 3: SAT filtering, both Lazy and Eager, using the Mistral solver (C++) (embedded within FairCORELS)
    - 4: OPT filtering & Guiding (Eager) using CPLEX (C++) (need to compile with CPLEX - see the installation instructions)
        priority queue is ordered by BFS and ties are broken by the MILP objective
        WARNING policy parameter is not used in this case (as priority queue ordering is defined here)
    - 5: OPT filtering & Guiding (Eager) using CPLEX (C++) (need to compile with CPLEX - see the installation instructions)
        priority queue is ordered by best-first search guided by the MILP objective
        WARNING policy parameter is not used in this case (as priority queue ordering is defined here)
    - 6: SAT filtering, Lazy, using the CPLEX solver (C++) (need to compile with CPLEX - see the installation instructions)
    - 7: SAT filtering, Eager, using the CPLEX solver (C++) (need to compile with CPLEX - see the installation instructions)
    - 8: SAT filtering, both Lazy and Eager, using the CPLEX solver (C++) (need to compile with CPLEX - see the installation instructions)

    Note that the CP/MILP filtering is implemented for SP (metric 1), PE (metric 3), EO (metric 4) and EOdds (metric 5).
    Also note that the enhanced pruning has to be used with mode 3 (epsilon-constraint)

* `upper_bound_filtering`: int optional (default=0)
        controls the way upper bound over well classified examples (for advanced pruning) is computed
    - 0 to use simple upper bound computation (use if dataset does not contain inconsistencies)
    - 1 to perform vector operations leveraging inconsistent examples (a little bit slower, but tight - hence may improve the pruning efficiency)
        
    Note that when using the MILP to guide the exploration (if filteringMode is 4 or 5), it is strongly advised to set upper_bound_filtering to 1.

* `pruning_memoisation`: int optional (default=1)
    defines the type of memoisation used to cache the results of the previous calls to the solver
    note that due to symmetries in the prefix tree, this option can save many calls to the solver
    - 0 means no memoisation at all
    - 1 means simple memoisation (saves solver results for the given parameters)
    - 2 means advanced memoisation (saves more information and avoids more calls to solver, but memoisation mechanism has larger overhead)

## Methods :

### .fit(self, X, y, features=[], prediction_name="prediction", max_evals=1000000000, time_limit = None, memory_limit=None):

Method for training the classifier.

* `X` : array-like, shape = [n_samples, n_features]
    The training input samples. All features must be binary, and the matrix is internally converted to dtype=np.uint8.

* `y` : array-line, shape = [n_samples]
    The target values for the training input. Must be binary.
        
* `features` : list, optional(default=[])
    A list of strings of length n_features. Specifies the names of each of the features. If an empty list is provided, the feature names are set to the default of ["feature1", "feature2"... ].

* `prediction_name` : string, optional(default="prediction")
    The name of the feature that is being predicted.

* `max_evals` : int, maximum number of calls to evaluate_children 
    (ie maximum number of nodes explored in the prefix tree)

* `time_limit` : int, maximum number of seconds allowed for the model (default None: no limit)
    building
    Note that this specifies the CPU time and NOT THE WALL-CLOCK TIME

* `memory_limit`: int, maximum memory use (in MB) (default None: no limit)

### .predict(X):
Method for predicting using the trained classifier.

* `X` : array-like, shape = [n_samples, n_features]
    The training input samples. All features must be binary, and the matrix is internally converted to dtype=np.uint8. The features must be the same as those of the data used to train the model.

=> Returns : `p` : array of shape = [n_samples] -> The classifications of the input samples.

### .predict_with_scores(X):
Method for predicting using the trained classifier.

* `X` : array-like, shape = [n_samples, n_features]
    The training input samples. All features must be binary, and the matrix is internally converted to dtype=np.uint8. The features must be the same as those of the data used to train the model.

=> Returns : `p` : array of shape = [[n_samples],[n_samples]].
    The first array contains the classifications of the input samples.
    The second array contains the associated confidence scores.

### .score(X, y):
Method that scores the algorithm on the input samples X with the labels y. Alternatively, score the predictions X against the labels y (where X has been generated by `predict` or something similar).

* `X` : array-like, shape = [n_samples, n_features] OR shape = [n_samples]
    The input samples, or the sample predictions. All features must be binary.
        
* `y` : array-like, shape = [n_samples]
    The input labels. All labels must be binary.

=> Returns : `a` : float
    The accuracy, from 0.0 to 1.0, of the rulelist predictions

### .get_params():
Method to get a list of all the model's parameters.

=> Returns : `params` : dict
Dictionary of all parameters, with the names of the parameters as the keys

### .set_params(params):
Method to set some of the model's parameters.

* `params` :  Set of model parameters. Takes an arbitrary number of keyword parameters, all of which must be valid parameter names (i.e. must be included in those returned by get_params).

### .save(fname):
Method to save the model to a file, using python's pickle module.

* `fname` : string
    File name to store the model in

### .load(fname):
Method to load a model from a file, using python's pickle module.

* `fname` : string
    File name to load the model from
        
### .rl(set_val=None):
Method to return or set the learned rulelist
        
* `set_val` : RuleList, optional
    Rulelist to set the model to

=> Returns : `rl` : obj
    The model's rulelist

### .__str__():
Method to get a string representation of the rule list

=> Returns : `rl` : str
    The rule list

### .__repr__():
Same behavior as the previous one.

### .explain(anEx):
Method to explain a prediction (by providing the matching rule).

* `anEx` : array-like, shape = [n_features] 
    The input sample

=> Returns : list `l` where
    `l[0]` is the instance's prediction
    `l[1]` is the implicant(s) that led to that decision
    (both are strings - user friendly)

### .explain_api(anEx):
Method to explain a prediction (by providing the matching rule) (shorter output).

* `anEx` : array-like, shape = [n_features] 
    The input sample

=> Returns : list `l` where
    `l[0]` is the instance's prediction
    `l[1]` is the implicant(s) that led to that decision
    (both are API-oriented - easy to use by a program)
   
### .explain_long(anEx):
Method to explain a prediction (by providing the matching rule and all the previous unmatched implicants).

* `anEx` : array-like, shape = [n_features] 
    The input sample

=> Returns : list `l` where
    `l[0]` is the instance's prediction
    `l[1]` is the implicant(s) that led to that decision
    (both are strings - user friendly)

### .explain_long_api(anEx):
Method to explain a prediction (by providing the matching rule and all the previous unmatched implicants) (shorter output).

* `anEx` : array-like, shape = [n_features] 
    The input sample

=> Returns : list `l` where
    `l[0]` is the instance's prediction
    `l[1]` is the implicant(s) that led to that decision
    (both are API-oriented - easy to use by a program)
