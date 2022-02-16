from sklearn.model_selection import train_test_split # version 0.24.1
from faircorelsv2 import load_from_csv, FairCorelsClassifierV2, ConfusionMatrix, Metric # version 1.0
import argparse

parser = argparse.ArgumentParser(description='Analysis of FairCORELS results')
parser.add_argument('--filteringMode', type=int, default=0, help='filteringMode argument for the FairCORESClassifierV2 object.')
args = parser.parse_args()

sensitive_attr_column = 0
unsensitive_attr_column = 1

dataset = 'compas' # can be either 'adult', 'compas', or 'german_credit' (all contained in the ./data folder)

print("Let's learn fair rule lists for the " + dataset + " dataset!")

X, y, features, prediction = load_from_csv("./data/%s_rules_full_single.csv" %dataset)  # Load the dataset

epsilon = 0.99 # Fairness constraint
fairnessMetric = 1 # 1 For Statistical Parity, 2 for Predictive Parity...etc
lambdaParam = 1e-3 # The regularization parameter penalizing rule lists length
N_ITER = 1*10**6 # The maximum number of nodes in the prefix tree
# Pruning mode parameters
pruningModeArg =  args.filteringMode # 2 for eager pruning with the Mistral solver
upper_bound_pruning_arg = 1 # computes tight upper bound
pruning_memoisation = 1 # Use memoisation for calls to solver for pruning
map_type_arg = "prefix" # This prefix permutation map is not CORELS' original one. It has been modified to guarantee optimality of the built rule lists.

print("Max #nodes in the prefix trie is %d" %N_ITER)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print("Using ", X_train.shape[0], " examples for training set, ", X_test.shape[0], " examples for test set.")

# Separate protected features to avoid disparate treatment
print("Sensitive attribute is ", features[sensitive_attr_column])
print("Unsensitive attribute is ", features[unsensitive_attr_column])
print("-----------------------------------------------------------")
# Training set
sensVect_train =  X_train[:,sensitive_attr_column]
unSensVect_train =  X_train[:,unsensitive_attr_column] 
X_train_unprotected = X_train[:,2:]

# Test set
sensVect_test =  X_test[:,sensitive_attr_column]
unSensVect_test =  X_test[:,unsensitive_attr_column] 
X_test_unprotected = X_test[:,2:]

# Create the FairCorelsClassifier object
clf = FairCorelsClassifierV2(n_iter=N_ITER, 
                        c=lambdaParam,
                        max_card=1, 
                        min_support = 0.01,
                        policy="bfs", # exploration heuristic
                        bfs_mode=2, # exploration heuristic
                        mode=3, # epsilon-constrained mode
                        map_type=map_type_arg,
                        fairness=fairnessMetric, 
                        verbosity=["pruning"],
                        epsilon=epsilon, 
                        maj_vect=unSensVect_train, 
                        min_vect=sensVect_train,
                        filteringMode=pruningModeArg,
                        upper_bound_filtering=upper_bound_pruning_arg,
                        fileName="./faircorels_logs/example-compas-logs.txt"
                        )

 # Train it
clf.fit(X_train_unprotected, y_train, features=features[2:], prediction_name="(income:>50K)", time_limit=120)

# Displays statistics
exploredBeforeBest = int(clf.nbExplored)
cacheSizeAtExit = int(clf.nbCache)
print("-----------------------------------------------------------")
print("Explored %d nodes before reaching best solution."%exploredBeforeBest)
print("Cache size was %d #nodes while reaching best solution."%cacheSizeAtExit)

# Print the fitted model
print(clf.rl_)

# Evaluate our model's accuracy
accTraining = clf.score(X_train_unprotected, y_train)
accTest = clf.score(X_test_unprotected, y_test)
print("---Accuracy ---")
print("Training set: ", accTraining)
print("Test set:", accTest)

# Evaluate our model's fairness

def compute_unfairness(sensVect, unSensVect, y, y_pred):
    cm = ConfusionMatrix(sensVect, unSensVect, y_pred, y)
    cm_minority, cm_majority = cm.get_matrix()
    fm = Metric(cm_minority, cm_majority)

    if fairnessMetric == 1:
        unf = fm.statistical_parity()
    elif fairnessMetric == 2:
        unf = fm.predictive_parity()
    elif fairnessMetric == 3:
        unf = fm.predictive_equality()
    elif fairnessMetric == 4:
        unf = fm.equal_opportunity()
    elif fairnessMetric == 5:
        unf = fm.equalized_odds()
    elif fairnessMetric == 6:
        unf = fm.conditional_use_accuracy_equality()
    else:
        unf = -1
    
    return unf

train_preds = clf.predict(X_train_unprotected)
unfTraining = compute_unfairness(sensVect_train, unSensVect_train, y_train, train_preds)

test_preds = clf.predict(X_test_unprotected)
unfTest = compute_unfairness(sensVect_test, unSensVect_test, y_test, test_preds)

print("---Unfairness (Metric %d)---" %fairnessMetric)
print("Training set: ", unfTraining)
print("Test set:", unfTest)

