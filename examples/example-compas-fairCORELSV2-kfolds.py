from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from faircorelsv2 import load_from_csv, FairCorelsClassifierV2, ConfusionMatrix, Metric # version 1.0
import argparse
import numpy as np
import csv

parser = argparse.ArgumentParser(description='Analysis of FairCORELS results')
parser.add_argument('--filteringMode', type=int, default=0, help='filteringMode argument for the FairCORESClassifierV2 object.')
args = parser.parse_args()

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

sensitive_attr_column = 0
unsensitive_attr_column = 1

dataset = 'compas' # can be either 'adult', 'compas', or 'german_credit' (all contained in the ./data folder)

print("Let's learn fair rule lists for the " + dataset + " dataset!")

X, y, features, prediction = load_from_csv("./data/%s_rules_full_single.csv" %dataset)  # Load the dataset

epsilon = 0.98 # Fairness constraint
fairnessMetric = 1 # 1 For Statistical Parity, 2 for Predictive Parity...etc
lambdaParam = 1e-3 # The regularization parameter penalizing rule lists length
N_ITER = 1*10**5 # The maximum number of nodes in the prefix tree
# Pruning mode parameters
pruningModeArg =  args.filteringMode # 2 for eager pruning with the Mistral solver
upper_bound_pruning_arg = 1 # computes tight upper bound
pruning_memoisation = 1 # Use memoisation for calls to solver for pruning
map_type_arg = "prefix" # This prefix permutation map is not CORELS' original one. It has been modified to guarantee optimality of the built rule lists.

# We prepare the folds for our 5-folds cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
folds = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    folds.append([X_train, y_train, X_test, y_test])

print("Will perform 5-folds cross-validation.")

def oneFold(foldIndex, X_fold_data): # This part could be multithreaded for better performance
        X_train, y_train, X_test, y_test = X_fold_data

        # Separate protected features to avoid disparate treatment
        # - Training set
        sensVect_train =  X_train[:,sensitive_attr_column]
        unSensVect_train =  X_train[:,unsensitive_attr_column] 
        X_train_unprotected = X_train[:,2:]

        # - Test set
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
                                verbosity=[],
                                epsilon=epsilon, 
                                maj_vect=unSensVect_train, 
                                min_vect=sensVect_train,
                                filteringMode=pruningModeArg,
                                upper_bound_filtering=upper_bound_pruning_arg,
                                fileName="./faircorels_logs/example-compas-logs-fold-%d.txt" %foldIndex
                                )

        # Train it
        clf.fit(X_train_unprotected, y_train, features=features[2:], prediction_name="(income:>50K)")

        # Print the fitted model
        #print(clf.rl_)

        # Evaluate our model's accuracy
        accTraining = clf.score(X_train_unprotected, y_train)
        accTest = clf.score(X_test_unprotected, y_test)
        

        # Evaluate our model's fairness
        train_preds = clf.predict(X_train_unprotected)
        unfTraining = compute_unfairness(sensVect_train, unSensVect_train, y_train, train_preds)

        test_preds = clf.predict(X_test_unprotected)
        unfTest = compute_unfairness(sensVect_test, unSensVect_test, y_test, test_preds)
        length = len(clf.rl_.rules)-1
        objF = ((1-accTraining) + (lambdaParam*length)) # best objective function reached
        exploredBeforeBest = int(clf.nbExplored)
        cacheSizeAtExit = int(clf.nbCache)
        return [foldIndex, accTraining, unfTraining, objF, accTest, unfTest, exploredBeforeBest, cacheSizeAtExit, str(clf.rl_), clf.get_solving_status()]

# Run training/evaluation for all folds using multi-threading
ret = Parallel(n_jobs=-1)(delayed(oneFold)(foldIndex, X_fold_data) for foldIndex, X_fold_data in enumerate(folds))

# Unwrap the results
accuracy = [ret[i][4] for i in range(0,5)]
unfairness = [ret[i][5] for i in range(0,5)]
objective_functions = [ret[i][3] for i in range(0,5)]
accuracyT = [ret[i][1] for i in range(0,5)]
unfairnessT = [ret[i][2] for i in range(0,5)]
print("=========> Training Accuracy (average)= ", np.average(accuracyT))
print("=========> Training Unfairness (average)= ", np.average(unfairnessT))
print("=========> Training Objective function value (average)= ", np.average(objective_functions))
print("=========> Test Accuracy (average)= ", np.average(accuracy))
print("=========> Test Unfairness (average)= ", np.average(unfairness))

#Save results in a csv file
resPerFold = dict()
for aRes in ret:
    resPerFold[aRes[0]] = [aRes[1], aRes[2], aRes[3], aRes[4], aRes[5], aRes[6], aRes[7], aRes[8], aRes[9]]
with open('./results/faircorels_eps%f_metric%d_LB%d.csv' %(epsilon, fairnessMetric, args.filteringMode), mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['Fold#', 'Training accuracy', 'Training Unfairness(%d)' %fairnessMetric, 'Training objective function', 'Test accuracy', 'Test unfairness', '#Nodes explored for best solution', 'Cache size for best solution', 'Solving status', 'Rule list'])#, 'Fairness STD', 'Accuracy STD'])
    for index in range(5):
        csv_writer.writerow([index, resPerFold[index][0], resPerFold[index][1], resPerFold[index][2], resPerFold[index][3], resPerFold[index][4], resPerFold[index][5], resPerFold[index][6], resPerFold[index][8], resPerFold[index][7]])#,fairnessSTD_list_test_tot[i][fairnessMetric-1], accuracySTD_list_test_tot[i]])
    csv_writer.writerow(['Average', np.average(accuracyT), np.average(unfairnessT), np.average(objective_functions), np.average(accuracy), np.average(unfairness), '', '', ''])#,fairnessSTD_list_test_tot[i][fairnessMetric-1], accuracySTD_list_test_tot[i]])

print("Results saved under './results/faircorels_eps%f_metric%d_LB%d.csv'" %(epsilon, fairnessMetric, args.filteringMode))

