import sys
from typing import Tuple, List, TextIO, Callable, cast, Union, Dict
import pandas as pd
import numpy as np
import string
from pprint import pprint
from line_profiler import LineProfiler

# REQUIRED PYTHON PACKAGES:
# mypy
# pandas
# pandas-stubs


# Mypy doesn't handle complicated recursive definitions very well, so we will use a simple recursive definition.
# Keep in mind that, in practice, every non-leaf node of the decision tree must have a string key called "feature" to represent
# which feature to test next (a painting's size, medium, or color, for example), AND a string key called "bias" to represent
# the majority label of data at that juncture. If there isn't an applicable value for the feature to split on (ie, the painting 
# that we are evaluating is yellow, but we have never seen a yellow painting before), then we classify that datum as whatever the "bias"
# value is.
DecisionTree = Union[Dict[str, 'DecisionTree'], str]
class DecisionTreeTrainer:
    def __init__(self, label_col: str, feature_cols: List[str]):
        self.label_col = label_col
        self.feature_cols = feature_cols


    # private
    def entropy(self, series: pd.Series) -> float:
        # do a vectorized numpy calculation for speed
        probs = series.value_counts(normalize=True)
        return -np.sum(probs * np.log2(probs))
    
    # private
    def info_gain(self, df: pd.DataFrame, split_col: str) -> float:
        def agg_ent_size(group):
            return pd.Series({'size': len(group), 'entropy': self.entropy(group)})
            
        ents = (df
                .groupby(split_col)[[self.label_col]]
                .apply(agg_ent_size))
        ents['weighted_entropy'] = ents['entropy'] * ents['size'] / df.shape[0]
        # apply the entropy calculation
        return self.entropy(df[self.label_col]) - np.sum(ents['weighted_entropy'])



    ###############################                 PRIVATE METHOD              ##################################
    # Induce a decision tree from training data in a pandas dataframe. If the labels are evenly split, decide in favor
    # of the provided bias.
    def induce_tree(self, training: pd.DataFrame, parent_bias: str) -> DecisionTree:
        # if there is only one type of value remaining in our induced subgroup, then we have hit our basecase
        counts = training[self.label_col].value_counts()
        if counts.shape[0] == 1:
            return cast(str, training[[self.label_col]].iloc[0][self.label_col])
        
        # figure out how our results should be biased in the case that our data is completely evenly labled.
        # ie if each label has the same number of data points
        curr_bias: str = parent_bias if counts.duplicated(keep=False).all() else str(counts.idxmax())
        # if all of the data points are identical, then we have our other basecase. Return the bias:
        if training[self.feature_cols].duplicated(keep=False).all():
            return curr_bias
        
        # make a tuple of tuples where each subtuple contains the information gain and the column to split on
        info_gains: Tuple[Tuple[float, str],...] = tuple((self.info_gain(training, feature_col), feature_col) for feature_col in self.feature_cols)
        #  if splitting on two columns would result in an identical information gain, then split on the leftmost one
        split_idx: int = info_gains.index(max(info_gains, key=lambda x: x[0]))
        split_feature: str = info_gains[split_idx][-1]
        tree: Dict[str, DecisionTree] = {'feature': split_feature, 'bias': curr_bias}
        # make a sub-decision-tree for each of the potential values in the split_feature column.
        # ie evaluate all of the potential answers to the 'split_feature' question
        for field in training[split_feature].unique():
            # recurse on the subgroup induced by the chosen feature with respect to the current field
            # ex. if we are classifying paintings, the chosen feature may be "medium", the current field may be "acrylic", 
            # and the induced subgroup might be every data point with "acrylic" as the medium
            induced_subset = training.loc[training[split_feature] == field]
            tree[field] = self.induce_tree(induced_subset, curr_bias)

        return tree
        
        

    # private
    def classify_datum(self, datum: pd.Series, root: DecisionTree) -> str:
        curr = root
        while not isinstance(curr, str):
            feat = datum[cast(str, curr['feature'])]
            if feat in curr:
                curr = curr[feat]
            else:
                return cast(str, curr['bias'])
        return curr
    
    # private
    def test_tree(self, testing: pd.DataFrame, root: DecisionTree) -> float:
        # apply our classify_datum function across each row
        classifications = testing.apply(lambda row: self.classify_datum(row, root), axis=1)
        # get all the rows where our classification was correct:
        matches = testing.loc[testing[self.label_col] == classifications]
        # return the percenttage that we got right
        return matches.shape[0]/testing.shape[0]
    
    def count_nodes_in_tree(self, tree: DecisionTree) -> int:
        # base case. we are at a leaf node:
        if isinstance(tree, str):
            return 1
        # recursive case. We have to count this node and all the subnodes
        count = 1
        for key,sub_tree in tree.items():
             if key not in ('feature', 'bias'):
                count += self.count_nodes_in_tree(sub_tree)

        return count

    # private
    def prune_tree_greedily(self, testing: pd.DataFrame, sub_tree: DecisionTree, whole_tree:  DecisionTree) -> Tuple[float, DecisionTree]:
        best_pruned_tree = sub_tree
        best_pruned_tree_score = 0.0
        # if we have found a leaf, return a really low score so that it doesn't get pruned. In reality, this should
        # never happen because of a check in the while loop before we recurse. It's mostly to appease the typechecker :(
        if isinstance(sub_tree, str) or isinstance(whole_tree, str):
            return best_pruned_tree_score, best_pruned_tree
        
        for key in sub_tree.keys():
            # turn a sub-tree of this sub-tree into a leaf-node with the majority label at this sub-tree
            # ie, we are pruning a sub-tree of this sub-tree into a leaf
            temp = sub_tree[key]
            # if the sub-sub-tree is a leaf or metadata, then continue
            if key in ('feature', 'bias') or isinstance(temp, str):
                continue

            sub_tree[key] = cast(Dict[str, DecisionTree], sub_tree[key])['bias']
            # if this pruning operation was the best that we've seen so far, then make it our new best
            tree_score = self.test_tree(testing, sub_tree)
            if tree_score > best_pruned_tree_score or (tree_score == best_pruned_tree_score and 
                                                       self.count_nodes_in_tree(whole_tree) < self.count_nodes_in_tree(best_pruned_tree)):
                best_pruned_tree = dict(whole_tree)
                best_pruned_tree_score = tree_score
            # undo our pruning
            sub_tree[key] = temp
            # get the best pruning from a branch further down the tree from where we are now:
            best_pruned_child_tree_score, best_pruned_child_tree = self.prune_tree_greedily(testing, sub_tree[key], whole_tree)
            # if it's better than any pruning we've found so far, set it as our best found pruning
            if best_pruned_child_tree_score > best_pruned_tree_score or (best_pruned_child_tree_score == best_pruned_tree_score and 
                                                       self.count_nodes_in_tree(best_pruned_child_tree) < self.count_nodes_in_tree(best_pruned_tree)):
                best_pruned_tree = best_pruned_child_tree
                best_pruned_tree_score = best_pruned_child_tree_score

        return best_pruned_tree_score, best_pruned_tree

    # private
    def tune_tree(self, data: pd.DataFrame, training_tuning_partitioner: Callable[[pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]]) ->Tuple[float, DecisionTree]:
        training, tuning = training_tuning_partitioner(data)
        # calculate the bias of the training set and make a decision tree:
        bias = str(training[self.label_col].value_counts().idxmax())
        tree = self.induce_tree(training, bias)
        # test the tree on the tuning set
        score = self.test_tree(tuning, tree)
        # prune the tree and test it on the tuning set
        pruned_score, pruned_tree = self.prune_tree_greedily(tuning, tree, tree)
        # keep pruning the tree until the best pruning doesn't produce a better score
        while pruned_score > score:
            score = pruned_score
            tree = pruned_tree
            pruned_score, pruned_tree = self.prune_tree_greedily(tuning, tree, tree)

        return score, tree

    # public
    def leave_one_out_train(self, data: pd.DataFrame, training_tuning_partitioner: Callable[[pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]]) -> Tuple[DecisionTree, float]:
        _, tree = self.tune_tree(data, training_tuning_partitioner)
        score = 0.0
        for index, row in data.iterrows():
            # make a tree with every row except for "index"
            _, minus_one_tree = self.tune_tree(data.drop(index), training_tuning_partitioner)
            # test our tree on the index we had dropped
            score += self.test_tree(data.loc[[index]], minus_one_tree)
            print(index)
        # return an average score across all trees and tests:
        return tree, score/data.shape[0]
            

        

def makeDecisionTree(tsvfile: TextIO):
    df = pd.read_csv(tsvfile, sep='\t', header=None, names = ['Rep', 'Party', 'Votes'])

    # concatenate df with the Votes column split into new columns for each issue (each character in the string).
    # Note that the regular expression that we split on is a positive lookbehind (checking for '+'s, '-'s, and '.'s) and a positive lookahead 
    # (checking for the same characters). We don't split on the empty string because then the zeroth and last columns would be empty.
    # https://pandas.pydata.org/docs/reference/api/pandas.Series.str.split.html#pandas.Series.str.split

    # I don't do this in two lines because I'm scared of leaving a pointer to a big chunk of memory that I won't use after the concatenation.
    # Maybe my concerns are unjustified?
    df = pd.concat([df, df['Votes'].str.split(r'(?<=\+|\-|\.)(?=\+|\-|\.)', expand=True)], axis=1)
    # rename the numbered columns to letters indicating which issue is being voted on:
    new_cols: List[str] = [string.ascii_lowercase[cast(int, col)] if isinstance(col, int) else col for col in df.columns]
    df.columns = pd.Index(new_cols)
    training_tuning_partitioner =  lambda x: (x.loc[lambda row: row.index % 4 != 0], x[::4])

    features =  list(string.ascii_lowercase[0:10])
    dtt = DecisionTreeTrainer('Party', features)
    #tree = dtt.induce_tree(df, 'Rep')
    # test =dtt.test_tree(df, tree)
    # pprint(tree)
    # print(test)
    profiler = LineProfiler()
    profiler.add_function(dtt.tune_tree)
    profiler.add_function(dtt.induce_tree)
    profiler.add_function(dtt.info_gain)
    profiler.add_function(dtt.entropy)
    profiler.runctx('dtt.tune_tree(df, training_tuning_partitioner)', globals(), locals())
    profiler.print_stats()
    #pruned_score, pruned_tree = dtt.leave_one_out_train(df, training_tuning_partitioner)
    #pprint(pruned_tree)
    #print(pruned_score)

    test_df = pd.DataFrame({
        'Size': ['big', 'medium', 'big', 'small', 'big', 'small', 'medium', 'small', 'medium', 'big'],
        'Color': ['red', 'blue', 'yellow', 'blue', 'red', 'red', 'blue', 'yellow', 'blue', 'red'],
        'Medium': ['oil', 'acrylic', 'acrylic', 'acrylic', 'watercolor', 'watercolor', 'watercolor', 'oil', 'oil', 'oil'],
        'Label': ['good', 'good', 'good', 'good', 'good', 'bad', 'bad', 'bad', 'bad', 'bad']
    })
    test_dtt = DecisionTreeTrainer('Label', ['Size', 'Color', 'Medium'])
    pruned_score, pruned_tree = test_dtt.leave_one_out_train(test_df,training_tuning_partitioner)
    pprint(pruned_tree)
    print(pruned_score)





# check if the user has provided a file name as a command-line argument
if len(sys.argv) != 2:
    print("Usage: python decision-tree-inducer.py file_name.tsv")
    sys.exit(1)

file_name: str = sys.argv[1]

try:
    tsvfile: TextIO = open(file_name, newline='')
    makeDecisionTree(tsvfile)
        
except FileNotFoundError:
    print(f"The file {file_name} was not found.")


