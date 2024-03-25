from typing import Tuple, List, Callable, cast, Union, Dict
import pandas as pd
import numpy as np
import multiprocessing
from copy import deepcopy
from functools import partial

# REQUIRED PYTHON PACKAGES:
# mypy
# numpy
# pandas
# pandas-stubs


# Mypy doesn't handle complicated recursive definitions very well, so we will use a simple recursive definition.
# Keep in mind that, in practice, every non-leaf node of the decision tree must have a string key called "feature" to represent
# which feature to test next (a painting's size, medium, or color, for example), AND a string key called "bias" to represent
# the majority label of data at that node. If there isn't an applicable value for the feature to split on (ie, the painting 
# that we are evaluating is yellow, but we have never seen a yellow painting before), then we classify that datum as whatever the "bias"
# value is.
DecisionTree = Union[Dict[str, 'DecisionTree'], str]

# Define a function type that splits a dataframe into two dataframes.
DFPartitioner = Callable[[pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]]

class DecisionTreeTrainer:
    def __init__(self, data: pd.DataFrame, label_col: str, feature_cols: List[str]):
        """
        DecisionTreeTrainer constructor. As input, we take a Pandas dataframe called *data* that has *feature_cols* as its columns
        along with a label column called *label_col*.
        """
        self.label_col = label_col
        self.feature_cols = feature_cols
        self.data = data

    # private
    def _entropy(self, series: pd.Series) -> float:
        """
        Given a pandas series, return the entropy of the set.
        """
        # fun fact: about 40 percent of the time doing a cross-fold validation is spent on this single line
        # find the percentage of the series with respect to each element:
        probs = series.value_counts(normalize=True)
        return -np.sum(probs * np.log2(probs))
    
    # private
    def _info_gain(self, df: pd.DataFrame, split_col: str, df_entropy: float) -> float:
        """
            df: a pandas dataframe
            split col: the column that the data is being split on. Ie, the question that is being asked of the data.
            df_entropy: the entropy of label column of df
            return the information gain from ascertaining the value of split_col. 
        """
        def agg_ent_size(group):
            return pd.Series({'size': len(group), 'entropy': self._entropy(group)})
            
        entropies = (df
                .groupby(split_col)[[self.label_col]]
                .apply(agg_ent_size))
        entropies['weighted_entropy'] = entropies['entropy'] * entropies['size'] / df.shape[0]
        # apply the entropy calculation
        return df_entropy - np.sum(entropies['weighted_entropy'])


    # private
    def _induce_tree(self, training: pd.DataFrame, parent_bias: str, splittable_cols: List[str]) -> DecisionTree:
        """
        Induce a decision tree from training data in a pandas dataframe. If the labels in the dataset are evenly split,
        decide in favor the provided bias. 
        """
        # if there is only one type of value remaining in our induced subgroup, then we have hit our basecase
        counts = training[self.label_col].value_counts()
        if counts.shape[0] == 1:
            return cast(str, training[[self.label_col]].iloc[0][self.label_col])
        
        # figure out how our results should be biased in the case that our data is completely evenly labled.
        # ie if each label has the same number of data points
        curr_bias: str = parent_bias if counts.duplicated(keep=False).all() else str(counts.idxmax())
        # if we are out of columns that we can split on or all of the data points are identical, 
        # then we have reached our other basecase. Return the bias:
        if (not splittable_cols) or training[self.feature_cols].duplicated(keep=False).all():
            return curr_bias
        
        # make a tuple of tuples where each subtuple contains the information gain and the column to split on
        df_entropy: float = self._entropy(training[self.label_col])
        info_gains: Tuple[Tuple[float, str],...] = tuple((self._info_gain(training, feature_col, df_entropy), feature_col) for feature_col in splittable_cols)
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
            # remove the feature that we just split on from the list of splittable columns
            next_splittable_cols: List[str] = list(splittable_cols)
            next_splittable_cols.remove(split_feature)
            tree[field] = self._induce_tree(induced_subset, curr_bias, next_splittable_cols)
        return tree
        
        

    # private
    def _classify_datum(self, datum: pd.Series, root: DecisionTree) -> str:
        """
        Given a data point and a decision tree, classify the data point.
        """
        curr = root
        while not isinstance(curr, str):
            feat = datum[cast(str, curr['feature'])]
            if feat in curr:
                curr = curr[feat]
            else:
                return cast(str, curr['bias'])
        return curr
    
    # private
    def _test_tree(self, testing: pd.DataFrame, root: DecisionTree) -> float:
        """
        Given a testing set, and a decision tree, find the accuracy of the decision tree on the test set.
        """
        # apply our _classify_datum function across each row
        classifications = testing.apply(lambda row: self._classify_datum(row, root), axis=1)
        # get all the rows where our classification was correct:
        matches = testing.loc[testing[self.label_col] == classifications]
        # return the percenttage that we got right
        return matches.shape[0]/testing.shape[0]
    
    def _count_nodes_in_tree(self, tree: DecisionTree) -> int:
        # base case. we are at a leaf node:
        if isinstance(tree, str):
            return 1
        # recursive case. We have to count this node and all the subnodes
        count = 1
        for key,sub_tree in tree.items():
             if key not in ('feature', 'bias'):
                count += self._count_nodes_in_tree(sub_tree)

        return count

    # private
    def _prune_tree_greedily(self, tuning: pd.DataFrame, sub_tree: DecisionTree, root: DecisionTree) -> Tuple[float, DecisionTree]:
        """
        Given a tuning set, a subtree, and the root of the entire tree to which the subtree belongs, return a tree with the single best pruning
        operation having been performed within the subtree. 
        """
        best_pruned_tree = sub_tree
        best_pruned_tree_score = 0.0
        # if we have found a leaf, return a really low score so that it doesn't get pruned. In reality, this should
        # never happen because of a check in the while loop before we recurse. It's mostly to appease the typechecker :(
        if isinstance(sub_tree, str) or isinstance(root, str):
            return best_pruned_tree_score, best_pruned_tree
        
        for key in sub_tree.keys():
            # turn a sub-tree of this sub-tree into a leaf-node with the majority label at this sub-tree
            # ie, we are pruning a sub-tree of this sub-tree into a leaf
            temp = sub_tree[key]
            # if the sub-sub-tree is a leaf or metadata, then continue
            if key in ('feature', 'bias') or isinstance(temp, str):
                continue
            # prune the given child node into a leaf
            sub_tree[key] = cast(Dict[str, DecisionTree], sub_tree[key])['bias']
            # if this pruning operation was the best that we've seen so far, then make it our new best
            tree_score = self._test_tree(tuning, root)
            if tree_score > best_pruned_tree_score or (tree_score == best_pruned_tree_score and 
                                                       self._count_nodes_in_tree(root) < self._count_nodes_in_tree(best_pruned_tree)):
                best_pruned_tree = deepcopy(root)
                best_pruned_tree_score = tree_score
            # undo our pruning
            sub_tree[key] = temp
            # get the best pruning from a branch further down the tree from where we are now:
            best_pruned_child_tree_score, best_pruned_child_tree = self._prune_tree_greedily(tuning, sub_tree[key], root)
            # if it's better than any pruning we've found so far, set it as our best found pruning
            if best_pruned_child_tree_score > best_pruned_tree_score or (best_pruned_child_tree_score == best_pruned_tree_score and 
                                                       self._count_nodes_in_tree(best_pruned_child_tree) < self._count_nodes_in_tree(best_pruned_tree)):
                best_pruned_tree = best_pruned_child_tree
                best_pruned_tree_score = best_pruned_child_tree_score

        return best_pruned_tree_score, best_pruned_tree

    # private
    def _tuned_tree(self, data: pd.DataFrame, training_tuning_partitioner: DFPartitioner) ->Tuple[float, DecisionTree]:
        """
        Return a tree trained using the training set inferred from training_tuning_partitioner and pruned using the tuning set inferred from
        training_tuning_partitioner. The accuracy of the tree on the tuning set is also returned.
        """
        training, tuning = training_tuning_partitioner(data)
        # calculate the bias of the training set and make a decision tree:
        bias = str(training[self.label_col].value_counts().idxmax())
        tree = self._induce_tree(training, bias, list(self.feature_cols))
        # test the tree on the tuning set
        score = self._test_tree(tuning, tree)
        # prune the tree and test it on the tuning set
        pruned_score, pruned_tree = self._prune_tree_greedily(tuning, tree, tree)
        # keep pruning the tree until the best pruning doesn't produce a better score
        while pruned_score >= score:
            score = pruned_score
            tree = pruned_tree
            pruned_score, pruned_tree = self._prune_tree_greedily(tuning, tree, tree)

        return score, tree
    
    def tuned_tree(self, training_tuning_partitioner: DFPartitioner) -> Tuple[float, DecisionTree]:
        """
        Return a tree trained using the training set inferred from training_tuning_partitioner and pruned using the tuning set inferred from
        training_tuning_partitioner. The accuracy of the tree on the tuning set is also returned.
        """
        return self._tuned_tree(self.data, training_tuning_partitioner)

    def _test_fold(self, nth_fold: int, folds: int, training_tuning_partitioner: DFPartitioner):
        """
        When doing a cross validation, return the tested accuracy of the nth_fold.
        """
        non_testing = self.data.loc[self.data.index % folds != nth_fold]
        testing = self.data.loc[self.data.index % folds == nth_fold]
        _, minus_one_tree = self._tuned_tree(non_testing, training_tuning_partitioner)
        return self._test_tree(testing, minus_one_tree)

    def cross_validate(self, folds: int, training_tuning_partitioner: DFPartitioner, parallel: bool) -> Tuple[float, DecisionTree]:
        """
        Induce and train a tree, then perform an n-fold-cross-validation to obtain an accuracy. Return the pruned tree and its estimated
        accuracy. Toggle "parallel" for multiprocessing. If multiprocessing, make sure that all module-level user code is wrapped in an
        "if __name__ == '__main__':" block. 
        """
        if not parallel:
            _, tree = self._tuned_tree(self.data, training_tuning_partitioner)
            score = 0.0
            for i in range(folds):
                score += self._test_fold(i, folds, training_tuning_partitioner)
            # return an average score across all trees and tests:
            return score/folds, tree

        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        # for some reason, we must define a partial function which is basically just process_fold, but always with the same set data and partitioner
        # (this is to make the multiprocessing api happy)
        partial_test_fold = partial(self._test_fold, folds=folds, training_tuning_partitioner=training_tuning_partitioner)
        # map each fold function to its own process
        results = pool.map(partial_test_fold, range(folds))
        # after we set all process to work, calculate the tree that we are testing on the main process:
        _, tree = self._tuned_tree(self.data, training_tuning_partitioner)
        # closs the pool and reap children
        pool.close()
        pool.join()
        
        # calculate the average score across all tests
        score = sum(results) / folds
        return score, tree
    
    # private
    @staticmethod
    def _print_tree(tree: Dict[str, DecisionTree], decorator: str, depth: int) -> None:
        """
        Prettyprint a decision tree!  
        """
        for sub_feature, sub_tree in tree.items():
            if sub_feature not in ('bias', 'feature'):
                indent = ('   '*depth)
                if isinstance(sub_tree, str):
                    line = f'{indent}{sub_feature} {sub_tree}'
                    print(line)
                else:
                    line = f'{indent}{sub_feature} {decorator} {sub_tree["feature"]}:'
                    print(line)
                    DecisionTreeTrainer._print_tree(sub_tree, decorator, depth + 1)

    @staticmethod
    def print_tree(tree: DecisionTree, decorator='') -> None:
        """
        Prettyprint a decision tree! The decorator label will appear next to every non-leaf node's feature name.
        """
        DecisionTreeTrainer._print_tree({'':tree}, decorator, 0)
