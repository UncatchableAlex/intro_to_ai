from decision_tree_trainer import DecisionTreeTrainer
import sys
from typing import Tuple, List, TextIO, cast
import pandas as pd
import string
from decision_tree_trainer import DecisionTreeTrainer


def training_tuning_partitioner(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return data.loc[lambda row: row.index % 4 != 0], data[::4]

def makeDecisionTree() -> None:
    # check if the user has provided a file name as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python decision-tree-inducer.py file_name.tsv")
        sys.exit(1)

    file_name: str = sys.argv[1]

    try:
        tsvfile: TextIO = open(file_name, newline='')
            
    except FileNotFoundError:
        print(f"The file {file_name} was not found.")
        return

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

    features =  list(string.ascii_lowercase[0:10])
    dtt = DecisionTreeTrainer(df, 'Party', features)
    # perform a leave-one-out accuracy test
    pruned_acc, pruned_tree = dtt.cross_validate(df.shape[0], training_tuning_partitioner, True)
    # print the resulting tree and its accuracy
    print(pruned_acc)
    DecisionTreeTrainer.print_tree(pruned_tree, decorator='Issue')


# wrap our module-level code in the __main__ check to facilitate multiprocessing
if __name__ == '__main__':
    makeDecisionTree()


