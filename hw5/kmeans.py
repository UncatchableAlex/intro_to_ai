############   I'm trying out a new formatter (https://github.com/psf/black). All weird formatting is their fault.
import sys
import numpy as np
import pandas as pd
import string

if len(sys.argv) != 3:
    print("usage: python kmeans.py <data_file.py> <num centroids>")
    sys.exit(0)

if not sys.argv[2].isdigit():
    print(
        "expected second command line argument (number of centroids) to be a digit >= 2"
    )
    sys.exit(0)

if int(sys.argv[2]) < 2:
    print("second command-line argument (number of centroids) must be >= 2")
    sys.exit(0)


def squared_distance_mat(pts_a, pts_b):
    """
    Make a distance array with shape (n,m) describing the distance between the ith point in pts_a and the jth point in pts_b
    for 0 <= i < n and 0 <= j < m

    :param pts_a: numpy array with shape (n,k)
    :param pts_b: numy array with shape (m,k)
    :return: a 2d numpy array of the squared distances between every point in pts_a and pts_b
    """

    # https://stackoverflow.com/a/60854278/12568696
    # first we add new dimensions to our data and use numpy broadcasting to subtract each combination of coordinates
    # then we square our differences and sum the results into a matrix of distances between each pair of coordinates
    return ((pts_a[:, np.newaxis, :] - pts_b[np.newaxis, :, :]) ** 2).sum(axis=2)


def kmeans(df, feature_cols, label_col, name_col, k):
    """
    Find and print k clusters of data in df.
    :param df: the dataframe to perform k means on
    :feature_cols: the columns of df to perform k-means on
    :label_col: the column to use to report cluster statistics at the end
    :name_col: the column to use to report initial starting positions of centroids
    :return: None
    """
    dists = squared_distance_mat(df[feature_cols].values, df[feature_cols].values)

    # initialize our first k centroids:

    # https://stackoverflow.com/a/75865949/12568696
    # np.argmax will return the maximum value in the lowest numbered column/row first. This corresponds with our
    # tie break rules in this assignment
    furthest_two = np.unravel_index(np.argmax(dists), dists.shape)
    centroids_idxs = list(furthest_two)
    for _ in range(2, k):
        # calculate the distance from each of our centroids to every other point
        squared_dists = squared_distance_mat(
            df.iloc[centroids_idxs][feature_cols].values, df[feature_cols].values
        )
        dists = squared_dists
        # set the distance between centroids to zero (we don't want a centroid to get chosen twice in the case of colinear data)
        dists[:, centroids_idxs] = 0
        # for every point, sum its distance to each centroid:
        dists = dists.sum(axis=0)
        # find the point with the greatest sum of distances to each centroid
        centroids_idxs.append(np.argmax(dists))

    print("Initial centroids based on:\t" + ", ".join(df.loc[centroids_idxs][name_col]))
    cluster_cols = feature_cols + ["cluster"]
    centroids = df[feature_cols].iloc[centroids_idxs]
    iter_count = 0
    prev_assignments = None
    while True:
        squared_dists = squared_distance_mat(
            centroids[feature_cols].values, df[feature_cols].values
        )
        # assign each point to a centroid
        df["cluster"] = np.argmin(squared_dists, axis=0)
        # check to see if any points were given new assignments
        if prev_assignments is not None and (prev_assignments == df["cluster"]).all():
            break
        # save the old assignments, and find new centroids
        prev_assignments = df["cluster"]
        # put each centroid at the center of its group
        prev_centroids = centroids
        prev_centroids["cluster"] = np.arange(0, len(prev_centroids), 1)
        # update the position of all centroids that have data points in their cluster
        centroids = df[cluster_cols].groupby("cluster").mean().reset_index()
        # add back any centroids that didn't have any data points in their cluster (just don't move them from last iteration)
        if len(prev_centroids) != len(centroids):
            # find the missing centroids from the prev_centroids table
            # https://stackoverflow.com/questions/52674718/pandas-is-there-a-way-to-compare-two-series-and-find-elements-that-are-recent-i
            missing_centroids = prev_centroids[
                ~prev_centroids["cluster"].isin(centroids["cluster"])
            ]
            # add the missing values back in to centroids and sort
            centroids = (
                pd.concat([centroids, missing_centroids], axis=0)
                .sort_values("cluster")
                .reset_index(drop=True)
            )
        iter_count += 1

    # group the data by cluster and count the labels
    label_counts = df.groupby(["cluster", label_col]).size().reset_index(name="count")
    cluster_sizes = (
        label_counts.groupby("cluster")["count"].sum().reset_index(name="size")
    )
    print(f"Converged after {iter_count} rounds of k-means.")
    # format the output
    for cluster, group in cluster_sizes.iterrows():
        cluster_id = group["cluster"]
        cluster_size = group["size"]
        label_counts_cluster = label_counts[label_counts["cluster"] == cluster_id]

        output_str = f"\tGroup {cluster_id}: size {cluster_size} ("

        for label, label_group in label_counts_cluster.groupby(label_col):
            label_count = label_group["count"].sum()
            label_percentage = (label_count / cluster_size) * 100
            output_str += f"{label_percentage:.3f}% {label}, "

        output_str = output_str[:-2] + ")"  # remove the trailing comma and space
        print(output_str)


tsvfile = sys.argv[1]
num_centroids = int(sys.argv[2])
try:
    df = pd.read_csv(tsvfile, sep="\t", header=None, names=["Rep", "Party", "Votes"])

    # concatenate df with the Votes column split into new columns for each issue (each character in the string).
    # Note that the regular expression that we split on is a positive lookbehind (checking for '+'s, '-'s, and '.'s) and a positive lookahead
    # (checking for the same characters). We don't split on the empty string because then the zeroth and last columns would be empty.
    # https://pandas.pydata.org/docs/reference/api/pandas.Series.str.split.html#pandas.Series.str.split
    df = pd.concat(
        [df, df["Votes"].str.split(r"(?<=\+|\-|\.)(?=\+|\-|\.)", expand=True)], axis=1
    )
    # rename the numbered columns to letters indicating which issue is being voted on:
    new_cols = [
        string.ascii_lowercase[int(col)] if isinstance(col, int) else col
        for col in df.columns
    ]
    feature_cols = [
        string.ascii_lowercase[int(col)] for col in df.columns if isinstance(col, int)
    ]
    df.columns = pd.Index(new_cols)
    print(new_cols, feature_cols)
    # silence annoying warning messages about """safety"""
    pd.set_option("future.no_silent_downcasting", True)
    # an error message that I get when I remove the previous line suggests using copy=False as a parameter for infer_objects, but my linter goes crazy when I do that. This code seems to work though
    df[feature_cols] = (
        df[feature_cols].replace({"+": 1, "-": -1, ".": 0}).infer_objects()
    )
    kmeans(df, feature_cols, "Party", "Rep", k=num_centroids)

except FileNotFoundError:
    print("could not find entered file")
