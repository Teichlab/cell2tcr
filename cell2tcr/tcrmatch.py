import multiprocessing as mp
import Levenshtein
import itertools
import pandas as pd
import numpy as np
import os
import re

# implementation based on https://github.com/IEDB/TCRMatch/blob/master/src/tcrmatch.cpp
# matrix hardcoded in original implementation
blm = np.array(
    [
        [
            0.0215,
            0.0023,
            0.0019,
            0.0022,
            0.0016,
            0.0019,
            0.003,
            0.0058,
            0.0011,
            0.0032,
            0.0044,
            0.0033,
            0.0013,
            0.0016,
            0.0022,
            0.0063,
            0.0037,
            0.0004,
            0.0013,
            0.0051,
        ],
        [
            0.0023,
            0.0178,
            0.002,
            0.0016,
            0.0004,
            0.0025,
            0.0027,
            0.0017,
            0.0012,
            0.0012,
            0.0024,
            0.0062,
            0.0008,
            0.0009,
            0.001,
            0.0023,
            0.0018,
            0.0003,
            0.0009,
            0.0016,
        ],
        [
            0.0019,
            0.002,
            0.0141,
            0.0037,
            0.0004,
            0.0015,
            0.0022,
            0.0029,
            0.0014,
            0.001,
            0.0014,
            0.0024,
            0.0005,
            0.0008,
            0.0009,
            0.0031,
            0.0022,
            0.0002,
            0.0007,
            0.0012,
        ],
        [
            0.0022,
            0.0016,
            0.0037,
            0.0213,
            0.0004,
            0.0016,
            0.0049,
            0.0025,
            0.001,
            0.0012,
            0.0015,
            0.0024,
            0.0005,
            0.0008,
            0.0012,
            0.0028,
            0.0019,
            0.0002,
            0.0006,
            0.0013,
        ],
        [
            0.0016,
            0.0004,
            0.0004,
            0.0004,
            0.0119,
            0.0003,
            0.0004,
            0.0008,
            0.0002,
            0.0011,
            0.0016,
            0.0005,
            0.0004,
            0.0005,
            0.0004,
            0.001,
            0.0009,
            0.0001,
            0.0003,
            0.0014,
        ],
        [
            0.0019,
            0.0025,
            0.0015,
            0.0016,
            0.0003,
            0.0073,
            0.0035,
            0.0014,
            0.001,
            0.0009,
            0.0016,
            0.0031,
            0.0007,
            0.0005,
            0.0008,
            0.0019,
            0.0014,
            0.0002,
            0.0007,
            0.0012,
        ],
        [
            0.003,
            0.0027,
            0.0022,
            0.0049,
            0.0004,
            0.0035,
            0.0161,
            0.0019,
            0.0014,
            0.0012,
            0.002,
            0.0041,
            0.0007,
            0.0009,
            0.0014,
            0.003,
            0.002,
            0.0003,
            0.0009,
            0.0017,
        ],
        [
            0.0058,
            0.0017,
            0.0029,
            0.0025,
            0.0008,
            0.0014,
            0.0019,
            0.0378,
            0.001,
            0.0014,
            0.0021,
            0.0025,
            0.0007,
            0.0012,
            0.0014,
            0.0038,
            0.0022,
            0.0004,
            0.0008,
            0.0018,
        ],
        [
            0.0011,
            0.0012,
            0.0014,
            0.001,
            0.0002,
            0.001,
            0.0014,
            0.001,
            0.0093,
            0.0006,
            0.001,
            0.0012,
            0.0004,
            0.0008,
            0.0005,
            0.0011,
            0.0007,
            0.0002,
            0.0015,
            0.0006,
        ],
        [
            0.0032,
            0.0012,
            0.001,
            0.0012,
            0.0011,
            0.0009,
            0.0012,
            0.0014,
            0.0006,
            0.0184,
            0.0114,
            0.0016,
            0.0025,
            0.003,
            0.001,
            0.0017,
            0.0027,
            0.0004,
            0.0014,
            0.012,
        ],
        [
            0.0044,
            0.0024,
            0.0014,
            0.0015,
            0.0016,
            0.0016,
            0.002,
            0.0021,
            0.001,
            0.0114,
            0.0371,
            0.0025,
            0.0049,
            0.0054,
            0.0014,
            0.0024,
            0.0033,
            0.0007,
            0.0022,
            0.0095,
        ],
        [
            0.0033,
            0.0062,
            0.0024,
            0.0024,
            0.0005,
            0.0031,
            0.0041,
            0.0025,
            0.0012,
            0.0016,
            0.0025,
            0.0161,
            0.0009,
            0.0009,
            0.0016,
            0.0031,
            0.0023,
            0.0003,
            0.001,
            0.0019,
        ],
        [
            0.0013,
            0.0008,
            0.0005,
            0.0005,
            0.0004,
            0.0007,
            0.0007,
            0.0007,
            0.0004,
            0.0025,
            0.0049,
            0.0009,
            0.004,
            0.0012,
            0.0004,
            0.0009,
            0.001,
            0.0002,
            0.0006,
            0.0023,
        ],
        [
            0.0016,
            0.0009,
            0.0008,
            0.0008,
            0.0005,
            0.0005,
            0.0009,
            0.0012,
            0.0008,
            0.003,
            0.0054,
            0.0009,
            0.0012,
            0.0183,
            0.0005,
            0.0012,
            0.0012,
            0.0008,
            0.0042,
            0.0026,
        ],
        [
            0.0022,
            0.001,
            0.0009,
            0.0012,
            0.0004,
            0.0008,
            0.0014,
            0.0014,
            0.0005,
            0.001,
            0.0014,
            0.0016,
            0.0004,
            0.0005,
            0.0191,
            0.0017,
            0.0014,
            0.0001,
            0.0005,
            0.0012,
        ],
        [
            0.0063,
            0.0023,
            0.0031,
            0.0028,
            0.001,
            0.0019,
            0.003,
            0.0038,
            0.0011,
            0.0017,
            0.0024,
            0.0031,
            0.0009,
            0.0012,
            0.0017,
            0.0126,
            0.0047,
            0.0003,
            0.001,
            0.0024,
        ],
        [
            0.0037,
            0.0018,
            0.0022,
            0.0019,
            0.0009,
            0.0014,
            0.002,
            0.0022,
            0.0007,
            0.0027,
            0.0033,
            0.0023,
            0.001,
            0.0012,
            0.0014,
            0.0047,
            0.0125,
            0.0003,
            0.0009,
            0.0036,
        ],
        [
            0.0004,
            0.0003,
            0.0002,
            0.0002,
            0.0001,
            0.0002,
            0.0003,
            0.0004,
            0.0002,
            0.0004,
            0.0007,
            0.0003,
            0.0002,
            0.0008,
            0.0001,
            0.0003,
            0.0003,
            0.0065,
            0.0009,
            0.0004,
        ],
        [
            0.0013,
            0.0009,
            0.0007,
            0.0006,
            0.0003,
            0.0007,
            0.0009,
            0.0008,
            0.0015,
            0.0014,
            0.0022,
            0.001,
            0.0006,
            0.0042,
            0.0005,
            0.001,
            0.0009,
            0.0009,
            0.0102,
            0.0015,
        ],
        [
            0.0051,
            0.0016,
            0.0012,
            0.0013,
            0.0014,
            0.0012,
            0.0017,
            0.0018,
            0.0006,
            0.012,
            0.0095,
            0.0019,
            0.0023,
            0.0026,
            0.0012,
            0.0024,
            0.0036,
            0.0004,
            0.0015,
            0.0196,
        ],
    ]
)

# construct k1 matrix based on marginals of blm
k1 = (
    blm / np.dot(np.sum(blm, axis=0)[:, None], np.sum(blm, axis=0)[None, :])
) ** 0.11387
# lists of lists are faster than numpy arrays. boo.
k1 = k1.tolist()

# prepare a quick translation dictionary of AA to index in k1/blm
alphstr = "ARNDCQEGHILKMFPSTWYV"
alphabet = {}
for i, char in enumerate(alphstr):
    alphabet[char] = i


def compute_k3(seq1, seq2):
    """
    Compute the K3 kernel for any pair of AA string sequences
    """
    k3 = 0
    k2 = np.zeros((len(seq1), len(seq2))).tolist()
    # this is a pretty smart algorithmic decision from the tcrmatch people
    # the k is the length (minus one) of the substring being compared
    # each possible substring start gets expanded by a single amino acid to compare
    for k in np.arange(np.min([len(seq1), len(seq2)])):
        for start1 in np.arange(len(seq1) - k):
            j1 = alphabet[seq1[start1 + k]]
            for start2 in np.arange(len(seq2) - k):
                j2 = alphabet[seq2[start2 + k]]
                term = k1[j1][j2]
                if k == 0:
                    k2[start1][start2] = term
                else:
                    k2[start1][start2] *= term
                k3 += k2[start1][start2]
    return k3


def score(seq1, seq2):
    """
    Compute the TCRMatch score for any pair of AA string sequences
    """
    return compute_k3(seq1, seq2) / np.sqrt(
        compute_k3(seq1, seq1) * compute_k3(seq2, seq2)
    )


# this is where the reimplementation ends and some smart stuff begins
def self_k3_parallel(seq_df, n_threads=10):
    """
    Pre-compute a set of AA sequences' K3 kernel of themselves.
    Returns the input data frame with an extra ``["k3"]`` column
    added.

    Input
    -----
    seq_df : ``pd.DataFrame``
        AA sequence in the first column of the data frame
    """
    with mp.Pool(n_threads) as pool:
        # K3 function needs two seqs on input, so just pass in same seq twice
        k3s = pool.starmap(compute_k3, zip(seq_df.iloc[:, 0], seq_df.iloc[:, 0]))
    seq_df["k3"] = k3s
    return seq_df


def compute_score_levenshtein(
    ind1, seq1in, ind2, seq2in, levenshtein_threshold=3, match_score=0.97
):
    """
    Try to compute the score for a pair of AA sequences, but do a
    preliminary check via Levenshtein to see that they are similar
    enough to each other to bother. Levenshtein is far quicker than
    computing K3, even with the self-K3s precomputed for each
    sequence and provided on input.

    Returns ``[ind1, ind2, score]`` if both the Levenshtein and score
    criteria are met, otherwise ``None``.

    Input
    -----
    ind1 : ``int``
        Index of the first sequence in its respective sequence set
    seq1in : ``list`` or ``tuple``
        Needs the actual AA string sequence in ``[0]`` and the
        pre-computed self-K3 in ``[-1]``
    ind2 : ``int``
        Index of the second sequence in its respective sequence set
    seq2in : ``list`` or ``tuple``
        Needs the actual AA string sequence in ``[0]`` and the
        pre-computed self-K3 in ``[-1]``
    """
    # the seq#ins are a list representation of df rows, maybe w/ metadata
    # the actual sequence is at the start, the self-k3 is at the end
    seq1 = seq1in[0]
    seq1k3 = seq1in[-1]
    seq2 = seq2in[0]
    seq2k3 = seq2in[-1]
    # levenshtein is wicked fast and serves as a good heuristic
    # to pull out only the promising candidates to do proper k3 on
    if Levenshtein.distance(seq1, seq2) <= levenshtein_threshold:
        k3 = compute_k3(seq1, seq2)
        score = k3 / np.sqrt(seq1k3 * seq2k3)
        if score > match_score:
            # just return the indices and the score for now
            return [ind1, ind2, score]
    # if the return is not hit, a None falls out


def compute_score_levenshtein_singlecdr(
    ind1, seq1in, seq2k3s, levenshtein_threshold=3, match_score=0.97
):
    """
    Compute TCRMatch scores with Levenshtein filtering for a single
    input CDR3 sequence against the entire IEDB database.

    Returns a list of lists, with entries of ``[ind1, ind2, score]``
    corresponding to indices of the CDR3 sequence and IEDB hit in
    their respective data frames.

    Input
    -----
    ind1 : ``int``
        Index of the first sequence in its respective sequence set
    seq1in : ``list`` or ``tuple``
        Needs the actual AA string sequence in ``[0]`` and the
        pre-computed self-K3 in ``[-1]``
    seq2k3s : ``list`` of ``list``
        Needs the actual AA string sequence in ``[0]`` of each
        element and the pre-computed self-K3 in ``[-1]` of each
        element
    """
    scores = []
    # pull out a single IEDB entry and compare it to the CDR3
    # pass index too for minimal output construction
    for ind2, seq2in in enumerate(seq2k3s):
        score = compute_score_levenshtein(
            ind1, seq1in, ind2, seq2in, levenshtein_threshold, match_score
        )
        # we don't care about Nones
        # in fact, they're actively harmful en masse by gunking up RAM
        if score is not None:
            scores.append(score)
    return scores


def _scores_pool_init(_seq2k3s, _levenshtein_threshold, _match_score):
    global seq2k3s, levenshtein_threshold, match_score
    seq2k3s = _seq2k3s
    levenshtein_threshold = _levenshtein_threshold
    match_score = _match_score


def pool_compute_score_levenshtein(ind1, seq1k3):
    # make use of the globals created in _scores_pool_init
    # require less input to be ferried from the map
    return compute_score_levenshtein_singlecdr(
        ind1, seq1k3, seq2k3s, levenshtein_threshold, match_score
    )


def paired_scores_parallel(
    seq1k3s, seq2k3s, levenshtein_threshold=3, match_score=0.97, n_threads=10
):
    # turn innards of DFs to lists of lists for speed
    with mp.Pool(
        n_threads,
        _scores_pool_init,
        (
            seq2k3s.iloc[:, [0, -1]].values.tolist(),
            levenshtein_threshold,
            match_score,
        ),
    ) as pool:
        # pass index of CDR3 along with seq and self-K3
        scores = pool.starmap(
            pool_compute_score_levenshtein,
            enumerate(seq1k3s.iloc[:, [0, -1]].values.tolist()),
        )
    # this creates a list of lists, flatten to a single level
    # lists within the original lists of lists will remain unharmed
    return [item for sublist in scores for item in sublist]


def db_match(
    seqs,
    iedb,
    tcrmatch=None,
    trim=True,
    levenshtein_threshold=3,
    match_score=0.97,
    n_threads=10,
    temp_in="tcrmatch_input.txt",
    temp_out="tcrmatch_output.tsv"
):
    """
    Compute TCRMatch scores, either via the original C++ binary or a
    Python reimplementation, of a set of CDR3 AA sequences against
    the IEDB database. Returns a data frame formatted like the C++
    tool's output, with an extra column with the raw input sequence
    at the start.

    Input
    -----
    seqs : ``list`` of ``str``
        AA sequences of the CDR3s of interest, must be restricted to
        ``"ARNDCQEGHILKMFPSTWYV"`` characters.
    iedb : path
        Path to the TCRMatch-formatted IEDB database, can be downloaded
        from ``https://downloads.iedb.org/misc/TCRMatch/IEDB_data.tsv``.
    tcrmatch : path, optional (default: ``None``)
        Path to compiled TCRMatch C++ binary. If ``None``, will use
        Python reimplementation.
    trim : ``bool``, optional (default: ``True``)
        By default, the TCRMatch C++ binary removes the flanking AAs of
        query sequences if they start with ``C`` and end with ``F`` or
        ``W``. Setting ``trim`` to ``True`` will mirror this behaviour.
        If ``False``, will copy the raw input sequences to the
        ``trimmed_input_sequence`` column of the output. Trimming is
        disabled if calling the C++ binary from the wrapper.
    levenshtein_threshold : ``int``, optional (default: 3)
        The Python implementation will only compute the TCRMatch score
        for a pair of sequences if their Levenshtein distance is at
        most this much.
    match_score : ``float``, optional (default: 0.97)
        Sequence pairs will be reported as a hit if their score is
        greater than this threshold. Matches C++ binary default.
    n_threads : ``int``, optional (default: 10)
        Number of threads to use in the computation, used by both the
        C++ binary and Python reimplementation.
    temp_in : path, optional (default: ``tcrmatch_input.txt``)
        If using the C++ binary, file name to write the CDR3 input to.
        Will be deleted upon successful execution.
    temp_out : path, optional (default: ``tcrmatch_output.tsv``)
        If using the C++ binary, file name to write the output to. Will
        be deleted upon successful execution.
    """
    # convert input to list of unique sequences
    seqs = list(np.unique(seqs))
    # check that we have sequences that won't make tcrmatch unhappy
    invalid_count = np.sum(
        [len(re.findall("[^ARNDCQEGHILKMFPSTWYV]", i)) > 0 for i in seqs]
    )
    if invalid_count > 0:
        raise ValueError(
            str(invalid_count)
            + " input CDR3s have characters outside of ARNDCQEGHILKMFPSTWYV"
        )
    # trim input sequences to match what tcrmatch does internally by default
    if trim:
        seqs_trimmed = [
            i[1:-1]
            if (i.startswith("C") and (i.endswith("F") or i.endswith("W")))
            else i
            for i in seqs
        ]
    else:
        # just use the unmodified input
        seqs_trimmed = seqs.copy()
    if tcrmatch is not None:
        # prepare input in file form
        with open(temp_in, "w") as fid:
            fid.writelines([i + "\n" for i in seqs_trimmed])
        # -k means skip the trimming, as this was already decided python side
        os.system(
            tcrmatch
            + " -k -i "
            + temp_in
            + " -s "
            + str(match_score)
            + " -t "
            + str(n_threads)
            + " -d "
            + iedb
            + " > "
            + temp_out
        )
        # read output df and remove temporary files
        scores_df = pd.read_table(temp_out, index_col=False)
        os.remove(temp_in)
        os.remove(temp_out)
        # we did the trimming python side, reflect that in column names
        scores_df.rename(
            columns={"input_sequence": "trimmed_input_sequence"}, inplace=True
        )
        # convert the trimmed sequence to the original input sequence
        # and append into the data frame for easy locating of input
        seqs_map = pd.Series(seqs, index=seqs_trimmed)
        scores_df["input_sequence"] = scores_df["trimmed_input_sequence"].map(seqs_map)
        scores_df = scores_df[
            [
                "input_sequence",
                "trimmed_input_sequence",
                "match_sequence",
                "score",
                "receptor_group",
                "epitope",
                "antigen",
                "organism",
            ]
        ]
        return scores_df
    else:
        seqs_df = pd.DataFrame(zip(seqs_trimmed, seqs))
        iedb_df = pd.read_table(iedb, index_col=False)
        # reshuffle columns of IEDB DF to match tcrmatch output
        iedb_df = iedb_df.iloc[:, [0, 2, 3, 5, 4]]
        # compute self-K3 scores for both the queries and IEDB
        iedb_k3s = self_k3_parallel(iedb_df, n_threads=n_threads)
        seqs_k3s = self_k3_parallel(seqs_df, n_threads=n_threads)
        # get super minimal hit table - index in seqs/iedb and score
        scores = paired_scores_parallel(
            seqs_k3s,
            iedb_k3s,
            levenshtein_threshold=levenshtein_threshold,
            match_score=match_score,
            n_threads=n_threads,
        )
        # turn the metadata data frames to lists of lists for speed
        # strip out the k3s which seem to have made their way to the original dfs
        seqs_list = seqs_df.values[:, :2].tolist()
        iedb_list = iedb_df.values[:, :5].tolist()
        # pull out matching metadata for each hit
        scores_meta_list = []
        for hit in scores:
            scores_meta_list.append(seqs_list[hit[0]] + iedb_list[hit[1]] + [hit[2]])
        # reorder the columns to match what falls out of tcrmatch, plus untrimmed input at the start
        scores_df = pd.DataFrame(
            scores_meta_list,
            columns=[
                "trimmed_input_sequence",
                "input_sequence",
                "match_sequence",
                "receptor_group",
                "epitope",
                "antigen",
                "organism",
                "score",
            ],
        )[
            [
                "input_sequence",
                "trimmed_input_sequence",
                "match_sequence",
                "score",
                "receptor_group",
                "epitope",
                "antigen",
                "organism",
            ]
        ]
    return scores_df


def db_annotate(df, tcrmatch_df, cdr3_column, sep=";", repl_sep=",", multi_score=False):
    """
    Transfer the obtained TCRMatch hits back to the original data frame
    housing initial CDR3 query information.

    Returns the data frame with extra columns named after non-``input
    sequence`` columns of the TCRMatch hits, populated with hit
    information where applicable.

    Input
    -----
    df : ``pd.DataFrame``
        With CDR3 query sequences used for ``tcrmatch.tcrmatch()``
        present as one of the columns.
    tcrmatch_df : ``pd.DataFrame``
        ``tcrmatch.tcrmatch()`` output.
    cdr3_column : ``str``
        Column of ``df`` with CDR3 query information.
    sep : ``str``, optional (default: ``";"``)
        In the event of multiple hits per CDR3, join the hits with this
        delimiter when populating information into the original data
        frame.
    repl_sep : ``str``, optional (default: ``","``)
        If ``sep`` is encountered in the values of ``tcrmatch_df``,
        replace it with this value prior to the collapsing.
    multi_score : ``bool``, optional (default: ``False``)
        Whether to fill in score info on more than one match in case
        several matches are found. By default, only the highest scoring 
        match is filled in.
    """
    # do a deep copy of the input scores DF to avoid disrupting it
    tdf = tcrmatch_df.copy(deep=True)
    # turn any encountered instances of the separator with the replacement separator
    # regex needs to be true as this is replacing characters in strings
    tdf = tdf.replace({sep: repl_sep}, regex=True)
    if not multi_score:
        # only keep highest scoring entry
        tdf = tdf.sort_values('score', ascending=False).drop_duplicates('input_sequence')
    for col in tdf.columns[1:]:
        # construct a simple pd.Series of this particular column collapsed on CDR3 input
        # in case of multiple hits, they are joined with the separator
        mapper = tdf.groupby("input_sequence")[col].apply(
            lambda x: sep.join(x.astype(str))
        )
        # can use this series to map hit information to the original sequences
        # add as extra column in input df
        df.loc[:,col] = df.loc[:,cdr3_column].map(mapper)
    if not multi_score:
        # if several epitopes map to same TCR, only keep the first
        for col in ['epitope','antigen','organism']:
            df_match[col] = df_match[col].str.split(',', expand=True)[0]
    return df
