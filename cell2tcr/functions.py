import tcrdist
from tcrdist.repertoire import TCRrep
import igraph
import leidenalg
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import logomaker

def assign_column_names(df, donor_id='donor_id', vdj_aa='IR_VDJ_1_junction_aa', vdj_v='IR_VDJ_1_v_call', vdj_j='IR_VDJ_1_j_call', vj_aa='IR_VJ_1_junction_aa', vj_v='IR_VJ_1_v_call', vj_j='IR_VJ_1_j_call'):
    '''
    Make column names  tcrdist3-compatible. Modifies the dataframe in-place.
    
    df : pd.DataFrame. Needs to have the fields for donor, CDR3 gene calls and CDR3 sequence specified in the function call.
    donor_id : str. Name of column distinguishing different donors.
    vdj_aa : str. Name of column holding TCR-beta/delta CDR3 amino acid sequences.
    vdj_v : str. Name of column holding TCR-beta/delta V gene calls.
    vdj_j : str. Name of column holding TCR-beta/delta J gene calls.
    vj_aa : str. Name of column holding TCR-alpha/gamma CDR3 amino acid sequences.
    vj_v : str. Name of column holding TCR-alpha/gamma V gene calls.
    vj_j : str. Name of column holding TCR-alpha/gamma J gene calls. 
    '''
    missing = [column for column in [donor_id, vdj_aa, vdj_v, vdj_j, vj_aa, vj_v, vj_j] if not column in df.columns]
    if missing:
        raise KeyError(f'Columns {missing} not found in dataframe - check spelling.')
    # add tcrdist-compatible column names
    for i, j in zip(
        ['donor_id', 'vdj_aa', 'vdj_v', 'vdj_j', 'vj_aa', 'vj_v', 'vj_j'],
        [donor_id, vdj_aa, vdj_v, vdj_j, vj_aa, vj_v, vj_j]):
        df.loc[:,i] = df.loc[:,j]
    # TODO : test the input argument order was not mixed up by user

def motifs(df, sparse=True, threshold=35, chunk_size=3000, return_distances=False, add_suffix=True, organism='human', receptor_type='ab', db_file='alphabeta_gammadelta_db.tsv'):
    '''
    Compute and cluster the TCR distance matrix. Compatible with alpha-beta and gamma-delta T cells.
    
    df : pd.DataFrame. Needs to have fields 'donor_id', 'vdj_aa', 'vdj_v', 'vdj_j', 'vj_aa', 'vj_v', 'vj_j'.
    sparse : bool. Select sparse=True implementation if more than ~1000 TCR clones are given.
    threshold : int. Threshold used to connect TCR distance matrix.
    chunk_size : int. Number of rows loaded into memory for sparse implementation.
    return_distances: bool. Whether to return the tcrdist object that also holds the distances, or modify the initial dataframe with the new column 'motif' in-place.
    add_suffix: bool. Whether to add generic *01 suffix to gene names. 
    organism: str. Choose between 'human' and 'mouse'. 
    receptor_type: str. Choose between 'ab' and 'gd' for alpha-beta or gamma-delta T cells. 
    db_file: str. Choose between the built-in 'alphabeta_gammadelta_db.tsv', 'combo_xcr_2024-03-05.tsv', or load your own tsv. 
    '''
    # check relevant columns are present
    missing = [column for column in ['donor_id', 'vdj_aa', 'vdj_v', 'vdj_j', 'vj_aa', 'vj_v', 'vj_j'] if not column in df.columns]
    if missing:
        raise KeyError(f'Columns {missing} not found in dataframe - did you run cell2tcr.assign_column_names?')

    # add generic allele suffix
    if add_suffix:
        for genes in ['vdj_v', 'vdj_j', 'vj_v', 'vj_j']:
            df.loc[:,genes] = df.loc[:,genes].astype(str) + '*01'

    # load gene list
    if os.path.isfile(db_file):
        tcrdist_genes = pd.read_csv(db_file, sep='\t')
    elif os.path.isfile(os.path.join(os.path.dirname(os.path.realpath(__file__)), db_file)):
        tcrdist_genes = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), db_file), sep='\t')
    else:
        FileNotFoundError(f'Provide a valid db_file or choose one from "alphabeta_gammadelta_db.tsv","combo_xcr_2024-03-05.tsv". You provided: {db_file}')
    
    # check which genes are not found in tcrdist list
    for gene in df[['vdj_v', 'vdj_j', 'vj_v', 'vj_j']].unstack().unique():
        if gene not in tcrdist_genes.id.values:
            # try quick fix
            if gene.replace('DV','/DV') in tcrdist_genes.id.values:
                df.replace(gene, gene.replace('DV','/DV'), inplace=True)
            else:
                raise Exception(f'VDJ gene {gene} not found in {db_file}, tcrdist will error out!')

    # compute unique clone_id
    df.loc[:,'clone_id'] = df.groupby(['donor_id','vdj_aa', 'vdj_v', 'vdj_j', 'vj_aa', 'vj_v', 'vj_j'], sort=False).ngroup()
    
    r = threshold # distance threshold
    
    if receptor_type == 'ab':
        chains = ['alpha', 'beta']
        new_cols = {'cdr3_a_aa':'vj_aa', 'cdr3_b_aa':'vdj_aa', 'v_b_gene':'vdj_v', 'j_b_gene':'vdj_j', 'v_a_gene':'vj_v', 'j_a_gene':'vj_j',}
    
    elif receptor_type == 'gd':
        chains = ['gamma', 'delta']
        new_cols = {'cdr3_g_aa':'vj_aa', 'cdr3_d_aa':'vdj_aa', 'v_d_gene':'vdj_v', 'j_d_gene':'vdj_j', 'v_g_gene':'vj_v', 'j_g_gene':'vj_j',}
    else:
        raise ValueError(f'receptor_type {receptor_type} not valid - choose among "ab" and "gd" for alpha-beta or gamma-delta T cells')
    
    # make tcrdist-compatible
    df.rename(columns = {v: k for k, v in new_cols.items()}, inplace=True)
    
    if sparse:
        tr = TCRrep(
            cell_df = df.drop_duplicates(subset = 'clone_id'),
            organism = organism, 
            chains = chains, 
            compute_distances = False, # sparse
            deduplicate = False,
            infer_index_cols = False,
            index_cols = ['clone_id'],
            cpus=24,
        )


        # modify chunk_size depending on RAM
        tr.compute_sparse_rect_distances(radius = r, chunk_size = chunk_size)

        # get chains and set diagonal to 0
        if receptor_type == 'ab':
            a = tr.rw_alpha.copy()
            b = tr.rw_beta.copy()
        else:
            a = tr.rw_gamma.copy()
            b = tr.rw_delta.copy()
        a.setdiag(0)
        b.setdiag(0)

        # get indices of summed chains which lie within threshold*2
        # (these will equal 2 in c_inds)
        a_inds = a.copy()
        a_inds[a_inds!=0] = 1
        b_inds = b.copy()
        b_inds[b_inds!=0] = 1
        c_inds = a_inds+b_inds

        # sum chains
        c = a+b
        # subset to allowed indices
        c.data[c_inds.data != 2] = 0
        # apply threshold
        c[c>r] = 0
        c.eliminate_zeros()
        # binarize
        c[c!=0] = 1
        # create graph
        g = igraph.Graph.Adjacency(c)
        
    else:
        tr = TCRrep(
            cell_df = df.drop_duplicates(subset = 'clone_id'),
            organism = 'human', 
            chains = chains, 
            compute_distances = True, # dense
            deduplicate = False,
            infer_index_cols = False,
            index_cols = ['clone_id'],
            cpus=24)
        if receptor_type == 'ab':
            g = igraph.Graph.Adjacency((tr.pw_alpha+tr.pw_beta) < r)
        else:
            g = igraph.Graph.Adjacency((tr.pw_gamma+tr.pw_delta) < r)
            
    
    # Leiden clustering
    partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=1, seed=1)
    tr.clone_df['motif'] = pd.DataFrame(partition.membership).values
    
    if return_distances:
        tr.clone_df.rename(columns = {v: k for k, v in new_cols.items()}, inplace=True) 
        return tr
    else:
        # assign motif to each original cell
        df['motif'] = df.clone_id.map(tr.clone_df[['clone_id','motif']].set_index('clone_id').motif.to_dict())
        df.rename(columns = new_cols, inplace=True) 

def draw_cdr3(
        df, 
        skip_singletons=False, 
        savefig_title=None, 
        put_title=True, 
        transparent=False,
        remove_duplicate_clones=False,
        ):
    '''
    df : pd.DataFrame. Needs to have fields 'donor_id', 'clone_id', 'vj_aa', 'vdj_aa'. Draws the CDR3 alpha and beta logo over all the entries in df, using the most common length. Can handle both alpha beta and gamma delta TCRs.
    skip_singletons : bool. Whether to skip motifs comprised of a single clone.
    savefig_title : None or str. If provided, save figure in savedir and using given title.
    put_title : bool|str. Whether to display any title, and optionally a user-defined title.
    transparent : bool. Make background transparent (e.g. for saving the figure).
    remove_duplicate_clones : bool. Remove clone_id duplicates before plotting.
    '''

    if not hasattr(df, 'vj_aa') and not hasattr(df, 'vdj_aa'):
        raise AttributeError('No "vj_aa" and "vdj_aa" found.')
    n_shared, n_clones = df[['donor_id','clone_id']].nunique().values
    if skip_singletons:
        if n_clones == 1:
            return
    fig, ax = plt.subplots(ncols=2, figsize=(10,1))
    if remove_duplicate_clones:
        df_ = df.drop_duplicates('clone_id')
    else:
        df_ = df
    for chain_ind, chain in enumerate(['vj_aa','vdj_aa']):
        cdr3 = df_[['vj_aa','vdj_aa']].copy()
        cdr3['length'] = cdr3[chain].apply(lambda x: len(x))
        n_rows = cdr3.length.mode()[0]
        cdr3 = cdr3[cdr3.length==n_rows]

        # AA frequencies for logo
        letters = np.unique(pd.DataFrame([list(x) for x in cdr3[chain]])).tolist()
        logo = pd.DataFrame(np.zeros((n_rows, len(letters))), columns=letters)
        for row in range(n_rows):
            vals = pd.DataFrame([list(x) for x in cdr3[chain]]).value_counts([row], normalize=True).reset_index(name='prop')
            logo.loc[row, vals[row]] = vals['prop'].values

        # plot logo
        logo_plt = logomaker.Logo(logo, color_scheme='chemistry', ax=ax[chain_ind])
        logo_plt.ax.grid(False)
        logo_plt.ax.axis(False)
    if put_title:
        if isinstance(put_title, str):
            plt.suptitle(put_title, y=1.2, x=0.6)
        else:
            # use default title
            plt.suptitle(f'Shared by: {n_shared},  Unique clones: {n_clones}', y=1.2, x=0.6)
    if savefig_title is not None:
        plt.savefig(savefig_title, transparent=transparent)
    plt.show()