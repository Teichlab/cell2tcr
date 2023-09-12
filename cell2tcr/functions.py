import tcrdist
from tcrdist.repertoire import TCRrep
import igraph
import leidenalg
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import logomaker

def motifs(df, sparse=True, threshold=35, chunk_size=3000, return_distances=False, add_suffix=True):
    '''
    Compute and cluster the TCR distance matrix.
    
    df : pd.DataFrame. Needs to have fields 'individual', 'IR_VDJ_1_junction_aa', 'IR_VDJ_1_v_call', 'IR_VDJ_1_j_call', 'IR_VJ_1_junction_aa', 'IR_VJ_1_v_call', 'IR_VJ_1_j_call'.
    sparse : bool. Select sparse=True implementation if more than ~1000 TCR clones are given.
    threshold : int. Threshold used to connect TCR distance matrix.
    chunk_size : int. Number of rows loaded into memory for sparse implementation.
    return_distances: bool. Whether to return the tcrdist object that also holds the distances, or modify the initial dataframe with the new column 'motif' in-place.
    add_suffix: bool. Whether to add generic *01 suffix to gene names. 
    '''
    # add tcrdist-compatible column names
    for i, j in zip(
        ['subject', 'cdr3_b_aa', 'v_b_gene', 'j_b_gene', 'cdr3_a_aa', 'v_a_gene', 'j_a_gene'],
        ['individual','IR_VDJ_1_junction_aa','IR_VDJ_1_v_call','IR_VDJ_1_j_call','IR_VJ_1_junction_aa','IR_VJ_1_v_call','IR_VJ_1_j_call']):
        df.loc[:,i] = df.loc[:,j]

    # add generic allele suffix
    if add_suffix:
        for genes in ['v_b_gene', 'j_b_gene', 'v_a_gene', 'j_a_gene']:
            df.loc[:,genes] = df.loc[:,genes].astype(str) + '*01'

    # load gene list
    tcrdist_genes = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'alphabeta_gammadelta_db.tsv'), sep='\t')
    
    # check which genes are not found in tcrdist list
    for gene in df[['v_b_gene', 'j_b_gene', 'v_a_gene', 'j_a_gene']].unstack().unique():
        if gene not in tcrdist_genes.id.values:
            # try quick fix
            if gene.replace('DV','/DV') in tcrdist_genes.id.values:
                df.replace(gene, gene.replace('DV','/DV'), inplace=True)
            else:
                raise Exception(f'VDJ gene {gene} not found in alphabeta_gammadelta_db.txt, tcrdist will error out!')

    # compute unique clone_id
    df.loc[:,'clone_id'] = df.groupby(['subject','cdr3_b_aa', 'v_b_gene', 'j_b_gene', 'cdr3_a_aa', 'v_a_gene', 'j_a_gene'], sort=False).ngroup()
    
    r = threshold # distance threshold
    
    if sparse:
        tr = TCRrep(
            cell_df = df.drop_duplicates(subset = 'clone_id'),
            organism = 'human', 
            chains = ['alpha', 'beta'], 
            compute_distances = False, # sparse
            deduplicate = False,
            infer_index_cols = False,
            index_cols = ['clone_id'],
            cpus=24)


        # modify chunk_size depending on RAM
        tr.compute_sparse_rect_distances(radius = r, chunk_size = chunk_size)

        # get chains and set diagonal to 0
        a = tr.rw_alpha.copy()
        b = tr.rw_beta.copy()
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
            chains = ['alpha', 'beta'], 
            compute_distances = True, # dense
            deduplicate = False,
            infer_index_cols = False,
            index_cols = ['clone_id'],
            cpus=24)
        g = igraph.Graph.Adjacency((tr.pw_alpha+tr.pw_beta) < r)
    
    # Leiden clustering
    partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=1, seed=1)
    tr.clone_df['motif'] = pd.DataFrame(partition.membership).values
    
    if return_distances:
        return tr
    else:
        # assign motif to each original cell
        df['motif'] = df.clone_id.map(tr.clone_df[['clone_id','motif']].set_index('clone_id').motif.to_dict())

def draw_cdr3(df, skip_singletons=False, savefig_title=None, put_title=True, transparent=False):
    '''
    df : pd.DataFrame. Needs to have fields 'subject', 'clone_id', 'cdr3_a/g_aa', 'cdr3_b/d_aa'. Draws the CDR3 alpha and beta logo over all the entries in df, using the most common length. Can handle both alpha beta and gamma delta TCRs.
    skip_singletons : bool. Whether to skip motifs comprised of a single clone.
    savefig_title : None or str. If provided, save figure in savedir and using given title.
    put_title : bool. Whether to display the title.
    transparent : bool. Make background transparent (e.g. for saving the figure).
    '''
    if hasattr(df, 'cdr3_a_aa') and hasattr(df, 'cdr3_b_aa'):
        # Alpha beta TCR 
        cdr3_vj_aa = 'cdr3_a_aa'
        cdr3_vdj_aa = 'cdr3_b_aa'
    elif hasattr(df, 'cdr3_g_aa') and hasattr(df, 'cdr3_d_aa'):
        # Gamma delta TCR 
        cdr3_vj_aa = 'cdr3_g_aa'
        cdr3_vdj_aa = 'cdr3_d_aa'
    else:
        raise 'No cdr3_aa found.'
    n_shared, n_clones = df.nunique()[['subject','clone_id']].values
    if skip_singletons:
        if n_clones == 1:
            return
    fig, ax = plt.subplots(ncols=2, figsize=(10,1))
    title =  f'Shared by: {n_shared},  Unique clones: {n_clones}'
    for chain_ind, chain in enumerate([cdr3_vj_aa,cdr3_vdj_aa]):
        cdr3 = df[[cdr3_vj_aa,cdr3_vdj_aa]].copy()
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
        plt.suptitle(title, y=1.2, x=0.6)
    if savefig_title is not None:
        plt.savefig(savefig_title, transparent=transparent)
    plt.show()