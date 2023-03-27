import tcrdist
from tcrdist.repertoire import TCRrep
import igraph
import leidenalg
import pandas as pd
import os

def tcrdist_motifs(df_clone, sparse=True, return_distances=False):
    '''
    Compute and cluster the TCR distance matrix.
    
    df_clone : a pd.DataFrame. Needs to have fields 'subject', 'cdr3_b_aa', 'v_b_gene', 'j_b_gene', 'cdr3_a_aa', 'v_a_gene', 'j_a_gene'.
    sparse : a bool. Select sparse=True implementation if more than ~1000 TCR clones are given.
    return_distances: a bool. Whether to return the tcrdist object that also holds the distances, or modify the initial dataframe with the new column 'motif' in-place.
    '''
    # add tcrdist-compatible column names
    for i, j in zip(
        ['subject', 'cdr3_b_aa', 'v_b_gene', 'j_b_gene', 'cdr3_a_aa', 'v_a_gene', 'j_a_gene'],
        ['individual', 'IR_VDJ_1_junction_aa','IR_VDJ_1_v_call','IR_VDJ_1_j_call','IR_VJ_1_junction_aa','IR_VJ_1_v_call','IR_VJ_1_j_call']):
        df_clone[i] = df_clone[j]

    # add generic allele suffix for tcrdist compatibility
    for genes in ['v_b_gene', 'j_b_gene', 'v_a_gene', 'j_a_gene']:
        df_clone[genes] = df_clone[genes].astype(str) + '*01'

    # load tcrdist gene list
    tcrdist_genes = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'alphabeta_gammadelta_db.tsv'), sep='\t')
    
    # check which genes are not found in tcrdist list
    for gene in df_clone[['v_b_gene', 'j_b_gene', 'v_a_gene', 'j_a_gene']].unstack().unique():
        if gene not in tcrdist_genes.id.values:
            # try quick fix
            if gene.replace('DV','/DV') in tcrdist_genes.id.values:
                df_clone.replace(gene, gene.replace('DV','/DV'), inplace=True)
            else:
                raise Exception(f'VDJ gene {gene} not found in alphabeta_gammadelta_db.txt, tcrdist will error out!')

    # compute unique clone_id
    df_clone['clone_id'] = df_clone.groupby(['subject','cdr3_b_aa', 'v_b_gene', 'j_b_gene', 'cdr3_a_aa', 'v_a_gene', 'j_a_gene'], sort=False).ngroup()
    
    r = 35 # distance threshold
    
    if sparse:
        tr = TCRrep(
            cell_df = df_clone.drop_duplicates(subset = 'clone_id'),
            organism = 'human', 
            chains = ['alpha', 'beta'], 
            compute_distances = False, # sparse
            deduplicate = False,
            infer_index_cols = False,
            index_cols = ['clone_id'],
            cpus=24)


        # modify chunk_size depending on RAM
        tr.compute_sparse_rect_distances(radius = r, chunk_size = 3000)

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
            cell_df = df_clone.drop_duplicates(subset = 'clone_id'),
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
        df_clone['motif'] = df_clone.clone_id.map(tr.clone_df[['clone_id','motif']].set_index('clone_id').motif.to_dict())