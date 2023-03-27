import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logomaker

def draw_cdr3(tr_f, skip_singletons=False, specificity=False, savefig_title=None, put_title=True, transparent=False):
    '''
    tr_f : a pd.DataFrame. Needs to have fields 'subject', 'clone_id', 'cdr3_a/g_aa', 'cdr3_b/d_aa'. Draws the CDR3 alpha and beta logo over all the entries in tr_f, using the most common length. Can handle both alpha beta and gamma delta TCRs. Options to 
    
    skip_singletons : bool. Whether to skip comprised of a single clone.
    specificity : bool. Whether to find matches in IEDB SARS-CoV-2 database and print match.
    savefig_title : None or str. If provided, save figure in savedir and using given title.
    put_title : bool. Whether to display the title.
    transparent : bool. Make background transparent (e.g. for saving the figure).
    '''
    if hasattr(tr_f, 'cdr3_a_aa') and hasattr(tr_f, 'cdr3_b_aa'):
        # Alpha beta TCR 
        cdr3_vj_aa = 'cdr3_a_aa'
        cdr3_vdj_aa = 'cdr3_b_aa'
    elif hasattr(tr_f, 'cdr3_g_aa') and hasattr(tr_f, 'cdr3_d_aa'):
        # Gamma delta TCR 
        cdr3_vj_aa = 'cdr3_g_aa'
        cdr3_vdj_aa = 'cdr3_d_aa'
        specificity = False # not currently implemented
    else:
        raise
    n_shared, n_clones = tr_f.nunique()[['subject','clone_id']].values
    if skip_singletons:
        if n_clones == 1:
            return
    fig, ax = plt.subplots(ncols=2, figsize=(10,1))
    if specificity:
        try:
            betas
        except:
            # TODO : Change hard-coded file path
            antigens = pd.read_csv('/nfs/team205/ld21/public/antigen/IEDB_sars_cov_2.csv', low_memory=False)
            antigens = antigens.loc[:,antigens.isna().sum(axis=0)<10000]
            betas = antigens[['Antigen','Chain 2 CDR3 Curated']]
        antigen_spec = betas[betas['Chain 2 CDR3 Curated'].isin(tr_f.cdr3_b_aa)]['Antigen'].unique()
        if antigen_spec.size > 0:
            antigen_spec = antigen_spec[0].split('[')[0]
            title = f'Shared by: {n_shared},  Unique clones: {n_clones},  Antigen: {antigen_spec}'
        else:
            title =  f'Shared by: {n_shared},  Unique clones: {n_clones}'
    else:
        title =  f'Shared by: {n_shared},  Unique clones: {n_clones}'
    for chain_ind, chain in enumerate([cdr3_vj_aa,cdr3_vdj_aa]):
        cdr3 = tr_f[[cdr3_vj_aa,cdr3_vdj_aa]].copy()
        cdr3['length'] = cdr3[chain].apply(lambda x: len(x))
        n_rows = cdr3.length.median().astype(int)
        cdr3 = cdr3[cdr3.length==n_rows]

        # AA frequencies for logo
        letters = np.unique(pd.DataFrame([list(x) for x in cdr3[chain]])).tolist()
        logo = pd.DataFrame(np.zeros((n_rows, len(letters))), columns=letters)
        for row in range(n_rows):
            try:
                vals = pd.DataFrame([list(x) for x in cdr3[chain]]).value_counts([row], normalize=True).reset_index()
                logo.loc[row, vals[row]] = vals[0].values
            except:
                vals = pd.DataFrame([list(x) for x in cdr3[chain]]).value_counts([0], normalize=True)
                logo.loc[row, vals.index[0]] = vals.values

        # plot logo
        logo_plt = logomaker.Logo(logo, color_scheme='chemistry', ax=ax[chain_ind])
        logo_plt.ax.grid(False)
        logo_plt.ax.axis(False)
    if put_title:
        plt.suptitle(title, y=1.2, x=0.6)
    if savefig_title is not None:
        if transparent:
            plt.savefig(savefig_title, transparent=True)
        else:
            plt.savefig(savefig_title, transparent=False)
    plt.show()