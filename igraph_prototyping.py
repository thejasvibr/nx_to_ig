#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 15:35:54 2022

@author: thejasvi
"""

import igraph as ig
from itertools import product, combinations
import networkx as nx
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/thejasvi/Documents/research_repos/pydatemm/ky2013')
import build_ccg as bccg
from build_ccg import make_fundamental_loops, make_triple_pairs
from pydatemm.timediffestim import generate_multich_crosscorr, get_multich_tdoas
import tqdm
import numpy as np
import soundfile as sf
import time 
ns_time = time.perf_counter_ns

def ig_make_fundamental_loops(nchannels):
    G = ig.Graph.Full(nchannels)
    G.vs['node'] = range(nchannels)
    minspan_G = G.spanning_tree()
    main_node = 0
    co_tree = minspan_G.complementer().simplify()
    fundamental_loops = []
    for edge in co_tree.es:
        fl_nodes = tuple((main_node, edge.source, edge.target))
        fundamental_loops.append(fl_nodes)
    return fundamental_loops

def ig_make_edges_for_fundamental_loops(nchannels):
    funda_loops = ig_make_fundamental_loops(nchannels)
    triple_definition = {}
    for fun_loop in funda_loops:
        edges = make_triple_pairs(fun_loop)
        triple_definition[fun_loop] = []
        # if the edge (ab) is present but the 'other' way round (ba) - then 
        # reverse polarity. 
        for edge in edges:
            triple_definition[fun_loop].append(edge)
    return triple_definition

def ig_make_consistent_fls(multich_tdes, **kwargs):
    max_loop_residual = kwargs.get('max_loop_residual', 1e-6)
    all_edges_fls = ig_make_edges_for_fundamental_loops(kwargs['nchannels'])
    all_cfls = []

    for fundaloop, edges in all_edges_fls.items():
        a,b,c = fundaloop
        ba_tdes = multich_tdes[(b,a)]
        ca_tdes = multich_tdes[(c,a)]
        cb_tdes = multich_tdes[(c,b)]
        abc_combinations = list(product(ba_tdes, ca_tdes, cb_tdes))
        node_to_index = {nodeid: index for index, nodeid in  zip(range(3), fundaloop)}
        for i, (tde1, tde2, tde3) in enumerate(abc_combinations):
            if abs(tde1[1]-tde2[1]+tde3[1]) < max_loop_residual:
                this_cfl = ig.Graph(3, directed=True)
                this_cfl.vs['node'] = fundaloop
                for e, tde in zip(edges, [tde1, tde2, tde3]):
                    this_cfl.add_edge(node_to_index[e[0]], node_to_index[e[1]],
                                      tde=tde[1])
                all_cfls.append(this_cfl)
    return all_cfls

audio, fs = sf.read('3-bats_trajectory_simulation_0-order-reflections.wav')
start, end = np.int64(fs*np.array([0.009, 0.017]))
sim_audio = audio[start:end,:]
nchannels = audio.shape[1]
kwargs = {'nchannels':nchannels,
          'fs':fs,
          'pctile_thresh': 95,
          'use_gcc':True,
          'gcc_variant':'phat', 
          'min_peak_diff':0.35e-4, 
          'vsound' : 343.0, 
          'no_neg':False}
kwargs['max_loop_residual'] = 0.5e-4
kwargs['K'] = 7

multich_cc = generate_multich_crosscorr(sim_audio, **kwargs )
cc_peaks = get_multich_tdoas(multich_cc, **kwargs)

K = kwargs.get('K',5)
top_K_tdes = {}
for ch_pair, tdes in cc_peaks.items():
    descending_quality = sorted(tdes, key=lambda X: X[-1], reverse=True)
    top_K_tdes[ch_pair] = []
    for i in range(K):
        try:
            top_K_tdes[ch_pair].append(descending_quality[i])
        except:
            pass
print('making the cfls...')

def node_names(ind_tup,X):
    node_names = X.vs['node']
    return tuple(node_names[i] for i in ind_tup)
    
    
def ig_check_for_one_common_edge(X,Y):

    X_edge_weights = [ (node_names(i.tuple, X), i['tde']) for i in X.es]
    Y_edge_weights = [ (node_names(i.tuple, Y), i['tde']) for i in Y.es]
    common_edge = set(Y_edge_weights).intersection(set(X_edge_weights))
    if len(common_edge)==1:
        return 1
    else:
        return -1


def ig_ccg_definer(X,Y):
    common_nodes = set(X.vs['node']).intersection(set(Y.vs['node']))
    if len(common_nodes) >= 2:
        if len(common_nodes) < 3:
            relation = ig_check_for_one_common_edge(X, Y)
        else:
            # all nodes the same
            relation = -1
    else:
        relation = -1
    return relation


def ig_make_ccg_matrix(cfls):
    '''
    Sped up version. Previous version had explicit assignment of i,j and j,i
    compatibilities.
    '''
        
    num_cfls = len(cfls)
    ccg = np.zeros((num_cfls, num_cfls), dtype='int32')
    cfl_ij = combinations(range(num_cfls), 2)
    for (i,j) in cfl_ij:
        trip1, trip2  = cfls[i], cfls[j]
        cc_out = ig_ccg_definer(trip1, trip2)
        ccg[i,j] = cc_out
    ccg += ccg.T
    return ccg


if __name__ == "__main__":
   
    nruns = 100
    sta = ns_time()/1e9
    for i in tqdm.trange(nruns):
        ig_make_edges_for_fundamental_loops(i)
    print(f'{ns_time()/1e9 - sta} s')

    sta = ns_time()/1e9
    for i in tqdm.trange(nruns):
        bccg.make_edges_for_fundamental_loops(i)
    print(f'\n{ns_time()/1e9 - sta} s')
    print('\n \n')
    a = ns_time(); a/=1e9
    cfls_from_tdes = bccg.make_consistent_fls(top_K_tdes, **kwargs)
    print(f'{ns_time()/1e9 - a} s')
    a = ns_time(); a/=1e9
    ig_cfls_from_tdes = ig_make_consistent_fls(top_K_tdes, **kwargs)
    print(f'{ns_time()/1e9 - a} s')

    #%%
    sta = ns_time()/1e9
    nx_ccg = bccg.make_ccg_matrix(cfls_from_tdes)
    print(f'NX: {ns_time()/1e9 - sta} s'); sta = ns_time()/1e9
    ig_ccg = ig_make_ccg_matrix(ig_cfls_from_tdes)
    print(f'IG: {ns_time()/1e9 - sta} s')
    try:
        assert np.all(ig_ccg==nx_ccg)
    except:
        noteq = np.argwhere(ig_ccg!=nx_ccg)
        X, Y = [ig_cfls_from_tdes[i] for i in noteq[0,:]]
        X, Y = [cfls_from_tdes[i] for i in noteq[0,:]]

    #%% 
    # Comparing nx and ig outputs 
    for nx_out, ig_out  in zip(cfls_from_tdes, ig_cfls_from_tdes):
        nx_tde = nx.adjacency_matrix(nx_out, weight='tde').todense()
        ig_tde = np.array(ig_out.get_adjacency(attribute='tde').data)
        
        nx_weights = nx.get_edge_attributes(nx_out,'tde')
        for edge in ig_out.es:
            b,a = edge.tuple
            node_b, node_a = ig_out.vs['node'][b], ig_out.vs['node'][a]
            try:
                assert edge['tde'] == nx_weights[(node_b, node_a)]
            except:
                assert edge['tde'] == nx_weights[(node_a, node_b)]

        
        
    #%%
    g = ig_cfls_from_tdes[-1]
    plt.figure()
    a0 = plt.subplot(111)
    vis_style = {}
    vis_style['target'] = a0
    vis_style["edge_label"] = np.around(np.array(g.es["tde"])*1e3,2)
    vis_style["vertex_label"] = g.vs['node']
    
    ig.plot(g,   **vis_style)
    
    

