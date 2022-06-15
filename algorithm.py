from pm4py.visualization.petri_net import visualizer as pn_visualizer
import copy
import numpy as np
from pm4py.algo.filtering.log.attributes import attributes_filter
from utils import *
from pm4py.objects.log.importer.xes import importer as xes_importer

def apply(log, order_type='bfs', theta=0.95, c=0.9, use_recall=False, 
            use_heuristics=True, visualize=False, return_intermediate=False, 
            debug=False, use_new_rule=True, exp=False, return_net_dict=False):
    # get the order
    if order_type == 'bfs':
        uniq_a_sorted = get_bf_order(log, exp=exp)
    if order_type == 'bfs_start':
        uniq_a_sorted = get_bf_order(log, from_start=True, from_end=False, exp=exp)
    elif order_type == 'bfs_start_end':
        uniq_a_sorted = get_bf_order(log=log, from_start=True, exp=exp)
    elif order_type == 'freq':
        uniq_a_sorted = get_freq_order(log, exp=exp)

    activity_name_log = [[a['concept:name'] for a in trace] for trace in log]

    if return_intermediate:
        the_chosen_nets = []

    for i, activity in enumerate(uniq_a_sorted):
        new_t_name = activity

        # projected activities
        projected_a = uniq_a_sorted[:i+1]

        projected_log = attributes_filter.apply_events(log, projected_a,
                                parameters={attributes_filter.Parameters.ATTRIBUTE_KEY: "concept:name", 
                                            attributes_filter.Parameters.POSITIVE: True})
        
        if i == 0:
            optional, loop = check_if_optional_loop(activity_name=new_t_name, activity_name_log=activity_name_log)
            net, im, fm = initialize_net(first_t_label=new_t_name, optional=optional, loop=loop)
            
            if visualize:
                gviz = pn_visualizer.apply(net, im, fm)
                pn_visualizer.view(gviz)

            if return_intermediate:
                m = incidence_matrix.construct(net)
                mat = np.array(m.a_matrix)
                trans_dict = {k.name: int(v) for k, v in m.transitions.items()}
                places_dict = {k.name: int(v) for k, v in m.places.items()}

                init_net_dict = {
                    'incidence_mat': mat, 
                    'trans_dict': trans_dict, 
                    'places_dict': places_dict,
                    'im': im, 
                    'fm': fm,
                    'petri': net,
                    'break_sc_petri': remove_tran_by_name(net, trans_name='short_circuit')
                }
                the_chosen_nets.append(init_net_dict)

        else:
            the_chosen_net = get_the_best_net(net=net, new_t_name=new_t_name, log=projected_log, 
                                            visualize=debug, use_heuristics=use_heuristics, t_Rs=c, 
                                            use_recall=use_recall, theta=theta, noise_threshold=0,use_new_rule=use_new_rule)
            the_chosen_net = remove_redundant_taus(the_chosen_net, final=False, use_new_rule=False)
            the_chosen_net = remove_redundant_taus(the_chosen_net, final=False, use_new_rule=True)
            net = copy.deepcopy(the_chosen_net['petri'])
            
            if return_intermediate:
                the_chosen_nets.append(the_chosen_net)
            
            if visualize:
                # for visualizatoin
                print('##########################################################################')
                print('the chosen one for the next')
                gviz = pn_visualizer.apply(the_chosen_net['break_sc_petri'], the_chosen_net['im'], the_chosen_net['fm'])
                pn_visualizer.view(gviz)
                print('recall:', the_chosen_net['recall'], 'precision', the_chosen_net['precision'], 'F1', the_chosen_net['F1'])
                print('fitness:', the_chosen_net['fitness_dict'])
            

    the_chosen_net = remove_redundant_taus(the_chosen_net, final=True, use_new_rule=False)
    the_chosen_net = remove_redundant_taus(the_chosen_net, final=True, use_new_rule=True)

    # if return_intermediate:
    #     the_chosen_nets.append(the_chosen_net)

    if visualize:
        gviz = pn_visualizer.apply(the_chosen_net['break_sc_petri'], the_chosen_net['im'], the_chosen_net['fm'])
        pn_visualizer.view(gviz)
        print('recall:', the_chosen_net['recall'], 'precision', the_chosen_net['precision'], 'F1', the_chosen_net['F1'])
        print('fitness:', the_chosen_net['fitness_dict'])

    if return_intermediate:
        return the_chosen_net, the_chosen_nets
    elif return_net_dict:
        return the_chosen_net
    else:
        return the_chosen_net['break_sc_petri'], the_chosen_net['im'], the_chosen_net['fm']