import algorithm as synthesisRulesMiner
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.visualization.petri_net import visualizer as pn_visualizer

log = xes_importer.apply('./data/demo.xes')
net, im, fm = synthesisRulesMiner.apply(log, order_type='bfs', theta=0.95, 
                                        c=0.9, use_new_rule=False)
gviz = pn_visualizer.apply(net, im, fm)
pn_visualizer.view(gviz)