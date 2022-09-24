import algorithm as synthesisRulesMiner
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter

log = xes_importer.apply('./data/demo.xes')
net, im, fm = synthesisRulesMiner.apply(log, order_type='dfs', theta=0.95, 
                                        c=0.9, use_new_rule=False)

pnml_exporter.apply(net, im, "./demo.pnml", final_marking=fm)

# gviz = pn_visualizer.apply(net, im, fm)
# pn_visualizer.view(gviz)