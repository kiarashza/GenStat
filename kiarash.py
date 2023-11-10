import Synthatic_graph_generator
import stat_rnn
graph = Synthatic_graph_generator.grid()
graph = [graph]
mmd_degree = stat_rnn.degree_stats(graph, graph)
mmd_clustering = stat_rnn.clustering_stats(graph, graph)
try:
    mmd_4orbits = stat_rnn.orbit_stats_all(graph, graph)
except:
    mmd_4orbits = -1