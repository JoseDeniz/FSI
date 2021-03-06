# Search methods

import search

ab = search.GPSProblem('O', 'N', search.romania)


print "Breadth first %s" % search.breadth_first_graph_search(ab).path()
# print "Depth First %s " % search.depth_first_graph_search(ab).path()
# print search.iterative_deepening_search(ab).path()
# print search.depth_limited_search(ab).path()
print "Branch and Bound: %s" % search.branch_and_bound(ab).path()
print "Branch and Bound with heuristic: %s" % search.branch_and_bound_with_heuristic(ab).path()

#print search.astar_search(ab).path()

# Result:
# [<Node B>, <Node P>, <Node R>, <Node S>, <Node A>] : 101 + 97 + 80 + 140 = 418
# [<Node B>, <Node F>, <Node S>, <Node A>] : 211 + 99 + 140 = 450
