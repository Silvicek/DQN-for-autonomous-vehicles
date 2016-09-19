from pypaths import astar

finder = astar.pathfinder()

print finder( (0,0), (2,2) )
