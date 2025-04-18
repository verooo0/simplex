set Intersections;
set Forbidden within Intersections;
set Required within Intersections;
set Paths;
param MaxSensors;
param Flow{Paths} >= 0;
set PathIntersections dimen 2;
set Neighborhoods within {Intersections, Intersections};

var x{Intersections} binary;
var y{Paths} binary;

maximize TotalFlow:
    sum{p in Paths} Flow[p] * y[p];

subject to SensorLimit:
    sum{i in Intersections} x[i] <= MaxSensors;

subject to ProhibitedSensors{i in Forbidden}:
    x[i] = 0;

subject to RequiredSensors{i in Required}:
    x[i] = 1;

subject to PathCovered{p in Paths}:
    sum{(pp, i) in PathIntersections: pp = p} x[i] >= 2 * y[p];
    
    
subject to NoCloseSensors{i in Intersections, j in Intersections: (i,j) in Neighborhoods and i < j}:
    x[i] + x[j] <= 1;