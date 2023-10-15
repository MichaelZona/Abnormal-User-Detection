import re
path = 'E:\GAL\Graph-Anomaly-Loss-master\configs\\file_paths.json'

file = open(path, "r")
content = file.read()
file.close()


t = content.replace("E:\GAL\Graph-Anomaly-Loss-master\src/data", "E:/GAL/Graph-Anomaly-Loss-master/src/data/")
with open(path, "w") as f2:
    f2.write(t)