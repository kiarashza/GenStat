import subprocess

# using Popen may suit better here depending on how you want to deal
# with the output of the child_script.

kernl_types = ["TotalNumberOfTriangles", "HistogramOfRandomWalks", "out_degree_dist", "ReachabilityInKsteps","HistogramOfCycles", "NumberOfVer"]
datasets = [ "lobster"]
dir = "DesFunSelection_all_IMDB/"

for dataset in datasets:
    dir_to_write_Graphs = dir+dataset+"/"+"all/"
    args = ["python", "permutationInvaeinentVAE.py", "-e","20000","-limited_to", "-1","-desc_fun"]
    # args.extend(kernl_types)
    subprocess.call(args+kernl_types+["-write_them_in",dir_to_write_Graphs]+["-dataset",dataset])

    for k,kernel in enumerate(kernl_types):
        dir_to_write_Graphs = dir+dataset+"/"+kernel+"/"
        the_kernels = kernl_types[:k] + kernl_types[k+1:]
        print(dir_to_write_Graphs)
        subprocess.call(args+the_kernels+["-write_them_in",dir_to_write_Graphs]+["-dataset",dataset])




