import subprocess
import GPU_manager
import time
import string
import itertools
# using Popen may suit better here depending on how you want to deal
# with the output of the child_script.

epochs= {"lobster":40000,"grid":40000,"triangular_grid":40000,"ogbg-molbbbp":40000,"MUTAG":40000,"DD":40000,"IMDBBINARY":10000,"PTC":40000}
defaults = {"in_normed_layer":True,"EvalOnTest":"True", "graphEmDim":"128", "scheduler_type":"OneCyle"} # ,"beta":"1","lr" : str(0.002),"dataset" : "lobster"
kernl_types =  [ "in_degree_dist", "HistogramOfCycles","ReachabilityInKsteps","HistogramOfRandomWalks","NumberOfVer"]
gpus = [0,1]
threshold = 22000
# binningStrategy = ["EqualWidth","EqualFreq","BounderBinner"]
label = "DD_HiddenLAyers_32_leaky"

# settings = {"-dataset":[ "grid"], "-lr" : [0.001,0.0003],"-beta":[1], "-e":[40000],"-steps_of_reachability":["1","2","3","4"], "-arg_step_num":["1","2","3","4"],}
# settings = {"-dataset":[ "grid","PTC","lobster","MUTAG"], "-lr" : [0.001,0.0003,sou0.0001],"-beta":[1,20,100], "-e":[40000],"-steps_of_reachability":["2"], "-arg_step_num":["3"], "-NumDecLayers":[4],"-decoder":["FC_R"]}

settings = {"-reach_act":["leaky"],"-dataset":[ "DD"], "-lr" : [0.0003 ],"-beta":[4,], "-e":[40000],"-steps_of_reachability":["3"], "-arg_step_num":["2"], "-NumDecLayers":[4],"-decoder":["FC_R"]}
settings = {"-reach_act":["leaky"],"-dataset":[  "grid","PTC","lobster","MUTAG","IMDBBINARY","triangular_grid"], "-lr" : [0.001,0.0003,0.0001],"-beta":[1,4,20], "-e":[40000],"-steps_of_reachability":["2"], "-arg_step_num":["3"], "-NumDecLayers":[4],"-decoder":["FC_R"]}

#settings = {"-dataset":[  "grid","PTC","lobster","MUTAG","IMDBBINARY","triangular_grid"], "-lr" : [0.0001,0.0003,0.001 ],"-beta":[4], "-e":[40000],"-steps_of_reachability":["2"], "-arg_step_num":["3"], "-NumDecLayers":[4],"-decoder":["FC_R"]}


all_combinations = [dict(zip(settings.keys(), values)) for values in itertools.product(*settings.values())]

common_swichs = ["python", "GlobalPrespective.py",  "-desc_fun"]+kernl_types
dir = label+"/"

for key in defaults:
    if key not in settings.keys():
        common_swichs.append("-" + key)
        common_swichs.append(str(defaults[key]))
exclude = set(string.punctuation)

for combination in all_combinations:
                this_setting = []
                this_run_dir = ""
                for key, val in combination.items():
                    this_run_dir+=key+str(val)+"_"
                    this_setting.append(key)
                    if type(val) == list:
                        this_setting.extend(val)
                    else:
                        this_setting.append(str(val))

                #path to write the result
                this_run_dir = ''.join(ch for ch in str(this_run_dir) if ch not in exclude)
                this_run_dir = this_run_dir.replace(" ","_")
                dir_to_write_Graphs = dir  + "/" + this_run_dir+"/"
                print("the result will be written in: "+dir_to_write_Graphs)

                avail = GPU_manager.get_free_gpu(threshold=threshold, targets = gpus, check=120, every=1)
                print("Avaiable GPUs were: "+str(avail))

                args_ = []
                args_ += common_swichs+["-write_them_in", dir_to_write_Graphs] + ["-device"] + [
                    "cuda:" + str(avail[0])] + this_setting


                # set the epoch number for the dataset

                if "-e" not in defaults and "-e" not in combination:
                    if "dataset" in defaults:
                        args_.append("-e")
                        args_.append(str(epochs[defaults["dataset"]]))
                    elif "dataset" in combination:
                        args_.append("-e")
                        args_.append(str(epochs[defaults["dataset"]]))

                # subprocess.call(args_)
                subprocess.Popen(args_)
                time.sleep(220)











