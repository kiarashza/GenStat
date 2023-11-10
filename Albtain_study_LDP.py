import subprocess
import GPU_manager
import time
import string
import itertools

epochs= {"lobster":40000,"grid":40000,"triangular_grid":40000,"ogbg-molbbbp":10000,"MUTAG":40000,"DD":10000,"IMDBBINARY":40000,"PTC":40000}
defaults = {"in_normed_layer":True,"EvalOnTest":"True", "graphEmDim":"128", "scheduler_type":"OneCyle",} # ,"beta":"1","lr" : str(0.002),"dataset" : "lobster"
kernl_types =  [ "in_degree_dist", "HistogramOfCycles"]
gpus = [0,1]
threshold = 13000
# binningStrategy = ["EqualWidth","EqualFreq","BounderBinner"]
label = "_LDP_"

# settings = {"-dataset":[ "grid"], "-lr" : [0.001,0.0003],"-beta":[1], "-e":[40000],"-steps_of_reachability":["1","2","3","4"], "-arg_step_num":["1","2","3","4"],}
settings = {"-dataset":[ "grid", "MUTAG", "lobster","PTC", "IMDBBINARY", "ogbg-molbbbp","DD","ogbg-molbbbp", "triangular_grid"],"-epsilon" : [4,3,2,1,0.1,0.5], "-lr" : [0.0003],"-beta":[1], "-e":[40000], "-NumDecLayers":[4],"-decoder":["FC_R"]}

settings = {"-dataset":[ "triangular_grid",],"-epsilon" : [4,3,2,1,0.1,0.5], "-lr" : [0.0003],"-beta":[1], "-e":[10000], "-NumDecLayers":[4],"-decoder":["FC_R"]}

all_combinations = [dict(zip(settings.keys(), values)) for values in itertools.product(*settings.values())]

common_swichs = ["python", "GlobalPrespective.py", "-LDP", "-desc_fun"]+kernl_types
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

                avail = GPU_manager.get_free_gpu(threshold=threshold, targets = gpus, check=60, every=1)
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
                time.sleep(120)

















