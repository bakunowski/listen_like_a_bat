import os, os.path
import utils as u

current_path = os.path.dirname(os.path.realpath(__file__))

datapath = current_path + '/FlowerClassification/FlowerIRs/Pguian_h_impuls_bin/Pguian h I 20cm_2009-06-03_NA_2019-08-20_impuls.csv'

folderpath = current_path + '/FlowerClassification/FlowerIRs/'
fingerprints = current_path + '/EchoFingerprints/'
call_datapath = current_path + '/FlowerClassification/EcholocationCalls/Glosso_call.wav'

for dirpath, dirnames, filenames in os.walk(folderpath):
    for filename in [f for f in filenames if f.endswith(".csv")]:
        print('Getting fingerprint for:', filename)
        data = u.load_data(dirpath + '/' + filename, show_head=False)
        call, _, _ = u.return_bat_calls(call_datapath, plot=False)
        u.retrieve_fingerprint(data, call, fingerprints + filename)
        break