import os
import utils as u

current_path = os.path.dirname(os.path.realpath(__file__))

folderpath = current_path + '/FlowerClassification/FlowerIRs/'
fingerprints = current_path + '/EchoFingerprints/'
conv_spectrograms = current_path + '/ConvolvedSpectrograms/'

# call_datapath = current_path + '/FlowerClassification/EcholocationCalls'
call_datapath = current_path + '/Glossophaga_soricina'

for filename in os.listdir(call_datapath):
    print(call_datapath + '/' + filename)
    call = u.return_bat_calls(
        call_datapath + '/' + filename, filename, plot=True, split=True)
    break

for dirpath, dirnames, filenames in os.walk(folderpath):
    for filename in [f for f in filenames if f.endswith(".csv")]:
        print('Getting fingerprint for:', filename)
        data = u.load_data(dirpath + '/' + filename, show_head=False)
        # u.retrieve_fingerprint(data, call2, fingerprints + filename)

        a = filename.split()
        concatname = a[0] + '_' + a[1] + '_' + a[2]
        u.get_convolved_call(data, call, conv_spectrograms + concatname)
        break
