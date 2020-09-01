import csv
import os

current_path = os.path.dirname(os.path.realpath(__file__))

folderpath = current_path + '/Data/Train_Test_Validation/Train/Interval10/'


with open('listen_like_a_bat10Train_Interval10_valset.csv', 'w') as f:
    writer = csv.writer(f)
    for dirpath, dirnames, filenames in os.walk(folderpath):
        for filename in [f for f in filenames]:
            # relative_path = os.path.relpath(dirpath, current_path)
            print(filename)
            a = filename.split()
            a = a[0][0:2]
            entry = filename
            if a == 'Bt':
                writer.writerow([entry, 0])
            elif a == 'Cc':
                writer.writerow([entry, 1])
            elif a == 'Cj':
                writer.writerow([entry, 2])
            elif a == 'Cq':
                writer.writerow([entry, 3])
            elif a == 'Cr':
                writer.writerow([entry, 4])
            elif a == 'Cs':
                writer.writerow([entry, 5])
            elif a == 'Mm':
                writer.writerow([entry, 6])
            elif a == 'Mn':
                writer.writerow([entry, 7])
            elif a == 'Pg':
                writer.writerow([entry, 8])
            elif a == 'Pp':
                writer.writerow([entry, 9])
            elif a == 'Sc':
                writer.writerow([entry, 10])
            elif a == 'Ws':
                writer.writerow([entry, 11])
