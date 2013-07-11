# The goal was to fetch 4096 seconds starting 32 seconds after the S5
# blind injection detailed at
# http://www.ligo-wa.caltech.edu/~michael.landry/s5blindinj/inspiral/
# and recolor it to the AdLIGO PSD.

# Here are the commands I used to fetch the data:

# On CIT, looking at
# https://wiki.ligo.org/Help/AGuideToFindingGravitationalWaveFrameData
# and
# https://dcc.ligo.org/DocDB/0096/T1200432/003/GuideToFindingGravitationalWaveData.pdf,
# I issued

START_TIME=873739840
INTERVAL=4096

ligo_data_find -o H -t H1_RDS_C03_L2 -s $START_TIME -e $((START_TIME + INTERVAL)) -u file  --lal-cache > LI-PSD-fit-H.cache
ligo_data_find -o L -t L1_RDS_C03_L2 -s $START_TIME -e $((START_TIME + INTERVAL)) -u file  --lal-cache > LI-PSD-fit-L.cache
ligo_data_find -o H -t RDS_R_L3 -s $START_TIME -e $((START_TIME + INTERVAL)) -u file  --lal-cache > LI-PSD-fit-detchar-H.cache
ligo_data_find -o L -t RDS_R_L3 -s $START_TIME -e $((START_TIME + INTERVAL)) -u file  --lal-cache > LI-PSD-fit-detchar-L.cache

# Then I copied all the *.cache files from my home folder on CIT to my
# laptop
gsiscp "ligo-cit:~/*.cache" .

# Then I copied the cache files to my local computer, and ran the
# following to fetch the actual data files

for file in *.cache; do 
    awk ' { print $NF } ' $file | sed -e 's|file://localhost||' >> ligo-cit-file-list.txt; 
done
rsync -e gsissh -avz --files-from=ligo-cit-file-list.txt --no-relative ligo-cit:/ ./ 

# And finally I duplicate the cache files locally:
ls H-H1*.gwf | lalapps_path2cache > local-H.cache
ls L-L1*.gwf | lalapps_path2cache > local-L.cache
