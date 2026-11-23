mkdir -p /work/sp/data/slr43 && cd /work/sp/data/slr43
U=https://www.openslr.org/resources/43/ne_np_female.zip
SZ=$(curl -sIL $U | grep -i content-length | tail -1 | tr -dc 0-9)
echo size $SZ
N=16; C=$(( (SZ+N-1)/N ))
for i in $(seq 0 $((N-1))); do s=$((i*C)); e=$((s+C-1)); [ $e -ge $SZ ] && e=$((SZ-1)); curl -s -L -r $s-$e -o part_$i $U & done
wait
cat $(for i in $(seq 0 $((N-1))); do echo part_$i; done) > ne.zip && rm part_*
unzip -q -o ne.zip && rm ne.zip
echo slr43 done >> /work/tts/logs/setup.log
