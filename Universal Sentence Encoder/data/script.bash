
preprocess_exec="sed -f tokenizer.sed"

for split in train dev test
do
    fpath=snli/$split.snli.txt
    awk '{ if ( $1 != "-" ) { print $0; } }' snli/snli_1.0_$split.txt | cut -f 1,6,7 | sed '1d' > $fpath
    s1=$(cut -f2 $fpath | $preprocess_exec)
    s2=$(cut -f3 $fpath | $preprocess_exec)
    label=$(cut -f1 $fpath)
    paste -d '\t' <(echo "$s1") <(echo "$s2") <(echo "$label") > snli/$split.txt
    rm $fpath
done

for split in train dev test
do
    fpath=quora/$split.quora.txt
    awk '{ if ( $1 != "-" ) { print $0; } }' quora/$split.tsv | cut -f 1,2,3 > $fpath
    s1=$(cut -f2 $fpath | $preprocess_exec)
    s2=$(cut -f3 $fpath | $preprocess_exec)
    label=$(cut -f1 $fpath)
    paste -d '\t' <(echo "$s1") <(echo "$s2") <(echo "$label") > quora/$split.txt
    rm $fpath
done

for split in train dev_matched dev_mismatched test_matched test_mismatched
do
    fpath=multinli_1.0/$split.multinli.txt
    awk '{ if ( $1 != "-" ) { print $0; } }' multinli_1.0/multinli_1.0_$split.txt | cut -f 1,6,7,9,10 | sed '1d' > $fpath
    s1=$(cut -f2 $fpath | $preprocess_exec)
    s2=$(cut -f3 $fpath | $preprocess_exec)
    label=$(cut -f1 $fpath)
    pid=$(cut -f4 $fpath)
    genre=$(cut -f5 $fpath)
    paste -d '\t' <(echo "$s1") <(echo "$s2") <(echo "$label") <(echo "$pid") <(echo "$genre") > multinli/$split.txt
    rm $fpath
done