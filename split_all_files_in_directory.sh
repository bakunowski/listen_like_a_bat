#!/bin/bash

if [ $# -gt 0 ]; then
    dir=$1
else
    dir=.
fi

for file in "$dir"/*; do
    split -dl 3 --additional-suffix=.csv "$file" "$file"
    rm "$file"
done

# delete all files with more than one line of text
# for file in * ; do awk 'NR==3{exit}END{exit NR!=1}' "$file" && rm "$file"; done
