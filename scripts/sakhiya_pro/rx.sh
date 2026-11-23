#!/bin/bash
# usage: rx.sh 'cmd'  -- run on MoLab box with utf8-safe output
python3 "$(dirname "$0")/../molab_exec.py" "( $1 ) 2>&1 | iconv -c -f utf8 -t ascii//TRANSLIT | tr -d '\000' | cut -c1-400"
