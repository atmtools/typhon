#!/bin/bash

DEBUG=echo
[[ $1 = "-r" ]] && DEBUG=

$DEBUG anaconda login
$DEBUG anaconda upload -u rttools build/*/typhon-*.bz2

