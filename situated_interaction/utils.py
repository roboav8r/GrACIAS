#!/usr/bin/env python3

def pmf_to_spec(pmf):

    spec = ''
    for row_idx in range(pmf.shape[0]):
        row = pmf[row_idx,:]
        row_spec = ''
        
        for col_idx in range(len(row)):
            if col_idx == 0: # If empty spec
                row_spec += str(row[col_idx])
            else:
                row_spec += '/' +  str(row[col_idx]) 
        
        if row_idx==0:
            spec += row_spec
        else:
            spec += ' ' + row_spec
        
    return spec