#ifndef _MT_H
#define _MT_H

//
// uint32 must be an unsigned integer type capable of holding at least 32
// bits; exactly 32 should be fastest, but 64 is better on an Alpha with
// GCC at -O3 optimization so try your options and see what's best for you
//

typedef unsigned long uint32;

void seedMT(uint32 seed);
uint32 reloadMT();
uint32 randomMT();
double frandomMT();

#endif /* _MT_H */
