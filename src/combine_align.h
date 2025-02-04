#ifndef FAST_ALIGN__COMBINE_ALIGN_H
#define FAST_ALIGN__COMBINE_ALIGN_H
#pragma once

#include "src/array2d.h"

namespace FastAlign {

enum CombineAlignment { kUnion, kIntersect, kGrowDiag, kGrowDiagFinal, kGrowDiagFinalAnd };

typedef Array2D<bool> AlignMatrix;

AlignMatrix combine(CombineAlignment combineAlignment, AlignMatrix const& a, AlignMatrix const& b);

void combine(CombineAlignment combineAlignment, AlignMatrix const& a, AlignMatrix const& b, AlignMatrix* r);

/// \return required length of aligned seequence for aplus1[na] whose values are 1+aligned index (or 0 if no align)
inline unsigned alignedLen(unsigned const* a, unsigned na) {
  unsigned m = 0;
  for (unsigned i = 0; i != na; ++i)
    if (a[i] > m) m = a[i];
  return m;
}

inline AlignMatrix alignMatrix(unsigned const* a, unsigned na, unsigned nb) {
  AlignMatrix m(na, nb);
  for (unsigned i = 0; i != na; ++i)
    if (a[i]) m(i, a[i]) = 1;
  return m;
}

inline AlignMatrix alignMatrixInvert(unsigned const* a, unsigned na, unsigned nb) {
  AlignMatrix m(nb, na);
  for (unsigned i = 0; i != nb; ++i)
    if (a[i]) m(a[i], i) = 1;
  return m;
}

/// \return na x nb matrix with i,a[i]-1 set (if a[i] > 0), inverted iff \param invert
inline AlignMatrix alignMatrix(unsigned const* a, unsigned na, unsigned nb, bool invert) {
  return invert ? alignMatrixInvert(a, na, nb) : alignMatrix(a, na, nb);
}

inline AlignMatrix combine(CombineAlignment c, unsigned const* a, unsigned na, unsigned const* b, unsigned nb) {
  return combine(c, alignMatrix(a, na, nb), alignMatrixInvert(b, nb, na));
}


}  // namespace FastAlign

#endif
