#include "ttables.h"

#include <cmath>
#include <fstream>
#include <string>

#include "corpus.h"

namespace FastAlign {

unsigned TTable::DeserializeLogProbsFromText(std::istream* in, Dict& d, std::ostream* log) {
  unsigned c = 0;
  std::string e, f;
  double p;
  if (!in) return 0;
  std::istream& input = *in;
  while (input) {
    if (!(input >> e)) break;
    if (!(input >> f)) {
      if (log) *log << "ERROR after " << c << " probs: no f read after e: " << e;
      return 0;
    }
    if (!(input >> p)) {
      if (log)
        *log << "ERROR after " << c << " probs: no p (floating point) read after e: " << e << " f: " << f;
      return 0;
    }
    ++c;
    unsigned ie = d.Convert(e);
    if (ie >= ttable.size()) ttable.resize(ie + 1);
    ttable[ie][d.Convert(f)] = std::exp(p);
  }
  if (log) *log << "Loaded " << c << " translation parameters.\n";
  probs_initialized_ = true;
  return c;
}

}  // namespace FastAlign
