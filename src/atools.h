#include <string>
#include <iostream>

namespace FastAlign {

struct AtoolsOpt {
  std::string input_1;
  std::string input_2;
  std::string command;
};

bool InitCommandLine(AtoolsOpt &opt, int argc, char** argv);

void AtoolsUsage(std::ostream &o = std::cerr, char const* exename="atools");

/// \return nonzero on error - old fast_align.cc main(...)
int Run(AtoolsOpt const& opt);

}  // namespace FastAlign
