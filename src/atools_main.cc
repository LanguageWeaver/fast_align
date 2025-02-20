#include "atools.h"

int main(int argc, char** argv) {
  FastAlign::AtoolsOpt opt;
  if (!InitCommandLine(opt, argc, argv)) {
    FastAlign::AtoolsUsage(std::cerr, argv[0]);
    return 1;
  }
  return Run(opt);
}
