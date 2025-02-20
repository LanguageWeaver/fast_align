#include "src/fast_align.h"

int main(int argc, char** argv) {
  FastAlign::TrainModelOpt opt;
  opt.LegacyDefaults();
  if (!InitCommandLine(opt, argc, argv)) {
    std::cerr
        << "Usage: " << argv[0] << " -i file.fr-en\n"
        << " Standard options ([USE] = strongly recommended):\n"
        << "  -i: [REQ] Input parallel corpus\n"
        << "  -v: [USE] Use Dirichlet prior on lexical translation distributions\n"
        << "  -d: [USE] Favor alignment points close to the monotonic diagonoal\n"
        << "  -o: [USE] Optimize how close to the diagonal alignment points should be\n"
        << "  -r: Run alignment in reverse (condition on target and predict source)\n"
        << "  -c: Output conditional probability table\n"
        << " Advanced options:\n"
        << "  -I: number of iterations in EM training (default = 5)\n"
        << "  -q: p_null parameter (default = 0.08)\n"
        << "  -N: No null word\n"
        << "  -a: alpha parameter for optional Dirichlet prior (default = 0.01)\n"
        << "  -T: starting lambda for diagonal distance parameter (default = 4)\n"
        << "  -s: print alignment scores (alignment ||| score, disabled by default)\n"
        << "  -O: enable legacy off-by-1 indexing bug to keep same function for hyperparams as in published "
           "experiments\n"
        << "  -t: beam_threshold conditional logprob output beam width, <=0 (default = -4). "
           "-99999=unlimited\n";
    return 1;
  }
  return Run(opt);
}
