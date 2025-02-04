#ifndef FAST_ALIGN__FAST_ALIGN_H_
#define FAST_ALIGN__FAST_ALIGN_H_
#pragma once

#include <iostream>
#include <string>

#include "src/corpus.h"
#include "src/da.h"
#include "src/ttables.h"

namespace FastAlign {

/// settings output by training and used in aligning
struct AlignModelOpt {
  std::string conditional_probability_filename;
  int favor_diagonal = 1;  // rec. 1
  double mean_srclen_multiplier = 1.0;  // set in training
  double prob_align_null = 0.08;  // hyperparam
  double diagonal_tension = 4.0;  // modified in training if optimize_tension
  int no_null_word = 0;
  int is_reverse = 0;
};

struct LogOpt {
  /// may be null:
  std::ostream* log = &std::cerr;
  std::ostream* warn = &std::cerr;
};

struct TrainModelOpt : AlignModelOpt, LogOpt {
  Dict d;  // integerization map. TODO shared_ptr? or separate (context)?

  std::string input;
  std::string conditional_probability_filename;
  int ITERATIONS = 5;
  int favor_diagonal = 0;  // rec. 1
  double beam_threshold = -4.0;  // limits written cond prob table size
  int optimize_tension = 0;  // rec. 1
  int variational_bayes = 0;  // rec. 1
  double alpha = 0.01;  // hyperparam if variational_bayes
  size_t thread_buffer_size = 10000;
  bool force_align = false;
  int print_scores = 0;
  int off_by_1 = 0;

  void CommonDefaults() {
    print_scores = 0;
    force_align = false;
    no_null_word = 0;
    thread_buffer_size = 10000;
    ITERATIONS = 5;
    is_reverse = 0;
    mean_srclen_multiplier = 1.0;
    diagonal_tension = 4.0;
    beam_threshold = -4.0;
    prob_align_null = 0.08;
    alpha = 0.01;
  }

  void LegacyDefaults() {
    CommonDefaults();
    favor_diagonal = 0;
    variational_bayes = 0;
    optimize_tension = 0;
    // off_by_1 = 1; // needs to be explicitly requested for now (single switch arg taking no value)
  }

  /// not used by int main (for cmdline option backward compat)
  void RecommendedDefaults() {
    CommonDefaults();
    favor_diagonal = 1;
    variational_bayes = 1;
    optimize_tension = 1;
    off_by_1 = 0;
    mean_srclen_multiplier = 1.0;

    /// TBD (impacted by !off_by_1):
    alpha = 0.01;
    beam_threshold = -5.0;  // TBD
    prob_align_null = 0.05;  // TBD
  }
};

bool InitCommandLine(TrainModelOpt& opt, int argc, char** argv);
void UpdateFromPairs(TrainModelOpt& opt, const std::vector<std::string>& lines, int lc, int iter,
                     bool final_iteration, bool use_null, unsigned kNULL, double prob_align_not_null, double* c0,
                     double* emp_feat, double* likelihood, TTable* s2t, std::vector<std::string>* outputs);
inline void AddTranslationOptions(std::vector<std::vector<unsigned>>& insert_buffer, TTable* s2t);
typedef std::pair<std::pair<short, short>, unsigned> SizeCount;
typedef std::vector<SizeCount> SizeCounts;

struct AlignContext : AlignModelOpt, LogOpt {
  void Init(AlignModelOpt const& opt, Dict& d);
  /// \return 0 on success
  int LoadProbsText();
  Dict* d = nullptr;
  unsigned kNULL;
  bool use_null;
  double prob_align_not_null;
  TTable s2t;
};

struct TrainContext : AlignContext {
  void Init(TrainModelOpt& opt);
  double tot_len_ratio = 0;
  double n_target_tokens = 0;
  SizeCounts size_counts;
  void SaveProbsText();
};

void InitialPass(TrainModelOpt& opt, unsigned kNULL, bool use_null, TTable* s2t, double* n_target_tokens,
                 double* tot_len_ratio, SizeCounts* size_counts);

void InitialPass(TrainContext& ctx, TrainModelOpt& opt);

/// \return nonzero on error
int TrainIterations(TrainContext& ctx, TrainModelOpt& opt, int ITERATIONS);

/// \return nonzero on error
int TrainIteration(TrainContext& ctx, TrainModelOpt& opt, int iter, int ITERATIONS);

/// sets opt.diagonal_tension if favor_diagonal and optimize_tension
void TrainIteration(TrainContext& ctx, TrainModelOpt& opt, std::istream& input, int iter, int ITERATIONS);

void InitialPass(TrainContext& ctx, TrainModelOpt& opt);

/// \return nonzero on error. calls InitialPass first, runs opt.ITERATIONS iters. (requires ctx.Init(opt))
int Train(TrainContext& ctx, TrainModelOpt& opt);

/// uses input opt.input ("-" means cin). \return total logprob (or HUGE_VAL if error)
double Align(AlignContext const& ctx, std::ostream& out, std::string const& input = "-");

double AlignLine(AlignContext const& ctx, std::string const& line, int lineno, std::ostream& out);

double LenLogProb(AlignContext const& ctx, unsigned nsrc, unsigned ntrg);

/// \post *srcPlus1 = 1 + a_j where a_j < nsrc (or 0 if null aligned), returning probability of trg[j]. \pre j < ntrg
double AlignedProb(AlignContext const& ctx, unsigned const* src, unsigned nsrc, unsigned const* trg,
                   unsigned ntrg, unsigned j, unsigned* srcPlus1);


/// \param[out] array unsigned srcPlus1ForTrg[ntrg]. gets 0 for null-aligned or srcindex=1 if aligned. \return total logprob (p(trg|src)
inline double AlignedLogProb(AlignContext const& ctx, unsigned const* src, unsigned nsrc, unsigned const* trg,
                             unsigned ntrg, unsigned* srcPlus1ForTrg) {
  double log_prob = LenLogProb(ctx, nsrc, ntrg);
  for (unsigned j = 0; j < ntrg; ++j)
    log_prob += log(AlignedProb(ctx, src, nsrc, trg, ntrg, j, srcPlus1ForTrg++));
  return log_prob;
}


/// print in-sentence pairs ||| alignments to out. \return total logprob (or HUGE_VAL if error)
double Align(AlignContext const& ctx, std::istream& in, std::ostream& out);

/// \return nonzero on error - old fast_align.cc main(...)
int Run(TrainModelOpt& opt);

}  // namespace FastAlign

#endif
