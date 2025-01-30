// Copyright 2013 by Chris Dyer
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>
#include <getopt.h>

#include "src/corpus.h"
#include "src/da.h"
#include "src/fast_align.h"
#include "src/ttables.h"


namespace FastAlign {
using namespace std;

struct PairHash {
  size_t operator()(const pair<short, short>& x) const {
    return (unsigned short)x.first << 16 | (unsigned)x.second;
  }
};

void ParseLine(Dict& d, const string& line, vector<unsigned>* src, vector<unsigned>* trg) {
  static const unsigned kDIV = d.Convert("|||");
  vector<unsigned> tmp;
  src->clear();
  trg->clear();
  d.ConvertWhitespaceDelimitedLine(line, kDIV, &tmp);
  unsigned i = 0;
  while (i < tmp.size() && tmp[i] != kDIV) {
    src->push_back(tmp[i]);
    ++i;
  }
  if (i < tmp.size() && tmp[i] == kDIV) {
    ++i;
    for (; i < tmp.size(); ++i) trg->push_back(tmp[i]);
  }
}


bool InitCommandLine(TrainModelOpt& opt, int argc, char** argv) {
  struct option options[] = {{"input", required_argument, 0, 'i'},
                             {"reverse", no_argument, &opt.is_reverse, 1},
                             {"iterations", required_argument, 0, 'I'},
                             {"favor_diagonal", no_argument, &opt.favor_diagonal, 1},
                             {"force_align", required_argument, 0, 'f'},
                             {"mean_srclen_multiplier", required_argument, 0, 'm'},
                             {"beam_threshold", required_argument, 0, 't'},
                             {"p0", required_argument, 0, 'q'},
                             {"diagonal_tension", required_argument, 0, 'T'},
                             {"optimize_tension", no_argument, &opt.optimize_tension, 1},
                             {"variational_bayes", no_argument, &opt.variational_bayes, 1},
                             {"alpha", required_argument, 0, 'a'},
                             {"no_null_word", no_argument, &opt.no_null_word, 1},
                             {"conditional_probabilities", required_argument, 0, 'p'},
                             {"thread_buffer_size", required_argument, 0, 'b'},
                             {"off_by_1", no_argument, &opt.off_by_1, 1},
                             {0, 0, 0, 0}};

  while (1) {
    int oi;
    int c = getopt_long(argc, argv, "i:rI:df:m:t:q:T:ova:Np:b:sO", options, &oi);
    if (c == -1) break;
    cerr << "ARG=" << (char)c << endl;
    switch (c) {
      case 'i': opt.input = optarg; break;
      case 'r': opt.is_reverse = 1; break;
      case 'I': opt.ITERATIONS = atoi(optarg); break;
      case 'd': opt.favor_diagonal = 1; break;
      case 'f':
        opt.force_align = 1;
        opt.conditional_probability_filename = optarg;
        break;
      case 'm': opt.mean_srclen_multiplier = atof(optarg); break;
      case 't': opt.beam_threshold = atof(optarg); break;
      case 'q': opt.prob_align_null = atof(optarg); break;
      case 'T':
        opt.favor_diagonal = 1;
        opt.diagonal_tension = atof(optarg);
        break;
      case 'o': opt.optimize_tension = 1; break;
      case 'v': opt.variational_bayes = 1; break;
      case 'a': opt.alpha = atof(optarg); break;
      case 'N': opt.no_null_word = 1; break;
      case 'p': opt.conditional_probability_filename = optarg; break;
      case 'b': opt.thread_buffer_size = atoi(optarg); break;
      case 's': opt.print_scores = 1; break;
      case 'O': opt.off_by_1 = 1; break;
      default: return false;
    }
  }
  if (opt.input.size() == 0) return false;
  return true;
}

void UpdateFromPairs(TrainModelOpt& opt, const vector<string>& lines, const int lc, const int iter,
                     const bool final_iteration, const bool use_null, const unsigned kNULL,
                     const double prob_align_not_null, double* c0, double* emp_feat, double* likelihood,
                     TTable* s2t, vector<string>* outputs) {
  if (final_iteration) {
    outputs->clear();
    outputs->resize(lines.size());
  }
  double emp_feat_ = 0.0;
  double c0_ = 0.0;
  double likelihood_ = 0.0;
  int off_by = 1 - opt.off_by_1;
#pragma omp parallel for schedule(dynamic) reduction(+ : emp_feat_, c0_, likelihood_)
  for (int line_idx = 0; line_idx < static_cast<int>(lines.size()); ++line_idx) {
    vector<unsigned> src, trg;
    ParseLine(opt.d, lines[line_idx], &src, &trg);
    if (opt.is_reverse) swap(src, trg);
    if (src.size() == 0 || trg.size() == 0) {
      if (opt.warn) *opt.warn << "Error in line " << lc << "\n" << lines[line_idx] << endl;
      // return 1;
    }
    ostringstream oss;  // collect output in last iteration
    vector<double> probs(src.size() + 1);
    bool first_al = true;  // used when printing alignments
    double local_likelihood = 0.0;
    for (unsigned j = 0; j < trg.size(); ++j) {
      const unsigned& f_j = trg[j];
      double sum = 0;
      double prob_a_i = 1.0 / (src.size() + use_null);  // uniform (model 1)
      if (use_null) {
        if (opt.favor_diagonal) prob_a_i = opt.prob_align_null;
        probs[0] = s2t->prob(kNULL, f_j) * prob_a_i;
        sum += probs[0];
      }
      double az = 0;
      if (opt.favor_diagonal)
        az = DiagonalAlignment::ComputeZ(j + 1, trg.size(), src.size(), opt.diagonal_tension) / prob_align_not_null;
      for (unsigned i = 1; i <= src.size(); ++i) {
        if (opt.favor_diagonal)
          prob_a_i
              = DiagonalAlignment::UnnormalizedProb(j + 1, i, trg.size(), src.size(), opt.diagonal_tension) / az;
        probs[i] = s2t->prob(src[i - 1], f_j) * prob_a_i;
        sum += probs[i];
      }
      if (final_iteration) {
        double max_p = -1;
        int max_index = -1;
        if (use_null) {
          max_index = 0;
          max_p = probs[0];
        }
        for (unsigned i = 1; i <= src.size(); ++i) {
          if (probs[i] > max_p) {
            max_index = i;
            max_p = probs[i];
          }
        }
        if (max_index > 0) {
          if (first_al)
            first_al = false;
          else
            oss << ' ';
          if (opt.is_reverse)
            oss << j << '-' << (max_index - 1);
          else
            oss << (max_index - 1) << '-' << j;
        }
      } else {
        if (use_null) {
          double count = probs[0] / sum;
          c0_ += count;
          s2t->Increment(kNULL, f_j, count);
        }
        for (unsigned i = 1; i <= src.size(); ++i) {
          const double p = probs[i] / sum;
          s2t->Increment(src[i - 1], f_j, p);
          emp_feat_ += DiagonalAlignment::Feature(j + off_by, i, trg.size(), src.size()) * p;
        }
      }
      local_likelihood += log(sum);
    }
    likelihood_ += local_likelihood;
    if (final_iteration) {
      if (opt.print_scores) {
        double log_prob = Md::log_poisson(trg.size(), 0.05 + src.size() * opt.mean_srclen_multiplier);
        log_prob += local_likelihood;
        oss << " ||| " << log_prob;
      }
      oss << endl;
      (*outputs)[line_idx] = oss.str();
    }
  }
  *emp_feat += emp_feat_;
  *c0 += c0_;
  *likelihood += likelihood_;
}

inline void AddTranslationOptions(vector<vector<unsigned>>& insert_buffer, TTable* s2t) {
  s2t->SetMaxE(insert_buffer.size() - 1);
#pragma omp parallel for schedule(dynamic)
  for (unsigned e = 0; e < insert_buffer.size(); ++e) {
    for (unsigned f : insert_buffer[e]) {
      s2t->Insert(e, f);
    }
    insert_buffer[e].clear();
  }
}

void InitialPass(TrainModelOpt& opt, const unsigned kNULL, const bool use_null, TTable* s2t,
                 double* n_target_tokens, double* tot_len_ratio, SizeCounts* size_counts) {
  ifstream in(opt.input.c_str());
  std::ostream* log = opt.log;
  std::ostream* warn = opt.warn;
  if (!in && warn) {
    *warn << "Can't read " << opt.input << endl;
  }
  unordered_map<pair<short, short>, unsigned, PairHash> size_counts_;
  vector<vector<unsigned>> insert_buffer;
  size_t insert_buffer_items = 0;
  vector<unsigned> src, trg;
  string line;
  bool flag = false;
  int lc = 0;
  *log << "INITIAL PASS " << endl;
  while (true) {
    getline(in, line);
    if (!in) break;
    lc++;
    if (log && lc % 1000 == 0) {
      *log << '.';
      flag = true;
    }
    if (log && lc % 50000 == 0) {
      *log << " [" << lc << "]\n" << flush;
      flag = false;
    }
    ParseLine(opt.d, line, &src, &trg);
    if (opt.is_reverse) swap(src, trg);
    if (src.size() == 0 || trg.size() == 0) {
      *warn << "Error in line " << lc << "\n" << line << endl;
    }
    *tot_len_ratio += static_cast<double>(trg.size()) / static_cast<double>(src.size());
    *n_target_tokens += trg.size();
    if (use_null) {
      for (const unsigned f : trg) {
        s2t->Insert(kNULL, f);
      }
    }
    for (const unsigned e : src) {
      if (e >= insert_buffer.size()) {
        insert_buffer.resize(e + 1);
      }
      for (const unsigned f : trg) {
        insert_buffer[e].push_back(f);
      }
      insert_buffer_items += trg.size();
    }
    if (insert_buffer_items > opt.thread_buffer_size * 100) {
      insert_buffer_items = 0;
      AddTranslationOptions(insert_buffer, s2t);
    }
    ++size_counts_[make_pair<short, short>(trg.size(), src.size())];
  }
  for (const auto& p : size_counts_) {
    size_counts->push_back(p);
  }
  AddTranslationOptions(insert_buffer, s2t);

  opt.mean_srclen_multiplier = (*tot_len_ratio) / lc;
  if (log) {
    if (flag) *log << endl;
    *log << "expected target length = source length * " << opt.mean_srclen_multiplier << endl;
  }
}

int TrainIteration(TrainContext& ctx, TrainModelOpt& opt, int iter, int ITERATIONS) {
  ifstream in(opt.input.c_str());
  if (!in) {
    if (opt.warn) *opt.warn << "Can't read " << opt.input << endl;
    return 1;
  }
  TrainIteration(ctx, opt, in, iter, ITERATIONS);
  return 0;
}

void TrainIteration(TrainContext& ctx, TrainModelOpt& opt, std::istream& in, int iter, int ITERATIONS) {
  const bool final_iteration = (iter == (ITERATIONS - 1));
  if (opt.log) *opt.log << "ITERATION " << (iter + 1) << (final_iteration ? " (FINAL)" : "") << endl;
  double likelihood = 0;
  const double denom = ctx.n_target_tokens;
  int lc = 0;
  bool flag = false;
  string line;
  double c0 = 0;
  double emp_feat = 0;
  vector<string> buffer;
  vector<string> outputs;
  while (true) {
    getline(in, line);
    if (!in) break;
    ++lc;
    if (opt.log && lc % 1000 == 0) {
      *opt.log << '.';
      flag = true;
    }
    if (opt.log && lc % 50000 == 0) {
      *opt.log << " [" << lc << "]\n" << flush;
      flag = false;
    }
    buffer.push_back(line);

    if (buffer.size() >= opt.thread_buffer_size) {
      UpdateFromPairs(opt, buffer, lc, iter, final_iteration, ctx.use_null, ctx.kNULL,
                      ctx.prob_align_not_null, &c0, &emp_feat, &likelihood, &ctx.s2t, &outputs);
      if (final_iteration) {
        for (const string& output : outputs) {
          cout << output;
        }
      }
      buffer.clear();
    }
  }  // end data loop
  if (buffer.size() > 0) {
    UpdateFromPairs(opt, buffer, lc, iter, final_iteration, ctx.use_null, ctx.kNULL, ctx.prob_align_not_null,
                    &c0, &emp_feat, &likelihood, &ctx.s2t, &outputs);
    if (final_iteration) {
      for (const string& output : outputs) {
        cout << output;
      }
    }
    buffer.clear();
  }

  // log(e) = 1.0
  double base2_likelihood = likelihood / log(2);

  if (opt.log && flag) {
    *opt.log << endl;
  }
  emp_feat /= ctx.n_target_tokens;
  cerr << "  log_e likelihood: " << likelihood << endl;
  cerr << "  log_2 likelihood: " << base2_likelihood << endl;
  cerr << "     cross entropy: " << (-base2_likelihood / denom) << endl;
  cerr << "        perplexity: " << pow(2.0, -base2_likelihood / denom) << endl;
  cerr << "      posterior p0: " << c0 / ctx.n_target_tokens << endl;
  cerr << " posterior al-feat: " << emp_feat << endl;
  // cerr << "     model tension: " << mod_feat / toks << endl;
  cerr << "       size counts: " << ctx.size_counts.size() << endl;
  if (!final_iteration) {
    if (opt.favor_diagonal && opt.optimize_tension && iter > 0) {
      for (int ii = 0; ii < 8; ++ii) {
        double mod_feat = 0;
#pragma omp parallel for reduction(+ : mod_feat)
        for (size_t i = 0; i < ctx.size_counts.size(); ++i) {
          SizeCount const& sc = ctx.size_counts[i];
          const pair<short, short>& p = sc.first;
          for (short j = 1; j <= p.first; ++j)
            mod_feat += sc.second * DiagonalAlignment::ComputeDLogZ(j, p.first, p.second, opt.diagonal_tension);
        }
        mod_feat /= ctx.n_target_tokens;
        cerr << "  " << ii + 1 << "  model al-feat: " << mod_feat << " (tension=" << opt.diagonal_tension << ")\n";
        opt.diagonal_tension += (emp_feat - mod_feat) * 20.0;
        if (opt.diagonal_tension <= 0.1) opt.diagonal_tension = 0.1;
        if (opt.diagonal_tension > 14) opt.diagonal_tension = 14;
      }
      cerr << "     final tension: " << opt.diagonal_tension << endl;
    }
    if (opt.variational_bayes)
      ctx.s2t.NormalizeVB(opt.alpha);
    else
      ctx.s2t.Normalize();
  }
}

int TrainIterations(TrainContext& ctx, TrainModelOpt& opt, int ITERATIONS) {
  int rc = 0;
  for (int iter = 0; iter < ITERATIONS; ++iter)
    if ((rc = TrainIteration(ctx, opt, iter, ITERATIONS))) return rc;
  return rc;
}

void AlignContext::Init(AlignModelOpt const& opt, Dict& d) {
  this->d = &d;
  conditional_probability_filename = opt.conditional_probability_filename;
  use_null = !opt.no_null_word;
  prob_align_not_null = 1.0 - opt.prob_align_null;
  kNULL = d.Convert("<eps>");
  AlignModelOpt::operator=(opt);
}

int AlignContext::LoadProbsText() {
  if (!d) return 1;
  ifstream in(conditional_probability_filename.c_str());
  if (in) {
    s2t.DeserializeLogProbsFromText(&in, *d, log);
    return 0;
  }
  if (warn) *warn << "LoadProbsText ERROR - couldn't read " << conditional_probability_filename;
  return 1;
}

void TrainContext::Init(TrainModelOpt& opt) {
  AlignContext::Init(opt, opt.d);
  LogOpt::operator=(opt);
}

void InitialPass(TrainContext& ctx, TrainModelOpt& opt) {
  InitialPass(opt, ctx.kNULL, ctx.use_null, &ctx.s2t, &ctx.n_target_tokens, &ctx.tot_len_ratio, &ctx.size_counts);
}

int Train(TrainContext& ctx, TrainModelOpt& opt) {
  InitialPass(ctx, opt);
  ctx.s2t.Freeze();
  int rc = TrainIterations(ctx, opt, opt.ITERATIONS);
  if (!rc && !opt.conditional_probability_filename.empty()) {
    if (opt.log) *opt.log << "conditional probabilities: " << opt.conditional_probability_filename << endl;
    ctx.s2t.ExportToFile(opt.conditional_probability_filename.c_str(), opt.d, opt.beam_threshold);
  }
  return rc;
}

double Align(AlignContext const& ctx, std::ostream& out, std::string const& input) {
  istream* pin = &cin;
  std::shared_ptr<std::istream> holder;
  if (input != "-" && !input.empty()) holder.reset(pin = new ifstream(input.c_str()));
  return Align(ctx, *pin, out);
}

double Align(AlignContext const& ctx, std::istream& in, std::ostream& out) {
  string line;
  int lc = 0;
  double tlp = 0;
  while (getline(in, line)) {
    ++lc;
    double lp = AlignLine(ctx, line, lc, out);
    if (lp == HUGE_VAL) return lp;
    tlp += lp;
  }
  return tlp;
}

double LenLogProb(AlignContext const& ctx, unsigned nsrc, unsigned ntrg) {
  return Md::log_poisson(ntrg, 0.05 + nsrc * ctx.mean_srclen_multiplier);
}

double AlignedProb(AlignContext const& ctx, unsigned const* src, unsigned nsrc, unsigned const* trg,
                   unsigned ntrg, unsigned j, unsigned* srcPlus1) {
  unsigned f_j = trg[j];
  double sum = 0;
  int a_j = 0;
  double max_pat = 0;
  double prob_a_i = 1.0 / (nsrc + ctx.use_null);  // uniform (model 1)
  if (ctx.use_null) {
    if (ctx.favor_diagonal) prob_a_i = ctx.prob_align_null;
    max_pat = ctx.s2t.safe_prob(ctx.kNULL, f_j) * prob_a_i;
    sum += max_pat;
  }
  double az = 0;
  if (ctx.favor_diagonal)
    az = DiagonalAlignment::ComputeZ(j + 1, ntrg, nsrc, ctx.diagonal_tension) / ctx.prob_align_not_null;
  for (unsigned i = 1; i <= nsrc; ++i) {
    if (ctx.favor_diagonal)
      prob_a_i = DiagonalAlignment::UnnormalizedProb(j + 1, i, ntrg, nsrc, ctx.diagonal_tension) / az;
    double pat = ctx.s2t.safe_prob(src[i - 1], f_j) * prob_a_i;
    if (pat > max_pat) {
      max_pat = pat;
      a_j = i;
    }
    sum += pat;
  }
  *srcPlus1 = a_j;
  return sum;
}

double AlignLine(AlignContext const& ctx, std::string const& line, int lc, std::ostream& out) {
  vector<unsigned> src, trg;
  if (!ctx.d) return HUGE_VAL;
  Dict& d = *ctx.d;
  ParseLine(d, line, &src, &trg);
  for (auto s : src) out << d.Convert(s) << ' ';
  out << "|||";
  for (auto t : trg) out << ' ' << d.Convert(t);
  out << " |||";
  if (src.size() == 0 || trg.size() == 0) {
    if (ctx.warn) *ctx.warn << "Error in line " << lc << endl;
    return HUGE_VAL;
  }
#if 0
  if (ctx.is_reverse) swap(src, trg);
  double log_prob = Md::log_poisson(trg.size(), 0.05 + src.size() * ctx.mean_srclen_multiplier);

  // compute likelihood
  for (unsigned j = 0; j < trg.size(); ++j) {
    unsigned f_j = trg[j];
    double sum = 0;
    int a_j = 0;
    double max_pat = 0;
    double prob_a_i = 1.0 / (src.size() + ctx.use_null);  // uniform (model 1)
    if (ctx.use_null) {
      if (ctx.favor_diagonal) prob_a_i = ctx.prob_align_null;
      max_pat = ctx.s2t.safe_prob(ctx.kNULL, f_j) * prob_a_i;
      sum += max_pat;
    }
    double az = 0;
    if (ctx.favor_diagonal)
      az = DiagonalAlignment::ComputeZ(j + 1, trg.size(), src.size(), ctx.diagonal_tension)
           / ctx.prob_align_not_null;
    for (unsigned i = 1; i <= src.size(); ++i) {
      if (ctx.favor_diagonal)
        prob_a_i
            = DiagonalAlignment::UnnormalizedProb(j + 1, i, trg.size(), src.size(), ctx.diagonal_tension) / az;
      double pat = ctx.s2t.safe_prob(src[i - 1], f_j) * prob_a_i;
      if (pat > max_pat) {
        max_pat = pat;
        a_j = i;
      }
      sum += pat;
    }
    log_prob += log(sum);
    if (true) {
      if (a_j > 0) {
        out << ' ';
        if (ctx.is_reverse)
          out << j << '-' << (a_j - 1);
        else
          out << (a_j - 1) << '-' << j;
      }
    }
  }
#else
  unsigned nsrc, ntrg;
  unsigned const* srcs;
  unsigned const* trgs;
  if (ctx.is_reverse) {
    srcs = src.data();
    nsrc = src.size();
    trgs = trg.data();
    ntrg = trg.size();
  } else {
    srcs = trg.data();
    nsrc = trg.size();
    trgs = src.data();
    ntrg = src.size();
  }
  double log_prob = LenLogProb(ctx, nsrc, ntrg);
  for (unsigned j = 0; j < ntrg; ++j) {
    unsigned a_j;
    log_prob += log(AlignedProb(ctx, srcs, nsrc, trgs, ntrg, j, &a_j));
    if (a_j > 0) {
      unsigned i = a_j - 1;
      out << ' ';
      if (ctx.is_reverse)
        out << j << '-' << i;
      else
        out << i << '-' << j;
    }
  }
#endif
  out << " ||| " << log_prob << endl;
  return log_prob;
}

int Run(TrainModelOpt& opt) {
  if (opt.variational_bayes && opt.alpha <= 0.0) {
    *opt.warn << "--alpha must be > 0\n";
    return 1;
  }

  int rc;
  if (opt.force_align) {
    AlignContext ctx;
    ctx.Init(opt, opt.d);
    if ((rc = ctx.LoadProbsText())) return rc;
    double tlp = Align(ctx, std::cout, opt.input);
    if (tlp == HUGE_VAL) return 1;
    if (opt.log) *opt.log << "TOTAL LOG PROB " << tlp << endl;
  } else {
    TrainContext ctx;
    ctx.Init(opt);
    if ((rc = Train(ctx, opt))) return rc;
  }
  return 0;
}

}  // namespace FastAlign
