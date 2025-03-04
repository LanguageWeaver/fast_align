#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include <map>
#include <queue>
#include <set>
#if !defined(_MSC_VER) || defined(HAVE_GETOPT)
#include <getopt.h>
#endif

#include "alignment_io.h"
#include "atools.h"
#include "combine_align.h"

namespace FastAlign {

using namespace std;


bool InitCommandLine(AtoolsOpt& opt, int argc, char** argv) {
#if defined(_MSC_VER) && !defined(HAVE_GETOPT)
  return false;
#else
  static struct option aoptions[] = {{"input_1", required_argument, 0, 'i'},
                                     {"input_2", required_argument, 0, 'j'},
                                     {"command", required_argument, 0, 'c'},
                                     {0, 0, 0, 0}};
  while (1) {
    int oi;
    int c = getopt_long(argc, argv, "i:j:c:", aoptions, &oi);
    if (c == -1) break;
    switch (c) {
      case 'i': opt.input_1 = optarg; break;
      case 'j': opt.input_2 = optarg; break;
      case 'c': opt.command = optarg; break;
      default: return false;
    }
  }
  return opt.input_1.size() && opt.command.size();
#endif
}

struct Command {
  virtual ~Command() {}
  virtual string Name() const = 0;

  // returns 1 for alignment grid output [default]
  // returns 2 if Summary() should be called [for AER, etc]
  virtual int Result() const { return 1; }

  virtual bool RequiresTwoOperands() const { return true; }
  virtual void Apply(const Array2D<bool>& a, const Array2D<bool>& b, Array2D<bool>* x) = 0;
  void EnsureSize(const Array2D<bool>& a, const Array2D<bool>& b, Array2D<bool>* x) {
    x->resize(max(a.width(), b.width()), max(a.height(), b.height()));
  }
  static bool Safe(const Array2D<bool>& a, int i, int j) {
    if (i >= 0 && j >= 0 && i < static_cast<int>(a.width()) && j < static_cast<int>(a.height()))
      return a(i, j);
    else
      return false;
  }
  virtual void Summary() { assert(!"Summary should have been overridden"); }
};

// compute fmeasure, second alignment is reference, first is hyp
struct FMeasureCommand : public Command {
  FMeasureCommand() : matches(), num_predicted(), num_in_ref() {}
  int Result() const { return 2; }
  string Name() const { return "fmeasure"; }
  bool RequiresTwoOperands() const { return true; }
  void Apply(const Array2D<bool>& hyp, const Array2D<bool>& ref, Array2D<bool>* x) {
    (void)x;  // AER just computes statistics, not an alignment
    unsigned i_len = ref.width();
    unsigned j_len = ref.height();
    for (unsigned i = 0; i < i_len; ++i) {
      for (unsigned j = 0; j < j_len; ++j) {
        if (ref(i, j)) {
          ++num_in_ref;
          if (Safe(hyp, i, j)) ++matches;
        }
      }
    }
    for (unsigned i = 0; i < hyp.width(); ++i)
      for (unsigned j = 0; j < hyp.height(); ++j)
        if (hyp(i, j)) ++num_predicted;
  }
  void Summary() {
    if (num_predicted == 0 || num_in_ref == 0) {
      cerr << "Insufficient statistics to compute f-measure!\n";
      abort();
    }
    const double prec = static_cast<double>(matches) / num_predicted;
    const double rec = static_cast<double>(matches) / num_in_ref;
    cout << "P: " << prec << endl;
    cout << "R: " << rec << endl;
    const double f = (2.0 * prec * rec) / (rec + prec);
    cout << "F: " << f << endl;
  }
  int matches;
  int num_predicted;
  int num_in_ref;
};

struct DisplayCommand : public Command {
  string Name() const { return "display"; }
  bool RequiresTwoOperands() const { return false; }
  void Apply(const Array2D<bool>& in, const Array2D<bool>&, Array2D<bool>* x) {
    *x = in;
    cout << *x << endl;
  }
};

struct ConvertCommand : public Command {
  string Name() const { return "convert"; }
  bool RequiresTwoOperands() const { return false; }
  void Apply(const Array2D<bool>& in, const Array2D<bool>&, Array2D<bool>* x) { *x = in; }
};

struct InvertCommand : public Command {
  string Name() const { return "invert"; }
  bool RequiresTwoOperands() const { return false; }
  void Apply(const Array2D<bool>& in, const Array2D<bool>&, Array2D<bool>* x) {
    Array2D<bool>& res = *x;
    res.resize(in.height(), in.width());
    for (unsigned i = 0; i < in.height(); ++i)
      for (unsigned j = 0; j < in.width(); ++j) res(i, j) = in(j, i);
  }
};

struct IntersectCommand : public Command {
  string Name() const { return "intersect"; }
  bool RequiresTwoOperands() const { return true; }
  void Apply(const Array2D<bool>& a, const Array2D<bool>& b, Array2D<bool>* x) {
    EnsureSize(a, b, x);
    Array2D<bool>& res = *x;
    for (unsigned i = 0; i < a.width(); ++i)
      for (unsigned j = 0; j < a.height(); ++j) res(i, j) = Safe(a, i, j) && Safe(b, i, j);
  }
};

struct UnionCommand : public Command {
  string Name() const { return "union"; }
  bool RequiresTwoOperands() const { return true; }
  void Apply(const Array2D<bool>& a, const Array2D<bool>& b, Array2D<bool>* x) {
    EnsureSize(a, b, x);
    Array2D<bool>& res = *x;
    for (unsigned i = 0; i < res.width(); ++i)
      for (unsigned j = 0; j < res.height(); ++j) res(i, j) = Safe(a, i, j) || Safe(b, i, j);
  }
};

struct RefineCommand : public Command {
  RefineCommand() {
    neighbors_.push_back(make_pair(1, 0));
    neighbors_.push_back(make_pair(-1, 0));
    neighbors_.push_back(make_pair(0, 1));
    neighbors_.push_back(make_pair(0, -1));
  }
  bool RequiresTwoOperands() const { return true; }

  void Align(unsigned i, unsigned j) {
    res_(i, j) = true;
    is_i_aligned_[i] = true;
    is_j_aligned_[j] = true;
  }

  bool IsNeighborAligned(int i, int j) const {
    for (unsigned k = 0; k < neighbors_.size(); ++k) {
      const int di = neighbors_[k].first;
      const int dj = neighbors_[k].second;
      if (Safe(res_, i + di, j + dj)) return true;
    }
    return false;
  }

  bool IsNeitherAligned(int i, int j) const { return !(is_i_aligned_[i] || is_j_aligned_[j]); }

  bool IsOneOrBothUnaligned(int i, int j) const { return !(is_i_aligned_[i] && is_j_aligned_[j]); }

  bool KoehnAligned(int i, int j) const { return IsOneOrBothUnaligned(i, j) && IsNeighborAligned(i, j); }

  typedef bool (RefineCommand::*Predicate)(int i, int j) const;

 protected:
  void InitRefine(const Array2D<bool>& a, const Array2D<bool>& b) {
    res_.clear();
    EnsureSize(a, b, &res_);
    unsigned width = res_.width(), height = res_.height();
    ;
    un_.clear();
    un_.resize(width, height);
    is_i_aligned_.clear();
    is_j_aligned_.clear();
    is_i_aligned_.resize(width, false);
    is_j_aligned_.resize(height, false);
    for (unsigned i = 0; i < width; ++i)
      for (unsigned j = 0; j < height; ++j) {
        bool const aij = Safe(a, i, j), bij = Safe(b, i, j);
        un_(i, j) = aij || bij;
        if (aij && bij) Align(i, j);
      }
  }
  // "grow" the resulting alignment using the points in adds
  // if they match the constraints determined by pred
  void Grow(Predicate pred, bool idempotent, const Array2D<bool>& adds) {
    if (idempotent) {
      for (unsigned i = 0; i < adds.width(); ++i)
        for (unsigned j = 0; j < adds.height(); ++j) {
          if (adds(i, j) && !res_(i, j) && (this->*pred)(i, j)) Align(i, j);
        }
      return;
    }
    set<pair<int, int>> p;
    for (unsigned i = 0; i < adds.width(); ++i)
      for (unsigned j = 0; j < adds.height(); ++j)
        if (adds(i, j) && !res_(i, j)) p.insert(make_pair(i, j));
    bool keep_going = !p.empty();
    while (keep_going) {
      keep_going = false;
      set<pair<int, int>> added;
      for (set<pair<int, int>>::iterator pi = p.begin(); pi != p.end(); ++pi) {
        if ((this->*pred)(pi->first, pi->second)) {
          Align(pi->first, pi->second);
          added.insert(make_pair(pi->first, pi->second));
          keep_going = true;
        }
      }
      for (set<pair<int, int>>::iterator ai = added.begin(); ai != added.end(); ++ai) p.erase(*ai);
    }
  }
  Array2D<bool> res_;  // refined alignment
  Array2D<bool> un_;  // union alignment
  vector<bool> is_i_aligned_;
  vector<bool> is_j_aligned_;
  vector<pair<int, int>> neighbors_;
};

struct DiagCommand : public RefineCommand {
  DiagCommand() {
    neighbors_.push_back(make_pair(1, 1));
    neighbors_.push_back(make_pair(-1, 1));
    neighbors_.push_back(make_pair(1, -1));
    neighbors_.push_back(make_pair(-1, -1));
  }
};

struct GDCommand : public DiagCommand {
  string Name() const { return "grow-diag"; }
  void Apply(const Array2D<bool>& a, const Array2D<bool>& b, Array2D<bool>* x) {
    InitRefine(a, b);
    Grow(&RefineCommand::KoehnAligned, false, un_);
    *x = res_;
  }
};

struct GDFCommand : public DiagCommand {
  string Name() const { return "grow-diag-final"; }
  void Apply(const Array2D<bool>& a, const Array2D<bool>& b, Array2D<bool>* x) {
    InitRefine(a, b);
    Grow(&RefineCommand::KoehnAligned, false, un_);
    Grow(&RefineCommand::IsOneOrBothUnaligned, true, a);
    Grow(&RefineCommand::IsOneOrBothUnaligned, true, b);
    *x = res_;
  }
};

struct GDFACommand : public DiagCommand {
  string Name() const { return "grow-diag-final-and"; }
  void Apply(const Array2D<bool>& a, const Array2D<bool>& b, Array2D<bool>* x) {
    InitRefine(a, b);
    Grow(&RefineCommand::KoehnAligned, false, un_);
    Grow(&RefineCommand::IsNeitherAligned, true, a);
    Grow(&RefineCommand::IsNeitherAligned, true, b);
    *x = res_;
  }
};


template <class C>
AlignMatrix callCombine(AlignMatrix const& a, AlignMatrix const& b) {
  C c;
  AlignMatrix r;
  c.Apply(a, b, &r);
  return r;
}

template <class C>
void callCombine(AlignMatrix const& a, AlignMatrix const& b, AlignMatrix* r) {
  C c;
  c.Apply(a, b, r);
}

void combine(CombineAlignment combine, AlignMatrix const& a, AlignMatrix const& b, AlignMatrix* r) {
  switch (combine) {
    case kIntersect: return callCombine<IntersectCommand>(a, b, r);
    case kGrowDiag: return callCombine<GDCommand>(a, b, r);
    case kGrowDiagFinal: return callCombine<GDFCommand>(a, b, r);
    case kGrowDiagFinalAnd: return callCombine<GDFACommand>(a, b, r);
    default: return callCombine<UnionCommand>(a, b, r);
  }
}

AlignMatrix combine(CombineAlignment c, AlignMatrix const& a, AlignMatrix const& b) {
  AlignMatrix r;
  combine(c, a, b, &r);
  return r;
}

std::string alignText(AlignMatrix const& alignment) {
  std::string r;
  bool need_space = false;
  for (unsigned i = 0; i < alignment.width(); ++i)
    for (unsigned j = 0; j < alignment.height(); ++j)
      if (alignment(i, j)) {
        if (need_space)
          r.push_back(' ');
        else
          need_space = true;
        r += std::to_string(i);
        r.push_back('-');
        r += std::to_string(j);
      }
  return r;
}

typedef map<string, shared_ptr<Command>> CommandMap;

template <class C>
static void AddCommand(CommandMap& commands) {
  C* c = new C;
  commands[c->Name()].reset(c);
}
static CommandMap gCommands;
CommandMap& getCommands() {
  AddCommand<ConvertCommand>(gCommands);
  AddCommand<DisplayCommand>(gCommands);
  AddCommand<InvertCommand>(gCommands);
  AddCommand<IntersectCommand>(gCommands);
  AddCommand<UnionCommand>(gCommands);
  AddCommand<GDCommand>(gCommands);
  AddCommand<GDFCommand>(gCommands);
  AddCommand<GDFACommand>(gCommands);
  AddCommand<FMeasureCommand>(gCommands);
  return gCommands;
}

void AtoolsUsage(std::ostream& o, char const* exename) {
  static CommandMap& commands = getCommands();
  o << "Usage: " << exename << " -c COMMAND -i FILE1.AL [-j FILE2.AL]\n";
  o << "Valid options for COMMAND:";
  for (auto it : commands) o << ' ' << it.first;
  o << endl;
}

int Run(AtoolsOpt const& opt) {
  static CommandMap& commands = getCommands();
  if (commands.count(opt.command) == 0) {
    cerr << "Don't understand command: " << opt.command << endl;
    return 1;
  }
  if (commands[opt.command]->RequiresTwoOperands()) {
    if (opt.input_2.size() == 0) {
      cerr << "Command '" << opt.command << "' requires two alignment files\n";
      return 1;
    }
  } else {
    if (opt.input_2.size() != 0) {
      cerr << "Command '" << opt.command << "' requires only one alignment file\n";
      return 1;
    }
  }
  Command& cmd = *commands[opt.command];
  istream* in1 = NULL;
  if (opt.input_1 == "-") {
    in1 = &cin;
  } else {
    in1 = new ifstream(opt.input_1.c_str());
  }
  istream* in2 = NULL;
  if (cmd.RequiresTwoOperands()) {
    if (opt.input_2 == "-") {
      in2 = &cin;
    } else {
      in2 = new ifstream(opt.input_2.c_str());
    }
  }
  string line1;
  string line2;
  while (*in1) {
    getline(*in1, line1);
    if (in2) {
      getline(*in2, line2);
      if ((*in1 && !*in2) || (*in2 && !*in1)) {
        cerr << "Mismatched number of lines!\n";
        exit(1);
      }
    }
    if (line1.empty() && !*in1) break;
    shared_ptr<Array2D<bool>> out(new Array2D<bool>);
    shared_ptr<Array2D<bool>> a1 = AlignmentIO::ReadPharaohAlignmentGrid(line1);
    if (in2) {
      shared_ptr<Array2D<bool>> a2 = AlignmentIO::ReadPharaohAlignmentGrid(line2);
      cmd.Apply(*a1, *a2, out.get());
    } else {
      Array2D<bool> dummy;
      cmd.Apply(*a1, dummy, out.get());
    }

    if (cmd.Result() == 1) {
      AlignmentIO::SerializePharaohFormat(*out, &cout);
    }
  }
  if (cmd.Result() == 2) cmd.Summary();
  return 0;
}

}  // namespace FastAlign
