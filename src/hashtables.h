#ifndef FAST_ALIGN__SRC_HASHTABLES_H
#define FAST_ALIGN__SRC_HASHTABLES_H
#pragma once

#ifdef HAVE_SPARSEHASH
#include <google/sparse_hash_map>
namespace FastAlign {
typedef google::sparse_hash_map<std::string, unsigned, std::hash<std::string> > MAP_TYPE;
typedef google::sparse_hash_map<unsigned, double> Word2Double;
}
#else
#include <unordered_map>
namespace FastAlign {
typedef std::unordered_map<std::string, unsigned, std::hash<std::string> > MAP_TYPE;
typedef std::unordered_map<unsigned, double> Word2Double;
}
#endif


#endif
