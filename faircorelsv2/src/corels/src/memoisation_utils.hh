#pragma once

#ifndef MEMOISATION_UTILS
#define MEMOISATION_UTILS

#include <unordered_map>
#include <vector>

// Types for memoisation
typedef std::tuple<int, int, int, int, int, int, int, int, int, int> mykey1_t;
typedef std::tuple<int, int, int, int, int, int, int, int> mykey2_t;

struct key1_equal : public std::binary_function<mykey1_t, mykey1_t, bool>
{
   bool operator()(const mykey1_t& v0, const mykey1_t& v1) const
   {
      return (
               std::get<0>(v0) == std::get<0>(v1) &&
               std::get<1>(v0) == std::get<1>(v1) &&
               std::get<2>(v0) == std::get<2>(v1) &&
               std::get<3>(v0) == std::get<3>(v1) &&
               std::get<4>(v0) == std::get<4>(v1) &&
               std::get<5>(v0) == std::get<5>(v1) &&
               std::get<6>(v0) == std::get<6>(v1) &&
               std::get<7>(v0) == std::get<7>(v1) &&
               std::get<8>(v0) == std::get<8>(v1) &&
               std::get<9>(v0) == std::get<9>(v1)
             ); 
   }
};

struct key2_equal : public std::binary_function<mykey2_t, mykey2_t, bool>
{
   bool operator()(const mykey2_t& v0, const mykey2_t& v1) const
   {
      return (
               std::get<0>(v0) == std::get<0>(v1) &&
               std::get<1>(v0) == std::get<1>(v1) &&
               std::get<2>(v0) == std::get<2>(v1) &&
               std::get<3>(v0) == std::get<3>(v1) &&
               std::get<4>(v0) == std::get<4>(v1) &&
               std::get<5>(v0) == std::get<5>(v1) &&
               std::get<6>(v0) == std::get<6>(v1) &&
               std::get<7>(v0) == std::get<7>(v1)
             ); 
   }
};


#endif