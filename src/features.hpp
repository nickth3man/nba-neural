// Feature engineering: era labels, age, imputation, scaling, target, splits.
#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "frame.hpp"

namespace era {

// Row-major feature matrix handed to the model.
struct Matrix {
    std::vector<float> data;
    int64_t rows = 0;
    int64_t cols = 0;

    float at(int64_t r, int64_t c) const { return data[r * cols + c]; }
};

struct Splits {
    std::vector<int64_t> train;
    std::vector<int64_t> val;
    std::vector<int64_t> test;
};

struct Scaler {
    std::vector<double> mean;
    std::vector<double> scale;
};

// '2002-03' -> '2000s'.
std::string decade_of(const std::string& season_year);

// Age at season start = season_end_year - birth_year; decade-mean fallback.
std::vector<double> compute_age(const Frame& df);

// Per-season z-score of a box-score composite.
//
// raw = pts + reb + ast + (stl + blk - tov if recorded else 0)
// Then z-scored within each season_year so the target is era-normalized.
std::vector<double> build_value_target(const Frame& df);

// One-hot of position, with anything outside G/F/C folded into Unknown.
std::map<std::string, std::vector<double>> build_position_oh(const Frame& df);

// Adds era, age, position one-hot and value_target; returns a copy.
Frame engineer_features(const Frame& df);

// Stratified shuffle split: each class is shuffled independently and the
// requested fraction is held out, so class proportions are preserved.
// sklearn's train_test_split draws from numpy's RNG, so the rows it picks differ.
std::pair<std::vector<int64_t>, std::vector<int64_t>> stratified_split(
    const std::vector<int64_t>& indices, const std::vector<std::string>& strata, double test_frac,
    std::mt19937_64& rng);

// Stratified train/val/test split by era. Returns positional indices.
Splits make_splits(const Frame& df);

std::vector<std::string> feature_columns(const Frame& df);

Scaler fit_scaler(const Frame& df, const std::vector<int64_t>& idx_train,
                  const std::vector<std::string>& cols);

Matrix transform_features(const Frame& df, const Scaler& scaler,
                          const std::vector<std::string>& cols);

// Era label -> index into ERA_ORDER.
std::vector<int64_t> era_codes(const Frame& df);

}  // namespace era
